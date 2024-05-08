import glob
import gunpowder as gp
import logging
import math
import numpy as np
import os
import operator
import sys
import zarr
import torch
from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from scipy.ndimage import binary_dilation, generate_binary_structure
#from skimage.transform import resize

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

samples = glob.glob(base_dir + '/*.zarr')
target_vox = (36,16,16) #(24,9,9) #tens of nanometers -- this is the "middle" size of the 3
voxel_size = gp.Coordinate(target_vox)

input_size = gp.Coordinate((1056, 1548, 1548))
output_size = gp.Coordinate((576, 720, 720))


batch_size = 1

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]
run_name = dt+'_3d_affs_IntWgt_b1_scaled_down' 

def calc_max_padding(
        output_size,
        voxel_size,
        mode='shrink'):

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = gp.Roi(
            (gp.Coordinate(
                [i/2 for i in [output_size[0], diag, diag]]) +
                voxel_size),
            (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin()

class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, prediction, target, weights):
        
        scaled = weights * (prediction - target) ** 2
        
        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)
        else:
            loss = torch.mean(scaled)
        
        return loss

#class ReScale(gp.BatchFilter):
#    def __init__(
#        self,
#        array,
#        scale_factor=(1.0, 1.0, 1.0),
#        order = 1,
#        preserve_range = True,
#        anti_aliasing = True,
#    ):
#        self.array = array
#        self.scale_factor = scale_factor
#        self.order = order
#        self.preserve_range = preserve_range
#        self.anti_aliasing = anti_aliasing
#
#    def process(self, batch, request):
#        voxel_size = self.spec[self.array].voxel_size
#        data_roi = request[self.array].roi
#        data_roi_vox = data_roi / voxel_size
#
#        out_shape = np.asarray(data_roi_vox.get_shape())
#
#        source_shape = np.ceil(out_shape / np.asarray(self.scale_factor))
#        context      = tuple(np.ceil(np.ceil(source_shape - out_shape)/2)*np.asarray(voxel_size))
#        source_roi = data_roi.grow(context, context)
#        data = batch[self.array].crop(source_roi)
#
#        temp_shape = np.ceil(
#                np.asarray((source_roi / voxel_size).get_shape()) *
#                np.asarray(self.scale_factor))
#        scaleddata = resize(data.data,
#                tuple(temp_shape),
#                order=self.order,
#                preserve_range=self.preserve_range,
#                anti_aliasing=self.anti_aliasing
#                ).astype(data.data.dtype)
#
#        context = out_shape - temp_shape
#        scaleddata = cropND(scaleddata, tuple(out_shape))
#        batch[self.array].data = scaleddata

class CreateMask(BatchFilter):

    def __init__(
            self,
            in_labels,
            out_mask,
            dtype=np.uint8,
            dilate_pix=0,
            dilate_struct=None
            ):
        self.in_labels = in_labels
        self.out_mask = out_mask
        self.dtype = dtype
        self.dilate_pix = dilate_pix
        self.dilate_struct = dilate_struct

    def setup(self):

        self.provides(
            self.out_mask,
            self.spec[self.in_labels].copy())

    def prepare(self, request):

        deps = BatchRequest()
        deps[self.in_labels] = request[self.out_mask].copy()

        return deps

    def process(self, batch, request):

        outputs = gp.Batch()

        labels_data = batch[self.in_labels].data
        mask_data = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.in_labels].spec.copy()
        spec.roi = request[self.out_mask].roi.copy()
        spec.dtype = self.dtype

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:
            mask_data = (labels_data>0).astype(self.dtype)

        if self.dilate_pix >0:
            mask_data = binary_dilation(
                    mask_data,
                    structure = self.dilate_struct,
                    iterations = self.dilate_pix).astype(self.dtype)

        outputs[self.out_mask] = gp.Array(mask_data, spec)

        return outputs

class BalanceLabelsWithIntensity(gp.BatchFilter):

    def __init__(
            self,
            raw,
            labels,
            scales,
            num_bins = 10,
            bin_min = 0.,
            bin_max = 1.,
            ):

        self.labels = labels
        self.raw = raw
        self.scales = scales
        self.bins = np.linspace(bin_min, bin_max, num=num_bins+1)

    def setup(self):

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.labels] = request[self.scales]
        deps[self.raw] = request[self.raw]

        return deps

    def process(self, batch, request):

        labels = batch.arrays[self.labels]

        raw = batch.arrays[self.raw]

        cropped_roi = labels.spec.roi.get_shape()
        vox_size = labels.spec.voxel_size
        raw_cropped = self.__cropND(raw.data, cropped_roi/voxel_size)

        raw_cropped = np.expand_dims(raw_cropped, 0)
        raw_cropped = np.repeat(raw_cropped, 3, 0)

        num_classes = len(self.bins) + 2

        binned = np.digitize(raw_cropped, self.bins) + 1
        # add one so values fall from 1 to num_bins+1 (if outside range 0-1)

        # set foreground labels to be class 0:
        binned[labels.data>0] = 0

        # initialize scales:
        scale_data = np.ones(labels.data.shape, dtype=np.float32)

        # calculate the fraction of per-class samples:
        classes, counts = np.unique(binned, return_counts=True)
        fracs = counts.astype(float) / scale_data.sum()

        # prevent background from being weighted over foreground:
        fg = np.where(classes==0)[0]
        if len(fg)>0:
            fracs = np.clip(fracs, fracs[fg], None)

        # calculate the class weights
        w_sparse = 1.0 / float(num_classes) / fracs
        w = np.zeros(num_classes)
        w[classes] = w_sparse

        # scale the scale with the class weights
        scale_data *= np.take(w, binned)

        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi
        batch.arrays[self.scales] = gp.Array(scale_data, spec)

    def __cropND(self, img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

def train(iterations):
    
    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
    raw_orig = gp.ArrayKey("RAW_ORIG")
    labels_orig = gp.ArrayKey("LABELS_ORIG")
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    aff_mask = gp.ArrayKey("AFF_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
    request.add(aff_mask, output_size)
    request.add(mask_weights, output_size)
    
    in_channels = 1
    num_fmaps=12
    num_fmaps_out = 14
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 3, [[1,]*3], activation='Sigmoid'),
            ) 

    torch.cuda.set_device(0)

    loss = WeightedMSELoss() 
    optimizer = torch.optim.Adam(lr=5e-5, 
            params=model.parameters(),
            #betas=(0.95, 0.999)
            )

    padding = calc_max_padding(output_size, voxel_size)

    def create_source(sample):
        if "airyscan" in sample:
            vox = (16,4,4)
        elif "spinning" in sample:
            vox = (24, 9, 9)
        else: #"confocal"
            vox = (36,16,16)

        vox = gp.Coordinate(vox)

        source = gp.ZarrSource(
                sample,
                datasets={
                    raw_orig:'3d/raw',
                    labels_orig:'3d/labeled',
                },
                array_specs={
                    raw_orig:gp.ArraySpec(interpolatable=True, voxel_size=vox),
                    labels_orig:gp.ArraySpec(interpolatable=False, voxel_size=vox),
                })
        
        source +=  gp.Normalize(raw_orig) 
        source += gp.Resample(raw_orig, target_vox, raw, ndim=3)
        source += gp.Resample(labels_orig, target_vox, labels, ndim=3)

        source +=  gp.Pad(raw, None)
        source +=  gp.Pad(labels, padding)
        source +=  gp.RandomLocation()

        #if "confocal" in sample:
        #    source += ReScale(raw, (1.5, 2.5, 2.5))
        #    source += ReScale(labels, (1.5, 2.5, 2.5), order=0, anti-aliasing=False)
        #elif "spinningdisk" in sample:
        #    source += ReScale(raw, (1., 1.8, 1.8))
        #    source += ReScale(labels, (1., 1.8, 1.8), order=0, anti_aliasing=False)

        source += gp.Reject_CMM(mask=labels, 
                    min_masked=0.005, 
                    min_min_masked=0.001, 
                    reject_probability=0.95) 

        return source

    sources = tuple(
            create_source(sample)
            for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2.,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))
    
    pipeline += gp.NoiseAugment(raw, var=0.01)
    
    pipeline += gp.AddAffinities(
            affinity_neighborhood=neighborhood, 
            labels=labels,
            affinities=gt, 
            dtype=np.float32)
    
    dilate_struct = generate_binary_structure(4,1).astype(np.uint8)
    dilate_struct[0,:,:,:] = 0
    dilate_struct[2,:,:,:] = 0

    pipeline += CreateMask(gt, aff_mask, dilate_pix=1, dilate_struct=dilate_struct)

    pipeline += BalanceLabelsWithIntensity(
            raw,
            aff_mask,
            mask_weights,
            )
    
    # raw: h,w
    # labels: h,w

    pipeline += gp.Unsqueeze([raw]) #, gt_fg])

    # raw: c,h,w
    # labels: c,h,w

    pipeline += gp.Stack(batch_size)

    # raw: b,c,h,w
    # labels: b,c,h,w

    pipeline += gp.PreCache(
            cache_size=40,
            num_workers=10)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred
        },
        loss_inputs={
            0: pred,
            1: gt,
            2: mask_weights
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name, 
        save_every=5000)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # pred_mask: b,c,h,w
    pipeline += gp.Squeeze([raw], axis=1)

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                aff_mask: "aff_mask",
                gt: "gt", 
                pred: "pred",
                mask_weights: "mask_weights",
            },
            every=250)
    
    pipeline += PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10001)
