import glob
import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import zarr
import torch
from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from scipy.ndimage import binary_erosion, binary_dilation
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from torch.nn.functional import binary_cross_entropy
from skimage.transform import resize

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

zarrlist = glob.glob(base_dir + '/*.zarr')
voxel_size = gp.Coordinate((1,)*2)

input_size = gp.Coordinate((140,)*2) * voxel_size
output_size = gp.Coordinate((100,)*2) * voxel_size

samples = {}
for i in range(len(zarrlist)):
    samples[zarrlist[i]] = len(glob.glob(zarrlist[i]+'/2d/raw/*'))

batch_size = 256

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]

run_name = dt+'_scaledDOWN_twoclass_wgtsurr3px'

def calc_max_padding(
        output_size,
        voxel_size):

    diag = np.sqrt(output_size[0]**2 + output_size[1]**2)

    max_padding = np.ceil(
            [i/2 + j for i,j in zip(
                [output_size[0], diag, diag],
                list(voxel_size)
                )]
            )

    return gp.Coordinate(max_padding)

class WeightedBCELoss(torch.nn.BCELoss):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, prediction, target, weights):
        loss = binary_cross_entropy(prediction, target, weight=weights)
        return loss

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(lambda a, da: a+da, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

class UpSample(BatchFilter):
    def __init__(
        self,
        array,
        scale_factor=1,
    ):
        self.array = array
        self.scale_factor = scale_factor
    
    def setup(self):
        spec = self.spec[self.array].copy()
        if not isinstance(self.scale_factor, tuple):
            self.scale_factor = (self.scale_factor,)*spec.roi.dims()
    
    def prepare(self, request):

        roi = request[self.array].roi
        voxel_size = self.spec[self.array].voxel_size

        roi_vox = roi / voxel_size
        roi_vox_shape = np.asarray(roi_vox.get_shape())

        source_vox = np.ceil(roi_vox_shape / np.asarray(self.scale_factor))
        context      = tuple(np.ceil(np.ceil(source_vox - roi_vox_shape)/2)*np.asarray(voxel_size))

        source_roi = roi.grow(context, context)

        deps = gp.BatchRequest()
        deps[self.array] = source_roi

        return deps

    #    deps = BatchRequest()

    #    logger.debug("preparing upsampling of " + str(self.source))
    #    
    #    request_roi = request[self.array].roi
    #    logger.debug("request ROI is %s"%request_roi)

    #    deps[self.array] = ArraySpec(roi=request_roi)

    #    return deps

    def process(self, batch, request):
        #voxel_size = self.spec[self.array].voxel_size
        #data_roi = request[self.array].roi
        #data_roi_vox = data_roi / voxel_size

        #out_shape = np.asarray(data_roi_vox.get_shape())

        #source_shape = np.ceil(out_shape / np.asarray(self.scale_factor))
        #context      = tuple(np.ceil(np.ceil(source_shape - out_shape)/2)*np.asarray(voxel_size))
        #source_roi = data_roi.grow(context, context)
        data = batch[self.array].data #crop(source_roi).data
        for d, f in enumerate(self.scale_factor):
            data = np.repeat(data, f, axis=d)
        
        batch[self.array].data = data
        batch[self.array] = batch[self.array].crop(request[self.array].roi)
        #data = cropND(data, tuple(out_shape))

        #batch[self.array].data = data 


class ReScale(BatchFilter):
    def __init__(
        self,
        array,
        scale_factor=(1.0, 1.0, 1.0),
        order = 1,
        preserve_range = True,
        anti_aliasing = True,
    ):
        self.array = array
        self.scale_factor = scale_factor
        self.order = order
        self.preserve_range = preserve_range
        self.anti_aliasing = anti_aliasing

    def process(self, batch, request):

        voxel_size = self.spec[self.array].voxel_size
        data_roi = request[self.array].roi
        data_roi_vox = data_roi / voxel_size

        out_shape = np.asarray(data_roi_vox.get_shape())

        source_shape = np.ceil(out_shape / np.asarray(self.scale_factor))
        context      = tuple(np.ceil(np.ceil(source_shape - out_shape)/2)*np.asarray(voxel_size))
        source_roi = data_roi.grow(context, context)
        data = batch[self.array].crop(source_roi)

        temp_shape = np.ceil(
                np.asarray((source_roi / voxel_size).get_shape()) *
                np.asarray(self.scale_factor))
        scaleddata = resize(data.data,
                tuple(temp_shape),
                order=self.order,
                preserve_range=self.preserve_range,
                anti_aliasing=self.anti_aliasing
                ).astype(data.data.dtype)

        context = out_shape - temp_shape
        scaleddata = cropND(scaleddata, tuple(out_shape))
        batch[self.array].data = scaleddata

class ComputeMask(BatchFilter):

    def __init__(
            self,
            labels,
            mask,
            surr_mask = None,
            erode_iterations=None,
            surr_pix=3,
            dtype=np.uint8,
            ):

        self.labels = labels
        self.mask = mask
        self.surr_mask = surr_mask
        self.erode_iterations = erode_iterations
        self.surr_pix = surr_pix
        self.dtype = dtype

    def setup(self):

        maskspec = self.spec[self.labels].copy()
        self.provides(self.mask, maskspec)
        
        if self.surr_mask:
            surrspec = self.spec[self.labels].copy()
            self.provides(self.surr_mask, surrspec)

    def prepare(self, request):
        
        deps = BatchRequest()
        deps[self.labels] = request[self.mask].copy()
       
        return deps

    def _compute_mask(self, label_data):
        
        fg = np.zeros_like(label_data, dtype=bool)
        surr = np.zeros_like(label_data, dtype=bool)

        for label in np.unique(label_data):
            if label == 0:
                continue
            label_mask = label_data == label

            if self.erode_iterations:
                label_mask = binary_erosion(label_mask, iterations = self.erode_iterations)

            fg = np.logical_or(label_mask, fg)
        
        if self.surr_mask:
            surr = binary_dilation(label_data>0, iterations = self.surr_pix)
            surr = np.logical_xor(surr, fg)

        return fg.astype(self.dtype), surr.astype(self.dtype)

    def process(self, batch, request):

        outputs = Batch()

        labels_data = batch[self.labels].data
        mask_data = np.zeros_like(labels_data).astype(self.dtype)
        surr_data = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.mask].roi.copy()
        spec.dtype = self.dtype

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:
            mask_data, surr_data = self._compute_mask(labels_data)
            surr_data[surr_data !=0] = 2
            surr_data = surr_data + mask_data
        outputs[self.mask] =  Array(mask_data, spec)
        
        if self.surr_mask:
            outputs[self.surr_mask] = Array(surr_data, spec)

        return outputs

def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    gt = gp.ArrayKey("GT")
    surr_mask = gp.ArrayKey("SURR_MASK")
    pred_mask = gp.ArrayKey("PRED_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(pred_mask, output_size)
    request.add(mask_weights, output_size)
    request.add(surr_mask, output_size)

    num_fmaps=30

    ds_fact = [(2,2), (2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3),(3,3)]]*num_levels
    ksu = [[(3,3),(3,3)]]*(num_levels-1)

    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*2], activation=None),
            torch.nn.Sigmoid()
            ) #'Sigmoid'))

    torch.cuda.set_device(1)

    loss = WeightedBCELoss() #torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    def create_source(sample, i):
        
        #scale = 1

        #if "confocal" in sample:
        #    scale = 2
        
        source = gp.ZarrSource(
                sample,
                datasets={
                    raw:f'2d/raw/{i}',
                    labels:f'2d/labeled/{i}',
                    labels_mask:f'2d/mask/{i}'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    labels:gp.ArraySpec(interpolatable=False),
                    labels_mask:gp.ArraySpec(interpolatable=False)
                })
        source += gp.Squeeze([raw,labels,labels_mask])

        source += gp.Pad(raw, None) 
        source += gp.Pad(labels, padding) 
        source += gp.Pad(labels_mask, padding) 
        source += gp.Normalize(raw) 
        source += gp.RandomLocation(mask=labels_mask) 
        if "airyscan" in sample:
            source += ReScale(raw, (0.5, 0.5))
            source += ReScale(labels, (0.5, 0.5), order=0, anti_aliasing=False)
            source += ReScale(labels_mask, (0.5, 0.5), order=0, anti_aliasing=False)
        source += gp.Reject(mask=labels_mask, min_masked=0.05, reject_probability=0.95)
        
        return source

    sources = tuple(
            create_source(sample, i)
            for sample, num_sections in samples.items()
            for i in range(num_sections))

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*2,
                    jitter_sigma=(2,)*2,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))
    
    pipeline += NoiseAugment(raw, var=0.01)

    pipeline += ComputeMask(
            labels,
            gt,
            surr_mask,
            erode_iterations=1,
            dtype=np.float32,
            )

    pipeline += BalanceLabels(
            surr_mask,
            mask_weights,
            num_classes = 3,
            clipmin = 0.005,
            )

    # raw: h,w
    # labels: h,w
    # labels_mask: h,w

    pipeline += gp.Unsqueeze([raw, gt, mask_weights]) #, gt_fg])

    # raw: c,h,w
    # labels: c,h,w
    # labels_mask: c,h,w

    pipeline += gp.Stack(batch_size)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w

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
            0: pred_mask
        },
        loss_inputs={
            0: pred_mask,
            1: gt,
            2: mask_weights
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name,
        save_every=500)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w
    
    pipeline += gp.Squeeze([raw, gt, pred_mask, mask_weights], axis=1)

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                labels_mask: "labels_mask",
                gt: "gt", 
                pred_mask: "pred_mask",
                mask_weights: "mask_weights",
            },
            every=250)
    
    pipeline += PrintProfilingStats(every=10)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10000)
