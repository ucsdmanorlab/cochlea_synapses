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
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from torch.nn.functional import binary_cross_entropy

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

zarrlist = glob.glob(base_dir + '/*.zarr')
voxel_size = gp.Coordinate((1,)*2)

input_size = gp.Coordinate((140,)*2) * voxel_size
output_size = gp.Coordinate((100,)*2) * voxel_size

samples = {}
for i in range(len(zarrlist)):
    samples[zarrlist[i]] = len(glob.glob(zarrlist[i]+'/2p5d/raw/*'))

batch_size = 128

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]

run_name = dt+'_2p5d_affs_intWgt_1e-5_lessAug'
#run_name = '2023-12-20_16-42-00_2p5d_affs_intWgt'

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

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):
        
        scaled = (weights * (prediction - target) ** 2)

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

class CreateMask(gp.BatchFilter):

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

        deps = gp.BatchRequest()
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
        
        cropped_roi = labels.data.shape #spec.roi.get_shape()
        
        #raw_cropped = self.__cropND(raw.data, cropped_roi/voxel_size)

        num_classes = len(self.bins) + 2

        binned = np.digitize(raw.data, self.bins) + 1
        binned = self.__cropND(binned, cropped_roi)
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
        out_dims = bounding
        while img.ndim > len(bounding):
            bounding = np.append(1, bounding)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices].reshape(out_dims)

def train(iterations):
    
    neighborhood = [[-1,0],[0,-1]]

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
    request.add(mask_weights, output_size)
    
    num_fmaps=30

    ds_fact = [(2,2), (2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3),(3,3)]]*num_levels
    ksu = [[(3,3),(3,3)]]*(num_levels-1)
    in_channels = 5

    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 2, [[1,]*2], activation='Sigmoid'),
            ) 

    torch.cuda.set_device(1)

    loss = WeightedMSELoss() 
    optimizer = torch.optim.Adam(lr=1e-5, 
            params=model.parameters(),
            betas=(0.95, 0.999))

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:f'2p5d/raw/{i}',
                    labels:f'2p5d/labeled/{i}',
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    labels:gp.ArraySpec(interpolatable=False),
                }) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Normalize(raw) +
                gp.RandomLocation() +
                gp.Reject_CMM(mask=labels, min_masked=0.01, min_min_masked=0.001, reject_probability=0.9)
                for sample, num_sections in samples.items()
                for i in range(num_sections))

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()

    pipeline += gp.IntensityAugment(raw, 0.8, 1.2, -0.1, 0.1)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(64,)*2,
                    jitter_sigma=(2,)*2,
                    spatial_dims=2,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))
    
    pipeline += gp.NoiseAugment(raw, var=0.01)
    
    pipeline += gp.AddAffinities(
            affinity_neighborhood=neighborhood, 
            labels=labels,
            affinities=gt, 
            labels_mask=None, 
            affinities_mask=None,
            dtype=np.float32)
    
    pipeline += BalanceLabelsWithIntensity(
            raw,
            gt,
            mask_weights,
            )

    # raw: c,h,w
    # gt: c,h,w
    # mask_weights: c,h,w

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
        save_every=1000)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # pred_mask: b,c,h,w
    #pipeline += gp.Squeeze([raw], axis=1)
    #pipeline += gp.Squeeze([raw, gt, pred])

    
    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt: "gt", 
                pred: "pred",
                mask_weights: "mask_weights",
            },
            every=500)
    
    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10001)
