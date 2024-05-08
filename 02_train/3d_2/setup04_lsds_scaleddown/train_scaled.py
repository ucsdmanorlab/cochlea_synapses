import glob
import gunpowder as gp
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr

from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from skimage.transform import resize

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

samples = glob.glob(base_dir + '/*.zarr')

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 1

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]
run_name = dt+'_3d_scaled_affs_lr1e-5_constup' #"2022-11-21_16-37-373d_affs" #dt+'3d_affs'

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
        
        scaled = (weights * (prediction - target) ** 2)

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(lambda a, da: a+da, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

class ReScale(gp.BatchFilter):
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

class CreateMask(gp.BatchFilter):

    def __init__(self, in_array, out_array):
        self.in_array = in_array
        self.out_array = out_array

    def setup(self):

        self.provides(
            self.out_array,
            self.spec[self.in_array].copy())

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.in_array] = request[self.out_array].copy()

        return deps

    def process(self, batch, request):

        data = np.ones_like(batch[self.in_array].data).astype(np.uint8)

        spec = batch[self.in_array].spec.copy()
        spec.roi = request[self.out_array].roi.copy()
        spec.dtype = np.uint8

        batch = gp.Batch()

        batch[self.out_array] = gp.Array(data, spec)

        return batch

def train(iterations):
    
    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    #labels_mask = gp.ArrayKey("LABELS_MASK")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    blank_mask = gp.ArrayKey("BLANK_MASK")
    aff_mask = gp.ArrayKey("AFF_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    #request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
    request.add(aff_mask, output_size)
    request.add(mask_weights, output_size)
    request.add(blank_mask, output_size)
    
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
    optimizer = torch.optim.Adam(lr=1e-5, 
            params=model.parameters(),
            betas=(0.95, 0.999))

    padding = calc_max_padding(output_size, voxel_size)
    
    def create_source(sample):
        source = gp.ZarrSource(
                sample,
                datasets={
                    raw:f'3d/raw',
                    labels:f'3d/labeled',
                    #labels_mask:f'3d/mask'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                    labels:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
                    #labels_mask:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)
                })

        source += gp.Normalize(raw) 
        source += gp.Pad(raw, None) 
        #source += gp.Pad(labels_mask, padding) 
        source += gp.Pad(labels, padding) 
        source += gp.RandomLocation()
        if "confocal" in sample:
            source += ReScale(raw, (1.5, 2.5, 2.5))
            source += ReScale(labels, (1.5, 2.5, 2.5), order=0, anti_aliasing=False)
            #source += ReScale(labels_mask, (1.5, 2.5, 2.5), order=0, anti_aliasing=False)
        if "spinningdisk" in sample:
            source += ReScale(raw, (1., 1.8, 1.8))
            source += ReScale(labels, (1., 1.8, 1.8), order=0, anti_aliasing=False)
            #source += ReScale(labels_mask, (1., 1.8, 1.8), order=0, anti_aliasing=False)

        source += gp.Reject(mask=labels, min_masked=0.0001)#, reject_probability=0.9)

        return source
    
    sources = tuple(
            create_source(sample)
            for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2, z_section_wise=False)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))
    
    pipeline += gp.NoiseAugment(raw, var=0.01)
    
    pipeline += CreateMask(labels, blank_mask)

    pipeline += gp.AddAffinities(
            affinity_neighborhood=neighborhood, 
            labels=labels,
            affinities=gt, 
            labels_mask=blank_mask, 
            affinities_mask=aff_mask,
            dtype=np.float32)
    
    pipeline += gp.BalanceLabels(
            gt,
            mask_weights,
            )
    
    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    # raw: h,w
    # labels: h,w
    # labels_mask: h,w

    pipeline += gp.Unsqueeze([raw]) #, gt_fg])

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
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w
    pipeline += gp.Squeeze([raw], axis=1)
    pipeline += gp.Squeeze([raw, gt, pred, labels, mask_weights]) #labels_mask

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    
    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                #labels_mask: "labels_mask",
                gt: "gt", 
                pred: "pred",
                mask_weights: "mask_weights",
            },
            every=250)
    
    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(20001)
