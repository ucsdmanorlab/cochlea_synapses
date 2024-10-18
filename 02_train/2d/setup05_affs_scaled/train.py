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
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
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

batch_size = 128

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]

run_name = dt+'_affs_scaled'

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
        
        weights = torch.tanh(weights/5)*5
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

class CreateMask(BatchFilter):

    def __init__(self, in_array, out_array):
        self.in_array = in_array
        self.out_array = out_array

    def setup(self):

        self.provides(
            self.out_array,
            self.spec[self.in_array].copy())

    def prepare(self, request):

        deps = BatchRequest()
        deps[self.in_array] = request[self.out_array].copy()

        return deps

    def process(self, batch, request):

        data = np.ones_like(batch[self.in_array].data).astype(np.uint8)

        spec = batch[self.in_array].spec.copy()
        spec.roi = request[self.out_array].roi.copy()
        spec.dtype = np.uint8

        batch = Batch()

        batch[self.out_array] = Array(data, spec)

        return batch

def train(iterations):
    
    neighborhood = [[-1,0],[0,-1]]

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    blank_mask = gp.ArrayKey("BLANK_MASK")
    aff_mask = gp.ArrayKey("AFF_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
    request.add(aff_mask, output_size)
    request.add(mask_weights, output_size)
    request.add(blank_mask, output_size)
    
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
            ConvPass(num_fmaps, 2, [[1,]*2], activation='Sigmoid'),
            ) 

    torch.cuda.set_device(1)

    loss = WeightedMSELoss() 
    optimizer = torch.optim.Adam(lr=1e-5, 
            params=model.parameters(),
            betas=(0.95, 0.999))

    padding = calc_max_padding(output_size, voxel_size)
    
    def create_source(sample, i):

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
        if "confocal" in sample:
            source += ReScale(raw, (2.0, 2.0))
            source += ReScale(labels, (2.0, 2.0), order=0, anti_aliasing=False)
            source += ReScale(labels_mask, (2.0, 2.0), order=0, anti_aliasing=False)
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
    #pipeline += gp.Squeeze([raw, gt, pred])

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    
    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                labels_mask: "labels_mask",
                gt: "gt", 
                pred: "pred",
                mask_weights: "mask_weights",
            },
            every=300)
    
    pipeline += PrintProfilingStats(every=10)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10000)
