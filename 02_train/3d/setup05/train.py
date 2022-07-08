import glob
import json
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr
import rescale_cmm

from funlib.learn.torch.models import UNet,ConvPass
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *

from scipy.ndimage import distance_transform_edt, \
        binary_erosion, binary_dilation

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../../01_data/zarrs/ctbp2/restest/training'

samples = os.listdir(data_dir)

batch_size = 1


def calc_max_padding(
        output_size,
        voxel_size,
        mode='shrink'):

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = Roi(
            (Coordinate(
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


def train_until(max_iteration):

    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]

    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 3, [[1]*3], activation='Sigmoid'))

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-5,
            betas=(0.95,0.999))

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    foreground_mask = ArrayKey('FOREGROUND_MASK')
    labels_mask = ArrayKey('LABELS_MASK')
    gt_affs = ArrayKey('GT_AFFS')
    pred_affs = ArrayKey('PRED_AFFS')
    affs_mask = ArrayKey('AFFS_MASK')
    affs_weights = ArrayKey('AFFS_WEIGHTS')

    input_shape = Coordinate((84,156,156))
    output_shape = Coordinate((52,64,64))

    voxel_size = Coordinate((4,1,1))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    labels_padding = calc_max_padding(
            output_size,
            voxel_size)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(foreground_mask, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(affs_mask, output_size)

    def create_source(sample):
        
        scalefactor = np.asarray(zarr.open(sample)['3d/raw'].attrs['scalefactor'])
        whichscale = ~np.multiply(scalefactor > 0.8, scalefactor < 1.2) # don't bother scaling differences <20%
        scalefactor = (tuple(np.multiply(scalefactor, whichscale) +
                np.multiply(1, ~whichscale)))

        print(scalefactor)
        print(sample)

        source = ZarrSource(sample,
                    {
                        raw: '3d/raw',
                        labels: '3d/labeled',
                        foreground_mask: '3d/mask',
                    },
                    {
                        raw: ArraySpec(interpolatable=True, voxel_size=voxel_size),
                        labels:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                        foreground_mask:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                    }
                )

        source += rescale_cmm.ScaleNode(raw, scalefactor)
        source += rescale_cmm.ScaleNode(labels, scalefactor, order=0)
        source += rescale_cmm.ScaleNode(foreground_mask, scalefactor, order=0)
        source += Normalize(raw)
        source += Pad(raw, None)
        source += Pad(labels, labels_padding)
        source += Pad(foreground_mask, labels_padding)
        source += RandomLocation()
        source += Reject(mask=foreground_mask, min_masked=0.01, reject_probability=0.9)

        return source

    data_sources = tuple(
            create_source(os.path.join(data_dir, sample))
            for sample in samples
            )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            control_point_spacing=[4,15,15],
            jitter_sigma=[1,2.5,2.5],
            rotation_interval=[0,math.pi/2.0],
            scale_interval=(0.8, 1.2))

    train_pipeline += SimpleAugment(transpose_only=[1, 2])

    train_pipeline += IntensityAugment(raw, 0.6, 1.1, -0.2, 0.2, z_section_wise=False)
    
    train_pipeline += NoiseAugment(raw, var=0.001)

    train_pipeline += CreateMask(labels, labels_mask)

    train_pipeline += AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            affinities_mask=affs_mask,
            dtype=np.float32)

    train_pipeline += BalanceLabels(
            gt_affs,
            affs_weights,
            affs_mask)

    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(batch_size)

    train_pipeline += PreCache(
            cache_size=40,
            num_workers=10)
    
    train_pipeline += Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'input': raw
            },
            loss_inputs={
                0: pred_affs,
                1: gt_affs,
                2: affs_weights
            },
            outputs={
                0: pred_affs
            },
            array_specs={
                pred_affs: ArraySpec(voxel_size=voxel_size)
            },
            save_every=10000,
            log_dir='log')

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, pred_affs])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                gt_affs: 'gt_affs',
                affs_weights: 'affs_weights',
                pred_affs: 'pred_affs'
            },
            every=1,
            output_filename='batch_{id}.zarr',
            )

    train_pipeline += PrintProfilingStats(every=10)

    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    iterations = 50
    train_until(iterations)
