import glob
import json
import logging
import math
import numpy as np
import random
import os
import sys
import torch
import zarr

from funlib.learn.torch.models import UNet,ConvPass
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *

from scipy.ndimage import distance_transform_edt, \
        binary_erosion, binary_dilation

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../../01_data/zarrs/selected_training' 

samples = glob.glob(data_dir+'/*.zarr')

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


class ComputeDT(BatchFilter):

    def __init__(
            self,
            labels,
            sdt,
            constant=0.5,
            dtype=np.float32,
            mode='3d',
            dilate_iterations=None,
            scale=None,
            mask=None,
            labels_mask=None,
            unlabelled=None):

        self.labels = labels
        self.sdt = sdt
        self.constant = constant
        self.dtype = dtype
        self.mode = mode
        self.dilate_iterations = dilate_iterations
        self.scale = scale
        self.mask = mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled

    def setup(self):

        spec = self.spec[self.labels].copy()

        self.provides(self.sdt,spec)

        if self.mask:
            self.provides(self.mask, spec)

    def prepare(self, request):

        deps = BatchRequest()
        deps[self.labels] = request[self.sdt].copy()

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.labels].copy()

        if self.unlabelled:
            deps[self.unlabelled] = deps[self.labels].copy()

        return deps

    def _compute_dt(self, data):

        dist_func = distance_transform_edt

        if self.dilate_iterations:
            data = binary_dilation(
                    data,
                    iterations=self.dilate_iterations)

        if self.scale:
            inner = dist_func(binary_erosion(data))
            outer = dist_func(np.logical_not(data))

            distance = (inner - outer) + self.constant

            distance = np.tanh(distance / self.scale)

        else:

            inner = dist_func(data) - self.constant
            outer = -(dist_func(1-np.logical_not(data)) - self.constant)

            distance = np.where(data, inner, outer)

        return distance.astype(self.dtype)

    def process(self, batch, request):

        outputs = Batch()

        labels_data = batch[self.labels].data
        distance = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.sdt].roi.copy()
        spec.dtype = np.float32

        labels_data = labels_data != 0

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:

            if self.mode == '3d':
                distance = self._compute_dt(labels_data)

            elif self.mode == '2d':
                for z in range(labels_data.shape[0]):
                    distance[z] = self._compute_dt(labels_data[z])
            else:
                raise ValueError('Only implemented for 2d or 3d labels')
                return

        if self.mask and self.mask in request:

            if self.labels_mask:
                mask = batch[self.labels_mask].data
            else:
                mask = (labels_data!=0).astype(self.dtype)

            if self.unlabelled:
                unlabelled_mask = batch[self.unlabelled].data
                mask *= unlabelled_mask

            outputs[self.mask] = Array(
                    mask.astype(self.dtype),
                    spec)

        outputs[self.sdt] =  Array(distance, spec)

        return outputs

class NormalizeDT(BatchFilter):

    # use if computing dt without scaling (eg not sdt)

    def __init__(self, distance):
        self.distance = distance

    def normalize(self, data):

        return (data - data.min()) / (data.max() - data.min())

    def process(self, batch, request):

        data = batch[self.distance].data

        # don't normalize zero batches
        if len(np.unique(data)) > 1:
            data = self.normalize(data)

        batch[self.distance].data = data


def train_until(max_iteration):

    in_channels = 1
    num_fmaps = 12
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
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1]*3], activation='Tanh'))

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-5,
            betas=(0.95,0.999))

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    foreground_mask = ArrayKey('FOREGROUND_MASK')
    gt_sdt = ArrayKey('GT_SDT')
    pred_sdt = ArrayKey('PRED_SDT')
    sdt_mask = ArrayKey('SDT_MASK')
    sdt_weights = ArrayKey('SDT_WEIGHTS')

    input_shape = Coordinate((36,172,172))
    output_shape = Coordinate((16,80,80))

    voxel_size = Coordinate((5,2,2))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    labels_padding = calc_max_padding(
            output_size,
            voxel_size)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(foreground_mask, output_size)
    
    request.add(gt_sdt, output_size)
    request.add(pred_sdt, output_size)
    request.add(sdt_weights, output_size)
    request.add(sdt_mask, output_size)

    def create_source(sample):
        
        print(sample)

        source = ZarrSource(sample,
                    {
                        raw: 'raw',
                        labels: 'labeled',
                        foreground_mask: 'mask',
                    },
                    {
                        raw: ArraySpec(interpolatable=True, voxel_size=voxel_size),
                        labels:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                        foreground_mask:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                    }
                )

        source += Normalize(raw)
        source += Pad(raw, None)
        source += Pad(labels, labels_padding)
        source += Pad(foreground_mask, labels_padding)
        source += RandomLocation()
        source += Reject(mask=foreground_mask, min_masked=0.05, reject_probability=0.9)

        return source

    data_sources = tuple(
            create_source(os.path.join(data_dir, sample))
            for sample in samples
            )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()
    
    train_pipeline += ElasticAugment(
            control_point_spacing=[6,24,24],
            jitter_sigma=[0.5,2,2],
            rotation_interval=[0,math.pi/2.0],
            scale_interval=(0.8, 1.2))

    train_pipeline += SimpleAugment(transpose_only=[1, 2])

    train_pipeline += IntensityAugment(raw, 0.6, 1.1, -0.2, 0.2, z_section_wise=False)
    
    train_pipeline += NoiseAugment(raw, var=0.01)

    train_pipeline += ComputeDT(
            labels,
            gt_sdt,
            mode='3d',
            dilate_iterations=1,
            scale=2,
            mask=sdt_mask,
            )

    train_pipeline += BalanceLabels(
            sdt_mask,
            sdt_weights)


    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Unsqueeze([raw, gt_sdt])
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
                0: pred_sdt,
                1: gt_sdt,
                2: sdt_weights
            },
            outputs={
                0: pred_sdt
            },
            array_specs={
                pred_sdt: ArraySpec(voxel_size=voxel_size)
            },
            save_every=50000,
            log_dir='log')

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_sdt, pred_sdt])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                gt_sdt: 'gt_sdt',
                foreground_mask: 'foreground_mask',
                sdt_weights: 'sdt_weights',
                pred_sdt: 'pred_sdt'
            },
            every=500,
            output_filename='batch_{iteration}.zarr',
            )

    train_pipeline += PrintProfilingStats(every=1000)

    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    iterations = 400000
    train_until(iterations)
