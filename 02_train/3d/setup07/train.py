import glob
import json
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr

from funlib.learn.torch.models import UNet,ConvPass
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from add_local_shape_descriptor import AddLocalShapeDescriptor

from scipy.ndimage import distance_transform_edt, \
        binary_erosion, binary_dilation

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../../01_data/zarrs/train'

samples = glob.glob(data_dir+'/confocal_NL3_Set5*.zarr') #os.listdir(data_dir)

batch_size = 1

run_name = '2023-01-31_test_confoc_WT_NL3_RejCMM'

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


class ChangeBackground(BatchFilter):

    def __init__(self, labels, in_background, out_background):
        self.labels = labels
        self.in_background = in_background
        self.out_background = out_background

    def process(self, batch, request):

        data = batch[self.labels].data

        data[data==self.in_background] = self.out_background

        batch[self.labels].data = data


class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out,
            constant_upsample,
            dims):

        super().__init__()
            
        self.unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        num_fmaps_out=num_fmaps_out,
        constant_upsample=constant_upsample)

        self.lsd_head = ConvPass(num_fmaps_out, dims, [[1]*3], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps_out, 3, [[1]*3], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds, affs

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _create_loss(self, pred, target, weights):

        scaled = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
            self,
            lsd_pred,
            lsd_target,
            lsd_weights,
            affs_pred,
            affs_target,
            affs_weights):

        loss_1 = self._create_loss(lsd_pred, lsd_target, lsd_weights)
        loss_2 = self._create_loss(affs_pred, affs_target, affs_weights)

        return loss_1 + loss_2


def train_until(max_iteration):

    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
    sigma = 10
    
    in_channels = 1
    num_fmaps = 12
    num_fmaps_out = 14
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    #num_levels = len(downsample_factors) + 1
    #kernel_size_down = [[(3,3,3),(3,3,3)]]*num_levels
    #kernel_size_up = [[(3,3,3),(3,3,3)]]*(num_levels-1)
    
    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]] 

    model = MtlsdModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out,
            constant_upsample=True,
            dims=4)

    torch.cuda.set_device(0)
    
    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-5,
            betas=(0.95,0.999))

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    #foreground_mask = ArrayKey('FOREGROUND_MASK')
    labels_mask = ArrayKey('LABELS_MASK')
    gt_affs = ArrayKey('GT_AFFS')
    gt_lsds = ArrayKey('GT_LSDS')
    pred_affs = ArrayKey('PRED_AFFS')
    pred_lsds = ArrayKey('PRED_LSDS')
    affs_mask = ArrayKey('AFFS_MASK')
    lsds_weights = ArrayKey('LSDS_WEIGHTS')
    affs_weights = ArrayKey('AFFS_WEIGHTS')

    #input_shape = Coordinate((84,156,156))
    #output_shape = Coordinate((52,64,64))

    input_shape = Coordinate((36,172,172))
    output_shape = Coordinate((16,80,80))

    voxel_size = Coordinate((4,1,1))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    labels_padding = calc_max_padding(
            output_size,
            voxel_size)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    #request.add(foreground_mask, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds, output_size)
    request.add(pred_affs, output_size)
    request.add(pred_lsds, output_size)
    request.add(affs_weights, output_size)
    request.add(lsds_weights, output_size)
    request.add(affs_mask, output_size)

    def create_source(sample):
        
        print(sample)

        source = ZarrSource(sample,
                    {
                        raw: '3d/raw',
                        labels: '3d/labeled',
                        #foreground_mask: '3d/mask',
                    },
                    {
                        raw: ArraySpec(interpolatable=True, voxel_size=voxel_size),
                        labels:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                        #foreground_mask:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                    }
                )

        source += Normalize(raw)
        source += Pad(raw, None)
        source += Pad(labels, labels_padding)
        #source += Pad(foreground_mask, labels_padding)
        source += RandomLocation()
        source += Reject_CMM(mask=labels, min_masked=0.005, min_min_masked=0.001, reject_probability=0.99)
        #source += Reject(mask=foreground_mask, min_masked=0.05, reject_probability=0.95) #commented out after 300k

        return source

    data_sources = tuple(
            create_source(sample) #os.path.join(data_dir, sample))
            for sample in samples
            )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            control_point_spacing=(32,)*3,
            jitter_sigma=(2,)*3,
            rotation_interval=[0,math.pi/2.0],
            scale_interval=(0.8, 1.2),
            subsample=4)

    train_pipeline += SimpleAugment(transpose_only=[1, 2])

    train_pipeline += IntensityAugment(raw, 0.6, 1.1, -0.2, 0.2, z_section_wise=False)
    
    train_pipeline += NoiseAugment(raw, var=0.005)

    train_pipeline += CreateMask(labels, labels_mask)

    train_pipeline += ChangeBackground(labels, 0, 500)

    train_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            mask=lsds_weights,
            sigma=sigma,
            downsample=1,
            component='mean_size')
    
    train_pipeline += ChangeBackground(labels, 500, 0)

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

    #train_pipeline += IntensityScaleShift(raw, 2,-1)

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
                0: pred_lsds,
                1: gt_lsds,
                2: lsds_weights,
                3: pred_affs,
                4: gt_affs,
                5: affs_weights
            },
            outputs={
                0: pred_lsds,
                1: pred_affs
            },
            array_specs={
                pred_lsds: ArraySpec(voxel_size=voxel_size),
                pred_affs: ArraySpec(voxel_size=voxel_size)
            },
            save_every=10000,
            log_dir='log/'+run_name,
            checkpoint_basename='model_'+run_name)

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, pred_affs, gt_lsds, pred_lsds])

    #train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot(
            output_dir = 'snapshots/'+run_name, 
            dataset_names={
                raw: 'raw',
                labels: 'labels',
                gt_affs: 'gt_affs',
                gt_lsds: 'gt_lsds',
                pred_affs: 'pred_affs', 
                pred_lsds: 'pred_lsds',
                #lsds_weights: 'lsds_weights',
                #affs_weights: 'affs_weights',
            },
            every=250,
            output_filename='batch_{iteration}.zarr',
            )

    train_pipeline += PrintProfilingStats(every=50)

    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    iterations = 1000
    train_until(iterations)
