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
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from scipy.ndimage import binary_dilation, generate_binary_structure

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train' ##overfit/validate' #train'

samples = glob.glob(base_dir + '/*.zarr')
print(samples)

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 1

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]
run_name = dt+'_3d_affs' 

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

def train(iterations):
    
    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
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

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:'3d/raw',
                    labels:'3d/labeled',
                #    labels_mask:'3d/mask'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                    labels:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
                #    labels_mask:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)
                }) +
                gp.Normalize(raw) + 
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                #gp.Pad(labels_mask, padding) +
                CreateMask(labels, labels_mask) +
                #gp.SpecifiedLocation((gp.Coordinate((22, 86, 86)), gp.Coordinate((22,86,86)))) 
                gp.RandomLocation()  +
                gp.Reject_CMM(mask=labels_mask, 
                    min_masked=0.005, 
                    min_min_masked=0.001, 
                    reject_probability=0.99) 
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
            #affinities_mask=aff_mask,
            dtype=np.float32)
    
    dilate_struct = generate_binary_structure(4,1).astype(np.uint8)
    dilate_struct[0,:,:,:] = 0
    dilate_struct[2,:,:,:] = 0

    pipeline += CreateMask(gt, aff_mask, dilate_pix=1, dilate_struct=dilate_struct)

    pipeline += gp.BalanceLabels(
            aff_mask,
            mask_weights,
            clipmin=None,
            clipmax=None,
            )
    
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
        save_every=5000)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w
    pipeline += gp.Squeeze([raw], axis=1)
    pipeline += gp.Squeeze([raw, gt, pred, labels, labels_mask, mask_weights])

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

    train(10000)
