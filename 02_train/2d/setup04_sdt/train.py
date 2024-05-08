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

run_name = dt+'_sdt_bgr-1_scale2'

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

        scaled = (weights * weights * (prediction - target) ** 2)

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

class ComputeDT(BatchFilter):

    def __init__(
            self,
            labels,   # input = labels
            sdt,      # output = gt_sdt
            constant=0.5,
            dtype=np.float32,
            mode='3d', #
            dilate_iterations=None, # 1
            scale=None, # 2
            mask=None,  # mask for balancing labels
            ):

        self.labels = labels
        self.sdt = sdt
        self.constant = constant
        self.dtype = dtype
        self.mode = mode
        self.dilate_iterations = dilate_iterations
        self.scale = scale
        self.mask = mask

    def setup(self):

        spec = self.spec[self.labels].copy()

        self.provides(self.sdt,spec)

        if self.mask:
            self.provides(self.mask, spec)

    def prepare(self, request):

        deps = BatchRequest()
        deps[self.labels] = request[self.sdt].copy()

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
        distance = np.zeros_like(labels_data).astype(self.dtype)-1

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

            mask = (labels_data!=0).astype(self.dtype)

            outputs[self.mask] = Array(
                    mask.astype(self.dtype),
                    spec)

        outputs[self.sdt] =  Array(distance, spec)

        return outputs

def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    sdt_mask = gp.ArrayKey("SDT_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
    
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
    request.add(sdt_mask, output_size)
    request.add(mask_weights, output_size)
    
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
            ConvPass(num_fmaps, 1, [[1,]*2], activation='Tanh'),
            ) 

    torch.cuda.set_device(0)

    loss = WeightedMSELoss() #torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=1e-5, 
            params=model.parameters(),
            betas=(0.95, 0.999))

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
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
                }) +
                gp.Squeeze([raw,labels,labels_mask]) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Pad(labels_mask, padding) +
                gp.Normalize(raw) +
                gp.RandomLocation(mask=labels_mask) + 
                gp.Reject(mask=labels_mask, min_masked=0.05, reject_probability=0.95)
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

    pipeline += ComputeDT(
            labels,
            gt,
            mode = '2d', 
            dilate_iterations = 1, 
            scale = 2, 
            mask = sdt_mask,
            )
    
    pipeline += BalanceLabels(
            sdt_mask,
            mask_weights,
            clipmin = None,
            clipmax = None,
            )
    
    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    # raw: h,w
    # labels: h,w
    # labels_mask: h,w

    pipeline += gp.Unsqueeze([raw, gt]) #, gt_fg])

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
        save_every=500)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w
    
    pipeline += gp.Squeeze([raw, gt, pred], axis=1)

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
            every=250)
    
    pipeline += PrintProfilingStats(every=10)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10000)
