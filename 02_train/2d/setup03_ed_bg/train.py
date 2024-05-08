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

batch_size = 128 #256

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')-3]

run_name = dt+'_2class_edg_bg'

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

class ComputeMask(gp.BatchFilter):

    def __init__(
            self,
            labels,
            mask,
            surr_mask = None,
            surr_pix=3,
            pred_type='edges',
            dtype=np.uint8,
            ):

        self.labels = labels
        self.mask = mask
        self.surr_mask = surr_mask
        self.surr_pix = surr_pix
        self.pred_type = pred_type
        self.dtype = dtype


    def setup(self):

        maskspec = self.spec[self.labels].copy()
        self.provides(self.mask, maskspec)
        
        if self.surr_mask:
            surrspec = self.spec[self.labels].copy()
            self.provides(self.surr_mask, surrspec)


    def prepare(self, request):
        
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.mask].copy()
       
        return deps

    def _compute_mask(self, label_data):
        
        fg = np.zeros_like(label_data, dtype=bool)
        edges = np.zeros_like(label_data, dtype=bool)

        for label in np.unique(label_data):
            if label == 0:
                continue
            label_mask = label_data == label

            label_mask = binary_erosion(label_mask, iterations = 1)

            fg = np.logical_or(label_mask, fg)

        edges = np.logical_xor(label_data>0, fg)

        return fg.astype(self.dtype), edges.astype(self.dtype)

    def _compute_surr(self, mask):

        surr = binary_dilation(mask>0, iterations = self.surr_pix)
        surr = np.logical_xor(surr, mask).astype(self.dtype)
        mask[mask != 0] = 1
        surr[surr != 0] = 2
        surr = mask+surr

        return surr.astype(self.dtype)

    def process(self, batch, request):

        outputs = gp.Batch()

        labels_data = batch[self.labels].data
        mask_data = np.zeros_like(labels_data).astype(self.dtype)
        surr_data = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.mask].roi.copy()
        spec.dtype = self.dtype

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:
            fg_data, edge_data = self._compute_mask(labels_data)

            if self.pred_type == 'fg_bg':
                mask_data = fg_data
            elif self.pred_type == 'edges':
                mask_data = edge_data
            elif self.pred_type == 'three_class':
                edge_data[edge_data != 0] = 2
                fg_data[fg_data != 0] = 1
                mask_data = fg_data + edge_data
            else:
                raise Exception('Choose from one of the following prediction types: two_class, three_class')
                
        if self.surr_mask:
            surr_data = self._compute_surr(mask_data)
            outputs[self.surr_mask] = Array(surr_data, spec)

        outputs[self.mask] =  gp.Array(mask_data, spec)
        
        return outputs


def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
    surr_mask = gp.ArrayKey("SURR_MASK")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
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

    torch.cuda.set_device(0)

    loss = WeightedBCELoss() #torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:f'2d/raw/{i}',
                    labels:f'2d/labeled/{i}',
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    labels:gp.ArraySpec(interpolatable=False),
                }) +
                gp.Squeeze([raw,labels,]) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Normalize(raw) +
                gp.RandomLocation() + 
                gp.Reject_CMM(mask=labels, min_masked=0.05, min_min_masked=0.001, reject_probability=0.9)
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

    pipeline += ComputeMask(
            labels,
            gt,
            surr_mask = surr_mask,
            surr_pix=1,
            pred_type='edges',
            dtype=np.float32,
            )

    pipeline += gp.BalanceLabels(
            surr_mask,
            mask_weights,
            num_classes = 3,
            clipmin = None,
            clipmax = None,
            )

    # raw: h,w
    # labels: h,w

    pipeline += gp.Unsqueeze([raw, gt, mask_weights]) #, gt_fg])

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
        save_every=500)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # pred_mask: b,c,h,w
    
    pipeline += gp.Squeeze([raw, gt, pred, mask_weights], axis=1)

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
