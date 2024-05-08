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
from datetime import datetime
from torch.nn.functional import binary_cross_entropy

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

zarrlist = glob.glob(base_dir + '/spinning*.zarr')
voxel_size = gp.Coordinate((1,)*2)

input_size = gp.Coordinate((140,)*2) * voxel_size
output_size = gp.Coordinate((100,)*2) * voxel_size

samples = {}
for i in range(len(zarrlist)):
    samples[zarrlist[i]] = len(glob.glob(zarrlist[i]+'/2d/raw/*'))

batch_size = 128

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')-3]

run_name = dt+'_2class_myo'

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

class AddChannel(gp.BatchFilter):
    def __init__(
            self, 
            in_array,
            add_array,
            ch_axis=0,
            ):

        self.in_array = in_array
        self.add_array = add_array
        self.ch_axis = ch_axis

    #def prepare(self, request):

    def process(self, batch, request):

        batch[self.in_array].data = np.stack((batch[self.in_array].data, batch[self.add_array].data), axis = self.ch_axis)
        spatial_dims = request[self.in_array].roi.dims

class ComputeMask(gp.BatchFilter):

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
        
        deps = gp.BatchRequest()
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

        outputs = gp.Batch()

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
        outputs[self.mask] =  gp.Array(mask_data, spec)
        
        if self.surr_mask:
            outputs[self.surr_mask] = gp.Array(surr_data, spec)

        return outputs

def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    myo = gp.ArrayKey("MYO")
    gt = gp.ArrayKey("GT")
    surr_mask = gp.ArrayKey("SURR_MASK")
    pred = gp.ArrayKey("PRED")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(myo, input_size)
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
        in_channels=2,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*2], activation=None),
            torch.nn.Sigmoid()
            )

    torch.cuda.set_device(0)

    loss = WeightedBCELoss() #torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:f'2d/raw/{i}',
                    labels:f'2d/labeled/{i}',
                    myo:f'2d/myo/{i}',
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    labels:gp.ArraySpec(interpolatable=False),
                    myo:gp.ArraySpec(interpolatable=True),
                }) +
                gp.Squeeze([raw,labels,myo]) + 
                gp.Pad(raw, None) +
                gp.Pad(myo, None) + 
                gp.Pad(labels, padding) +
                gp.Normalize(raw) +
                gp.Normalize(myo) + 
                gp.RandomLocation() +
                gp.Reject_CMM(mask=labels, min_masked=0.01, min_min_masked=0.001, reject_probability=0.95)
                for sample, num_sections in samples.items()
                for i in range(num_sections))

    pipeline = sources

    pipeline += gp.RandomProvider()
    
    pipeline += AddChannel(raw, myo)

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
            surr_mask,
            erode_iterations=0,
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
    # labels_mask: h,w

    pipeline += gp.Unsqueeze([gt, mask_weights]) 

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
    
    pipeline += gp.Squeeze([gt, pred, mask_weights], axis=1)

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

    train(10000)
