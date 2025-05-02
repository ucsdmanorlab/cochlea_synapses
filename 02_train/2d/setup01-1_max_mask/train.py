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
logger = logging.getLogger(__name__)

base_dir = '../../../01_data/zarrs/train'

zarrlist = glob.glob(base_dir + '/*.zarr')
voxel_size = gp.Coordinate((1,)*2)

input_size = gp.Coordinate((312,)*2) * voxel_size
output_size = gp.Coordinate((272,)*2) * voxel_size

batch_size = 32

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')-3]

run_name = dt+'_synapse_mask3'

samples = {name: zarr.open(name)['raw_max'].shape for name in zarrlist}

for sample in samples:
    print(sample, 
          samples[sample],
          gp.Coordinate(
              [max(math.ceil((b-a)), 0) for a,b in zip(samples[sample],input_size)])
    )

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
    
class CheckDtype(gp.BatchFilter):
    def __init__(self, keys):
        self.keys = keys

    def process(self, batch, request):
        for key in self.keys:
            logger.info("%s dtype: %s", key, batch[key].data.dtype)

class ChangeDtype(gp.BatchFilter):
    def __init__(self, key, dtype):
        self.key = key
        self.dtype = dtype

    # def setup(self):
    #     spec = self.spec[self.key].copy()
    #     spec.dtype = self.dtype
    #     self.provides(self.key, spec)

    def process(self, batch, request):
        spec = batch[self.key].spec.copy()
        spec.dtype = self.dtype
        batch[self.key] = gp.Array(
            batch[self.key].data.astype(self.dtype),
            spec)
        
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
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
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
            ConvPass(num_fmaps, 1, [[1,]*2], activation=None),
            torch.nn.Sigmoid()
            )

    torch.cuda.set_device(0)

    loss = WeightedBCELoss() 
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:f'raw_max',
                    gt:f'label_max_dilated',
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    gt:gp.ArraySpec(interpolatable=False),
                }) +
                gp.Pad(raw, 
                       gp.Coordinate(
                           [max(math.ceil((b-a)), 200) for a,b in zip(samples[sample],input_size)]
                       )) +
                gp.Pad(gt, 
                       gp.Coordinate(
                           [max(math.ceil((b-a)), 200) for a,b in zip(samples[sample],input_size)])+padding) +
                gp.Normalize(raw) +
                gp.RandomLocation() +
                gp.Reject(mask=gt, min_masked=0.05, reject_probability=0.9)
                for sample in zarrlist)

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

    pipeline += gp.BalanceLabels(
            gt,
            mask_weights,
            num_classes = 2,
            clipmin = None,
            clipmax = None,
            )

    # raw: h,w
    # labels: h,w
    # labels_mask: h,w
    #pipeline += CheckDtype([raw, gt, mask_weights])
    pipeline += ChangeDtype(gt, np.float32)
    pipeline += ChangeDtype(mask_weights, np.float32)
    #pipeline += CheckDtype([raw, gt, mask_weights])
    pipeline += gp.Unsqueeze([raw, gt, mask_weights]) 

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
    #pipeline += CheckDtype([raw, gt, mask_weights, pred])
    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w
    
    pipeline += gp.Squeeze([raw, gt, pred, mask_weights], axis=1)

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
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

    train(2000)
