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
#from gunpowder import *
#from gunpowder.ext import torch
#from gunpowder.torch import *
from datetime import datetime
from torch.nn.functional import binary_cross_entropy, cross_entropy

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

zarrlist = glob.glob(base_dir + '/*.zarr')
voxel_size = gp.Coordinate((1,)*2)
wgts = [0.334, 175, 50]
input_size = gp.Coordinate((140,)*2) * voxel_size
output_size = gp.Coordinate((100,)*2) * voxel_size

samples = {}
for i in range(len(zarrlist)):
    samples[zarrlist[i]] = len(glob.glob(zarrlist[i]+'/2d/raw/*'))

batch_size = 128

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')-3]

run_name = dt+'_3class_CELoss_new'

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

class ComputeMask(gp.BatchFilter):

    def __init__(
            self,
            labels,
            mask,
            erode_iterations=None,
            dtype=np.uint8,
            pred_type = 'two_class',
            ):

        self.labels = labels
        self.mask = mask
        self.pred_type = pred_type
        self.dtype = dtype
        
    def setup(self):

        maskspec = self.spec[self.labels].copy()
        self.provides(self.mask, maskspec)
        
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

            edge_mask = np.logical_xor(label_mask, binary_dilation(label_mask))
            
            if np.any( label_data[edge_mask]>0):

                edge_mask = np.logical_xor(label_mask, binary_erosion(label_mask))
                label_mask = binary_erosion(label_mask)

            fg = np.logical_or(label_mask, fg)
            edges = np.logical_or(edge_mask, edges)
        
        #edges = np.logical_xor(label_data>0, fg)

        return fg.astype(self.dtype), edges.astype(self.dtype)

    def process(self, batch, request):

        outputs = gp.Batch()

        labels_data = batch[self.labels].data
        mask_data = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.mask].roi.copy()
        spec.dtype = self.dtype

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:
            fg_data, edge_data = self._compute_mask(labels_data)
            
            if self.pred_type == 'two_class':
                mask_data = fg_data.astype(self.dtype)
            elif self.pred_type == 'three_class':
                edge_data = edge_data.astype(self.dtype)
                edge_data[edge_data != 0] = 2
                fg_data[fg_data != 0] = 1
                mask_data = fg_data + edge_data
                #print(mask_data.min(), mask_data.max())
            else:
                raise Exception('Choose from one of the following prediction types: two_class, three_class')

        outputs[self.mask] =  gp.Array(mask_data, spec)

        return outputs

def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    #labels_mask = gp.ArrayKey("LABELS_MASK")
    gt = gp.ArrayKey("GT")
    pred = gp.ArrayKey("PRED")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    #request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(pred, output_size)
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

    final_conv = torch.nn.Conv2d(
        in_channels=num_fmaps,
        out_channels=3,
        kernel_size=1)

    model = torch.nn.Sequential(
            unet,
            final_conv,
            torch.nn.Softmax(dim=1))
    
    torch.cuda.set_device(1)
    
    loss = torch.nn.CrossEntropyLoss(weight = torch.tensor(
                wgts, device=torch.device('cuda:1')))
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
                gp.Squeeze([raw,labels]) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Normalize(raw) +
                gp.RandomLocation() + 
                gp.Reject_CMM(mask=labels, min_masked=0.01, min_min_masked=0.001, reject_probability=0.95)
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
            dtype=np.int64,
            pred_type='three_class',
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
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name,
        save_every=500)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w
    
    pipeline += gp.Squeeze([raw], axis=1)
    #pipeline += ArgMax(pred_mask, axis=1)

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw", 
                labels: "labels",
                gt: "gt", 
                pred: "pred",
            },
            every=500)
    
    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10001)
