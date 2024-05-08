import glob
import gunpowder as gp
import logging
import math
import numpy as np
import os
import operator
import sys
import torch
import zarr

from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from skimage.transform import resize
from scipy.ndimage import binary_dilation, generate_binary_structure
from add_local_shape_descriptor import AddLocalShapeDescriptor

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

samples = glob.glob(base_dir + '/*.zarr')
for sample in samples:
    print(sample)

target_vox = (24,9,9)
voxel_size = gp.Coordinate(target_vox)

input_size = gp.Coordinate((1056, 1548, 1548))
output_size = gp.Coordinate((576, 720, 720))

batch_size = 1

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]
run_name = dt+'_3d_scaled_down_lsds'


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

class BalanceLabelsWithIntensity(gp.BatchFilter):

    def __init__(
            self,
            raw,
            labels,
            scales,
            num_bins = 10,
            bin_min = 0.,
            bin_max = 1.,
            ):

        self.labels = labels
        self.raw = raw
        self.scales = scales
        self.bins = np.linspace(bin_min, bin_max, num=num_bins+1)

    def setup(self):
        
        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.labels] = request[self.scales]
        deps[self.raw] = request[self.raw]

        return deps

    def process(self, batch, request):
        
        labels = batch.arrays[self.labels]

        raw = batch.arrays[self.raw]
        
        cropped_roi = labels.spec.roi.get_shape()
        vox_size = labels.spec.voxel_size
        raw_cropped = self.__cropND(raw.data, cropped_roi/voxel_size)

        num_classes = len(self.bins) + 2 

        binned = np.digitize(raw_cropped, self.bins) + 1 
        # add one so values fall from 1 to num_bins+1 (if outside range 0-1)
        
        # set foreground labels to be class 0:
        binned[labels.data>0] = 0

        # initialize scales:
        scale_data = np.ones(labels.data.shape, dtype=np.float32)

        # calculate the fraction of per-class samples:
        classes, counts = np.unique(binned, return_counts=True)
        fracs = counts.astype(float) / scale_data.sum()
        
        # prevent background from being weighted over foreground:
        fg = np.where(classes==0)[0]
        if len(fg)>0:
            fracs = np.clip(fracs, fracs[fg], None)

        # calculate the class weights
        w_sparse = 1.0 / float(num_classes) / fracs
        w = np.zeros(num_classes)
        w[classes] = w_sparse
        
        # scale the scale with the class weights
        scale_data *= np.take(w, binned)
        
        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi
        batch.arrays[self.scales] = gp.Array(scale_data, spec)
        
        #outputs = gp.Batch()

        #labels_data = batch[self.in_labels].data
        #mask_data = np.zeros_like(labels_data).astype(self.dtype)

        #spec = batch[self.in_labels].spec.copy()
        #spec.roi = request[self.out_mask].roi.copy()
        #spec.dtype = self.dtype

        ## don't need to compute on entirely background batches
        #if np.sum(labels_data) != 0:
        #    mask_data = (labels_data>0).astype(self.dtype)

        #if self.dilate_pix >0:
        #    mask_data = binary_dilation(
        #            mask_data,
        #            structure = self.dilate_struct,
        #            iterations = self.dilate_pix).astype(self.dtype)

        #outputs[self.out_mask] = gp.Array(mask_data, spec)

        #return outputs
    def __cropND(self, img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

def train(iterations):
    
    raw_orig = gp.ArrayKey("RAW_ORIG")
    labels_orig = gp.ArrayKey("LABELS_ORIG")
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    scales = gp.ArrayKey("SCALES")

    request = gp.BatchRequest()
    #request.add(raw_orig, input_size)
    #request.add(labels_orig, output_size)
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(scales, output_size)

    torch.cuda.set_device(0)

    padding = calc_max_padding(output_size, voxel_size)
    
    def create_source(sample):
        if "airyscan" in sample:
            vox = (16,4,4)
        elif "spinning" in sample:
            vox = (24, 9, 9)
        else:
            vox = (35,14,14)
        vox = gp.Coordinate(vox)
        
        source = gp.ZarrSource(
                sample,
                datasets={
                    raw_orig:f'3d/raw',
                    labels_orig:f'3d/labeled',
                },
                array_specs={
                    raw_orig:gp.ArraySpec(interpolatable=True, voxel_size=vox),
                    labels_orig:gp.ArraySpec(interpolatable=False, voxel_size=vox),
                })
        #source += gp.RandomLocation()

        source += gp.Normalize(raw_orig)
        source += gp.Resample(raw_orig, target_vox, raw, ndim=3)
        source += gp.Resample(labels_orig, target_vox, labels, ndim=3)

        source += gp.Pad(raw, None)
        source += gp.Pad(labels, padding)
        
        source += gp.RandomLocation()
        
        return source
    
    sources =tuple(
            create_source(sample)
            for sample in samples)
    pipeline = sources

    pipeline += gp.RandomProvider()
    
    pipeline += BalanceLabelsWithIntensity(raw, labels, scales)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.Snapshot(
            output_filename="batch_{id}.zarr",
            output_dir = 'snapshots/test',
            dataset_names={
                raw: "raw",
                labels: "labels",
                scales: "scales",
                #raw_orig: "raw_orig",
                #labels_orig: "labels_orig",
            },
            every=1)
    
    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":
    for i in range(10):
        train(1)
