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

def train(iterations):
    
    raw_orig = gp.ArrayKey("RAW_ORIG")
    labels_orig = gp.ArrayKey("LABELS_ORIG")
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    request = gp.BatchRequest()
    #request.add(raw_orig, input_size)
    #request.add(labels_orig, output_size)
    request.add(raw, input_size)
    request.add(labels, output_size)

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
    
    pipeline += gp.Stack(batch_size)

    pipeline += gp.Snapshot(
            output_filename="batch_{id}.zarr",
            output_dir = 'snapshots/test_resamp',
            dataset_names={
                raw: "raw",
                labels: "labels",
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
