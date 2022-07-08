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

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/training'

samples = glob.glob(base_dir + '/*.zarr')

voxel_size = gp.Coordinate((1,)*2)

input_size = gp.Coordinate((132,)*2) * voxel_size
output_size = gp.Coordinate((92,)*2) * voxel_size

num_sections_array = np.zeros([len(samples),1])

new_dict = {}

for i in range(len(samples)):
    new_dict[samples[i]] = len(glob.glob(samples[i]+'/2d/raw/*'))


for x, y in new_dict.items(): #map(samples, num_sections_array):
    print(x)
    print(y)
