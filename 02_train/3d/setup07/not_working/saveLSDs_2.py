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

data_dir = '../../../01_data/zarrs/ctbp2/restest/validation'

samples = glob.glob(data_dir+'/*3.zarr') #os.listdir(data_dir)

batch_size = 1

def save_out(out_file,
        array,
        key,
        res=[1,1,1],
        offset=[0,0,0]):

    out_file[key] = array
    out_file[key].attrs['offset'] = offset
    out_file[key].attrs['resolution'] = res


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


def save_LSDs(raw_file, raw_shape):

    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
    sigma = 10
    
    #raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    #foreground_mask = ArrayKey('FOREGROUND_MASK')
    labels_mask = ArrayKey('LABELS_MASK')
    gt_affs = ArrayKey('GT_AFFS')
    gt_lsds = ArrayKey('GT_LSDS')
    affs_mask = ArrayKey('AFFS_MASK')
    lsds_weights = ArrayKey('LSDS_WEIGHTS')
    affs_weights = ArrayKey('AFFS_WEIGHTS')

    input_shape = Coordinate(raw_shape) #Coordinate((36,172,172))
    output_shape = input_shape #Coordinate((16,80,80))

    voxel_size = Coordinate((4,1,1))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    
    context = (input_shape - output_shape) / 2
    context = context * voxel_size

    request = BatchRequest()
    #request.add(raw, input_size)
    request.add(labels, output_size)
    #request.add(foreground_mask, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    #request.add(gt_lsds, output_size)
    #request.add(affs_weights, input_size)
    #request.add(lsds_weights, output_size)
    #request.add(affs_mask, output_size)
    
    print('source/n')

    source = ZarrSource(
            raw_file,
            {
                #raw: '3d/raw',
                labels: '3d/labeled',
                #foreground_mask: '3d/mask',
            },
            {
                #raw: ArraySpec(interpolatable=True, voxel_size=voxel_size),
                labels:ArraySpec(interpolatable=False, voxel_size=voxel_size),
                #foreground_mask:ArraySpec(interpolatable=False, voxel_size=voxel_size),
            }
        )

    #source += Normalize(raw)

    #with build(source):
    #    total_output_roi = source.spec[raw].roi
    #    print(total_output_roi)
    #    total_input_roi =  total_output_roi.grow(context,context)
    #    print(total_input_roi)

    #total_input_roi = Roi(offset=None, shape=input_size)
    
    #scan = Scan(request)
    
    pipeline = source
    
    #pipeline += Pad(raw, context)

    pipeline += CreateMask(labels, labels_mask)

    pipeline += ChangeBackground(labels, 0, 500)

    #pipeline += AddLocalShapeDescriptor(
    #        labels,
    #        gt_lsds,
    #        mask=lsds_weights,
    #        sigma=sigma,
    #        downsample=1,
    #        component='mean_size')
    
    pipeline += ChangeBackground(labels, 500, 0)

    pipeline += AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            #affinities_mask=affs_mask,
            dtype=np.float32)

    #pipeline += BalanceLabels(
    #        gt_affs,
    #        affs_weights,
    #        affs_mask)

    #pipeline += scan
    
    #pipeline += Squeeze([raw, gt_affs])

    #lsd_request = BatchRequest()

    #lsd_request.add(raw, total_input_roi.get_shape())
    #lsd_request[raw].roi = total_input_roi
    #lsd_request.add(labels, total_output_roi.get_shape())
    #lsd_request.add(foreground_mask, total_output_roi.get_shape())
    #lsd_request.add(labels_mask, total_output_roi.get_shape())
    #lsd_request.add(affs_mask, total_output_roi.get_shape())
    ##lsd_request.add(affs_weights, total_output_roi.get_shape())
    ##lsd_request.add(lsds_weights, total_output_roi.get_shape())
    #lsd_request.add(gt_affs, total_output_roi.get_shape())
    ##lsd_request.add(gt_lsds, total_output_roi.get_shape())
    
    #print(total_input_roi.get_shape())
    print('build/n')

    with build(pipeline):
        batch = pipeline.request_batch(request)

    #return batch[gt_affs].data, batch[gt_lsds].data 

if __name__ == '__main__':

    for filename in samples:
        print(filename)
        out_file_path = filename.replace('validation', 'out')
        out_file = zarr.open(out_file_path)
        
        imageshape = np.asarray(zarr.open(filename+'/3d/raw')).shape
        print(imageshape)

        #gt_affs, gt_lsds = 
        save_LSDs(filename, imageshape)

        #save_out(out_file,  np.asarray(gt_affs), 'gt_affs')
        #save_out(out_file,  np.asarray(gt_lsds), 'gt_lsds')





