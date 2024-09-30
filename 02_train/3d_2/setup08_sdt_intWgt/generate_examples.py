import glob
import gunpowder as gp
import logging
import math
import numpy as np
import os
import operator
import sys
import zarr
import torch
from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

group = "airyscan"
samples = glob.glob(base_dir + '/' + group + '*.zarr')

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 1

run_name = '2024-04-15_12-39-03_3d_sdt_IntWgt_b1'


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

class ComputeDT(gp.BatchFilter):
    def __init__(
            self,
            labels,
            sdt,
            constant=0.5, 
            dtype=np.float32,
            mode='3d',
            dilate_iterations=None,
            scale=None,
            mask=None,
            labels_mask=None,
            unlabelled=None):
        
        self.labels = labels
        self.sdt = sdt
        self.constant = constant
        self.dtype = dtype
        self.mode = mode
        self.dilate_iterations = dilate_iterations                                           
        self.scale = scale
        self.mask = mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled
    
    def setup(self):
        spec = self.spec[self.labels].copy()
        
        self.provides(self.sdt,spec)
        if self.mask:
            self.provides(self.mask, spec)
  
    def prepare(self, request):
  
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.sdt].copy()
  
        if self.labels_mask:
            deps[self.labels_mask] = deps[self.labels].copy()
  
        if self.unlabelled:
            deps[self.unlabelled] = deps[self.labels].copy()
  
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
        
        outputs = gp.Batch()

        labels_data = batch[self.labels].data
        distance = np.zeros_like(labels_data).astype(self.dtype)

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
            if self.labels_mask:
                mask = batch[self.labels_mask].data
            else:
                mask = (distance>0).astype(self.dtype) #(labels_data!=0).astype(self.dtype)

            if self.unlabelled:
                unlabelled_mask = batch[self.unlabelled].data
                mask *= unlabelled_mask

            outputs[self.mask] = gp.Array(
                    mask.astype(self.dtype),
                    spec)

        outputs[self.sdt] =  gp.Array(distance, spec)

        return outputs

class CopyArray(gp.BatchFilter):

    def __init__(
            self,
            array_in,
            array_out,
            dtype=np.uint8,
            ):

        self.array_in = array_in
        self.array_out = array_out

    def setup(self):

        arr_spec = self.spec[self.array_in].copy()
        self.provides(self.array_out, arr_spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.array_in] = request[self.array_out].copy()

        return deps

    def process(self, batch, request):

        outputs = gp.Batch()

        array_data = batch[self.array_in].data

        spec = batch[self.array_in].spec.copy()
        spec.roi = request[self.array_out].roi.copy()
        spec.dtype = array_data.dtype
        
        outputs[self.array_out] =  gp.Array(array_data, spec)

        return outputs

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

        #raw_cropped = np.expand_dims(raw_cropped, 0)
        #raw_cropped = np.repeat(raw_cropped, 3, 0)

        num_classes = len(self.bins) + 2

        binned = np.digitize(raw_cropped, self.bins) + 1
        # add one so values fall from 1 to num_bins+1 (if outside range 0-1)

        # set foreground labels to be class 0:
        #print(labels.shape, binned.shape)
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

    def __cropND(self, img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]


def train(count):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    gt = gp.ArrayKey("GT")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt, output_size)


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
                gp.RandomLocation() +
                gp.Reject_CMM(mask=labels, 
                    min_masked=0.005, 
                    min_min_masked=0.001,
                    reject_probability=1) #0.95)
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.Snapshot(
            output_filename=group+"_preAug_"+str(count)+".zarr",
            output_dir="snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
            },
            every=1)
    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2.,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))

    pipeline += gp.NoiseAugment(raw, var=0.01)

    pipeline += ComputeDT(
            labels,
            gt,
            mode='3d',
            dilate_iterations=1,
            scale=2,
            )

    
    pipeline += gp.Snapshot(
            output_filename=group+"_postAug_"+str(count)+".zarr",
            output_dir="snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt: "gt",
            },
            every=1)
    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    with gp.build(pipeline):
        for i in range(1):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    for i in range(3):
        train(i)
