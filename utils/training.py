import os
import torch
import logging
import numpy as np
import gunpowder as gp
import operator
import glob
from natsort import natsorted
import zarr
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from .reject_cmm import *
from .reject_batch import *
from .noise_augment import * 

logger = logging.getLogger(__name__)

def check_snaps(run_name):
    #check if directory exists:
    if not os.path.exists("snapshots/"+run_name):
        return True
    snaps = natsorted(glob.glob("snapshots/"+run_name+"/batch*.zarr"))[-1]
    f = zarr.open(snaps)
    if(f['pred'][:].max() <= -1):
        print('bad snap')
        return False
    else:
        return True
    
class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):
        
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

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

class CheckSnaps(gp.BatchFilter):
    def __init__(
            self, 
            snapshot_dir="snapshots",
            every=1,
            ):
        self.snapshot_dir = snapshot_dir
        self.every = max(1, every)
        self.n = 0

    def process(self, batch, request):

        if self.n % self.every == 0:

            if not os.path.exists(self.snapshot_dir):
                batch['snap_check'] = True
                return

            last_snap = natsorted(glob.glob(self.snapshot_dir + "/batch*.zarr"))[-1]
            f = zarr.open(last_snap)
            if f['pred'][:].max() <= -1:
                snap_check = False
            else:
                snap_check = True
            if not snap_check:
                print('Terminating training due to bad snapshot.')
                raise gp.BatchFilterError("Bad snapshot detected, terminating training.")
        self.n += 1

class CheckPred(gp.BatchFilter):
    """
    A custom Gunpowder BatchFilter to check predictions during training.
    This filter checks the predictions at a specified interval and terminates 
    the training if a certain number of consecutive bad predictions (converged 
    to -1) are detected.
    Args:
        array (gp.ArrayKey): 
            The key of the array to check.
        every (int, optional): 
            The interval at which to check predictions. Default 1.
        consistency (int, optional): 
            The number of consecutive bad predictions required to terminate training. Default 10.
    """

    def __init__(
            self, 
            array,
            every=1,
            consistency=10,
            ):
        self.every = max(1, every)
        self.array = array
        self.consistency = consistency
        self.n = 0
        self.bad_n = 0

    def prepare(self, request):
        deps = gp.BatchRequest()

        deps[self.array] = request[self.array]

        self.check_if = self.n % self.every == 0
        return deps
    
    def process(self, batch, request):

        if self.check_if:
            data = batch.arrays[self.array].data
            if data.max() <= -1 or data.min() >= 1:
                self.bad_n += 1
                logger.warning(f"{self.n}: bad prediction detected #{self.bad_n}")
            if self.bad_n >= self.consistency:
                raise Exception(f"Bad prediction, terminating training at {self.n}.")
        self.n += 1

class ComputeDT(gp.BatchFilter):
    """Compute distance transform. This filter computes the signed distance transform (SDT) of the input labels.
    
    The distance transform is computed using the Euclidean distance transform (EDT) from the scipy.ndimage module.
    The resulting distance transform can be scaled and/or dilated based on the provided parameters.
    
    Args:

        labels (gp.ArrayKey): 
            The input labels for which the distance transform is computed.
        sdt (gp.ArrayKey): 
            The output signed distance transform.
        constant (float): 
            A constant value to be added/subtracted from the distance transform.
        dtype (numpy.dtype): 
            The data type of the output distance transform.
        mode (str): 
            The mode of computation, either '2d' or '3d'.
        dilate_iterations (int): 
            Number of iterations for binary dilation of the input labels.
        scale (float): 
            Scaling factor for the distance transform.
        mask (gp.ArrayKey): 
            Optional mask to be computed based on the distance transform.
        labels_mask (gp.ArrayKey): 
            Optional mask for the input labels.
        unlabelled (gp.ArrayKey): 
            Optional mask for unlabelled regions.
    """

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
        distance = -np.ones_like(labels_data).astype(self.dtype)

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

class NormalizeDT(gp.BatchFilter):

    # use if computing dt without scaling (eg not sdt)

    def __init__(self, distance):
        self.distance = distance

    def normalize(self, data):

        return (data - data.min()) / (data.max() - data.min())

    def process(self, batch, request):

        data = batch[self.distance].data

        # don't normalize zero batches
        if len(np.unique(data)) > 1:
            data = self.normalize(data)

        batch[self.distance].data = data


class BalanceLabelsWithIntensity(gp.BatchFilter):

    def __init__(
            self,
            raw,
            labels,
            scales,
            num_bins = 10,
            bin_min = 0.,
            bin_max = 1.,
            dilate_iter = None,
            ):

        self.labels = labels
        self.raw = raw
        self.scales = scales
        self.bins = np.linspace(bin_min, bin_max, num=num_bins+1)
        self.dilate_iter = dilate_iter

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
        raw_cropped = self.__cropND(raw.data, cropped_roi/vox_size)

        #raw_cropped = np.expand_dims(raw_cropped, 0)
        #raw_cropped = np.repeat(raw_cropped, 3, 0)

        num_classes = len(self.bins) + 2

        binned = np.digitize(raw_cropped, self.bins) + 1
        # add one so values fall from 1 to num_bins+1 (if outside range 0-1)

        # set foreground labels to be class 0:
        #print(labels.shape, binned.shape)
        if self.dilate_iter is not None:
            binned[binary_dilation(labels.data>0, iterations=self.dilate_iter)] = 0
        else:
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
