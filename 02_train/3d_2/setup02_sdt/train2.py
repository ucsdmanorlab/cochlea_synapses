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
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train'

samples = glob.glob(base_dir + '/*.zarr')

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 1

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]
run_name = dt+'_3d_sdt_bgr0_scale2_dil2'

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

    return max_padding.get_begin() #gp.Coordinate(max_padding)
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
                mask = (labels_data!=0).astype(self.dtype)

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


def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    pred = gp.ArrayKey("PRED")
    gt = gp.ArrayKey("GT")
    surr_mask = gp.ArrayKey("SURR_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(pred, output_size)
    request.add(gt, output_size)
    request.add(mask_weights, output_size)
    request.add(surr_mask, output_size)

    in_channels = 1
    num_fmaps=12
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors, 
        kernel_size_down=kernel_size_down, 
        kernel_size_up=kernel_size_up,
        constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*3], activation='Tanh'))
    
    torch.cuda.set_device(1)

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters(), betas=(0.95, 0.999))

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:'3d/raw',
                    labels:'3d/labeled',
                    labels_mask:'3d/mask'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                    labels:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
                    labels_mask:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)
                }) +
                gp.Normalize(raw) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Pad(labels_mask, padding) +
                gp.RandomLocation() +
                gp.Reject(mask=labels_mask, min_masked=0.05, reject_probability=0.9)
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2.0,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8,1.2))

    pipeline += gp.NoiseAugment(raw, var=0.01)

    pipeline += ComputeDT(
            labels,
            gt,
            mode='3d',
            dilate_iterations=2,
            scale=2,
            mask=surr_mask,
            )

    pipeline += gp.BalanceLabels(
            surr_mask,
            mask_weights,
            )
    
    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw, gt])

    # raw: c,d,h,w
    # labels: c,d,h,w
    # labels_mask: d,h,w
    
    pipeline += gp.Stack(batch_size)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w
    # labels_mask: b,d,h,w

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
        array_specs={
            pred: gp.ArraySpec(voxel_size=voxel_size)
            },
        loss_inputs={
            0: pred,
            1: gt,
            2: mask_weights
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name,
        save_every=1000)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w
    # labels_mask: b,d,h,w
    # pred_mask: b,c,d,h,w

    pipeline += gp.Squeeze([raw, labels, labels_mask])

    # raw: c,d,h,w
    # labels: c,d,h,w
    # labels_mask: d,h,w

#    pipeline += gp.Squeeze([raw, labels])

    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w
    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir="snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                pred: "pred",
                gt: "gt",
                mask_weights: "mask_weights",
            },
            every=250)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10000)
