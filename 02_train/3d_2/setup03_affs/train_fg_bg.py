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
from scipy.ndimage import binary_erosion, binary_dilation

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
run_name = dt+'_3d_affs_from_fgbg_train'

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


class ToDtype(gp.BatchFilter):

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def process(self, batch, request):

        data = batch[self.array].data

        batch[self.array].data = data.astype(self.dtype)

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
    labels_mask = gp.ArrayKey("LABELS_MASK")
    pred = gp.ArrayKey("PRED")
    gt = gp.ArrayKey("GT")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(pred, output_size)
    request.add(gt, output_size)
    request.add(mask_weights, output_size)

    in_channels = 1
    num_fmaps=12
    num_fmaps_out = 14
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
        kernel_size_up=kernel_size_up)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 3, [[1,]*3], activation='Sigmoid'))

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

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

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2, z_section_wise=False)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2.0,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8,1.2))

    pipeline += gp.NoiseAugment(raw, var=0.01)
    
    pipeline += gp.AddAffinities([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], labels, gt)

    pipeline += gp.BalanceLabels(
            gt,
            mask_weights,
            clipmin = 0.005,
            )
    
    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    pipeline += gp.Unsqueeze([raw])

    # raw: c,d,h,w
    # labels: c,d,h,w
    # labels_mask: d,h,w
    
    #pipeline += ToDtype(raw, np.float32)
    #pipeline += ToDtype(labels, np.float32)
    #pipeline += ToDtype(mask_weights, np.float32)

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
        loss_inputs={
            0: pred,
            1: gt,
            2: mask_weights
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name,
        save_every=1000)

    pipeline += gp.Squeeze([raw])

    pipeline += gp.Squeeze([raw, pred, gt, labels, labels_mask, mask_weights])
    # gt: c,d,h,w
    # pred: c,d,h,w
    # mask_weights: c,d,h,w

    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir="snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                labels_mask: "labels_mask",
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
