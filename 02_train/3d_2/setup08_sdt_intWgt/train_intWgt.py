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
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

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
run_name = dt+'_3d_twoclass_IntWgt_b1'
#run_name = '2023-01-12_12-40-40_3d_twoclass_wgt3px_constUp_rejRamp_noClip'

class WeightedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):

    def __init__(self):
        super(WeightedBCEWithLogitsLoss, self).__init__()

    def forward(self, prediction, target, weights):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction, target, weight=weights)
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
            #surr = np.logical_xor(surr, fg)

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


def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    pred = gp.ArrayKey("PRED")
    gt = gp.ArrayKey("GT")
    surr_mask = gp.ArrayKey("SURR_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(surr_mask, output_size)
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
        kernel_size_up=kernel_size_up,
        constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*3], activation=None),
            )

    torch.cuda.set_device(0)

    loss = WeightedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lr=5e-5, 
            params=model.parameters())

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
                gp.Reject_CMM(mask=labels, min_masked=0.005, 
                    min_min_masked=0.001, reject_probability=0.95)
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2.,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))

    pipeline += gp.NoiseAugment(raw, var=0.01)

    pipeline += ComputeMask(
            labels,
            gt,
            surr_mask,
            erode_iterations=1,
            surr_pix=1,
            dtype=np.float32,
            )

    pipeline += BalanceLabelsWithIntensity(
            raw,
            surr_mask,
            mask_weights,
            )
    
    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    pipeline += gp.Unsqueeze([raw, gt, mask_weights])

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
        loss_inputs={
            0: pred,
            1: gt,
            2: mask_weights
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name,
        save_every=5000)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w
    # pred_mask: b,c,d,h,w

    pipeline += gp.Squeeze([raw, labels, mask_weights])

    # raw: c,d,h,w
    # labels: c,d,h,w

#    pipeline += gp.Squeeze([raw, labels])

    # raw: d,h,w
    # labels: d,h,w

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
    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10001)
