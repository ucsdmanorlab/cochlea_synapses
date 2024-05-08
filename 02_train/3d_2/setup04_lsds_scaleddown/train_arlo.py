import glob
import gunpowder as gp
import logging
import math
import numpy as np
import torch
import zarr

from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from gunpowder.ext import torch
from gunpowder.torch import *
from datetime import datetime
from scipy.ndimage import binary_dilation, generate_binary_structure
from lsd.train.gp import AddLocalShapeDescriptor

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = "../../../01_data/zarrs/train"

samples = glob.glob(base_dir + "/*.zarr")

target_vox = (9, 9, 9)  # for isotropic
# target_vox = (24,9,9) #tens of nanometers -- this is the "middle" size of the 3
voxel_size = gp.Coordinate(target_vox)

batch_size = 1

dt = str(datetime.now()).replace(":", "-").replace(" ", "_")
dt = dt[0 : dt.rfind(".")]
run_name = dt + "_3d_scaled_mid_r0p9"
# run_name = '2023-03-23_15-47-26_3d_scaled_mid_r0p9'


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _create_loss(self, pred, target, weights):
        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self, lsd_pred, lsd_target, lsd_weights, affs_pred, affs_target, affs_weights
    ):
        loss_1 = self._create_loss(lsd_pred, lsd_target, lsd_weights)
        loss_2 = self._create_loss(affs_pred, affs_target, affs_weights)

        return loss_1 + loss_2


class MtlsdModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        num_fmaps_out,
        constant_upsample,
        dims,
        num_heads,
    ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            num_fmaps_out=num_fmaps_out,
            constant_upsample=constant_upsample,
            num_heads=num_heads,
        )

        self.lsd_head = ConvPass(num_fmaps_out, dims, [[1] * 3], activation="Sigmoid")
        self.aff_head = ConvPass(num_fmaps_out, 3, [[1] * 3], activation="Sigmoid")

    def forward(self, input):
        z = self.unet(input)
        lsds = self.lsd_head(z[0])
        affs = self.aff_head(z[0])

        return lsds, affs


def calc_max_padding(output_size, voxel_size, mode="shrink"):
    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + voxel_size),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class CreateMask(gp.BatchFilter):
    def __init__(
        self, in_labels, out_mask, dtype=np.uint8, dilate_pix=0, dilate_struct=None
    ):
        self.in_labels = in_labels
        self.out_mask = out_mask
        self.dtype = dtype
        self.dilate_pix = dilate_pix
        self.dilate_struct = dilate_struct

    def setup(self):
        self.provides(self.out_mask, self.spec[self.in_labels].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.in_labels] = request[self.out_mask].copy()

        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        labels_data = batch[self.in_labels].data
        mask_data = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.in_labels].spec.copy()
        spec.roi = request[self.out_mask].roi.copy()
        spec.dtype = self.dtype

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:
            mask_data = (labels_data > 0).astype(self.dtype)

        if self.dilate_pix > 0:
            mask_data = binary_dilation(
                mask_data, structure=self.dilate_struct, iterations=self.dilate_pix
            ).astype(self.dtype)

        outputs[self.out_mask] = gp.Array(mask_data, spec)

        return outputs


class ChangeBackground(gp.BatchFilter):
    def __init__(self, labels, in_background, out_background):
        self.labels = labels
        self.in_background = in_background
        self.out_background = out_background

    def process(self, batch, request):
        data = batch[self.labels].data

        data[data == self.in_background] = self.out_background

        batch[self.labels].data = data


def train(iterations):
    neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    sigma = 20

    raw_orig = gp.ArrayKey("RAW_ORIG")
    labels_orig = gp.ArrayKey("LABELS_ORIG")
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    labels_mask = gp.ArrayKey("LABELS_MASK")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds = gp.ArrayKey("GT_LSDS")

    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    aff_mask = gp.ArrayKey("AFF_MASK")
    aff_weights = gp.ArrayKey("AFF_WEIGHTS")
    lsd_weights = gp.ArrayKey("LDS_WEIGHTS")

    in_channels = 1
    num_fmaps = 12
    num_fmaps_out = 14
    fmap_inc_factor = 5
    downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

    model = MtlsdModel(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        num_fmaps_out,
        constant_upsample=True,
        dims=4,
        num_heads=2,
    )

    input_shape = [148] * 3
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]

    input_size = gp.Coordinate(input_shape) * target_vox
    output_size = gp.Coordinate(output_shape) * target_vox

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds, output_size)
    request.add(pred_affs, output_size)
    request.add(pred_lsds, output_size)
    request.add(aff_mask, output_size)
    request.add(aff_weights, output_size)
    request.add(lsd_weights, output_size)

    torch.cuda.set_device(1)

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(
        lr=5e-5,
        params=model.parameters(),
        # betas=(0.95, 0.999)
    )

    padding = calc_max_padding(output_size, voxel_size)

    def create_source(sample):
        if "airyscan" in sample:
            vox = (16, 4, 4)
        elif "spinning" in sample:
            vox = (24, 9, 9)
        else:  # "confocal"
            vox = (36, 16, 16)

        vox = gp.Coordinate(vox)

        source = gp.ZarrSource(
            sample,
            datasets={
                raw_orig: f"3d/raw",
                labels_orig: f"3d/labeled",
            },
            array_specs={
                raw_orig: gp.ArraySpec(interpolatable=True, voxel_size=vox),
                labels_orig: gp.ArraySpec(interpolatable=False, voxel_size=vox),
            },
        )
        source += gp.Normalize(raw_orig)
        source += gp.Resample(raw_orig, target_vox, raw, ndim=3)
        source += gp.Resample(labels_orig, target_vox, labels, ndim=3)

        source += gp.Pad(raw, None)
        source += gp.Pad(labels, padding)
        source += gp.RandomLocation()

        source += CreateMask(labels, labels_mask)
        source += gp.Reject_CMM(mask=labels_mask, 
                min_masked=0.005, 
                min_min_masked=0.001,
                reject_probability=0.9)

        return source

    sources = tuple(create_source(sample) for sample in samples)
    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
        control_point_spacing=(32,) * 3,
        jitter_sigma=(2.0,) * 3,
        rotation_interval=(0, math.pi / 2),
        scale_interval=(0.8, 1.2),
        subsample=4,
    )

    pipeline += gp.NoiseAugment(raw, var=0.01)

    pipeline += ChangeBackground(labels, 0, 500)

    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsd_weights,
        sigma=sigma,
        downsample=1,
        components="0129",
    )

    pipeline += ChangeBackground(labels, 500, 0)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        # affinities_mask=aff_mask,
        dtype=np.float32,
    )

    dilate_struct = generate_binary_structure(4, 1).astype(np.uint8)
    dilate_struct[0, :, :, :] = 0
    dilate_struct[2, :, :, :] = 0

    pipeline += CreateMask(gt_affs, aff_mask, dilate_pix=1, dilate_struct=dilate_struct)

    pipeline += gp.BalanceLabels(
        aff_mask,
        aff_weights,
        clipmin=None,
        clipmax=None,
    )

    # raw: h,w
    # labels: h,w

    pipeline += gp.Unsqueeze([raw])  # , gt_fg])

    # raw: c,h,w
    # labels: c,h,w

    pipeline += gp.Stack(batch_size)

    # raw: b,c,h,w
    # labels: b,c,h,w

    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsd_weights,
            3: pred_affs,
            4: gt_affs,
            5: aff_weights,
        },
        log_dir="log/" + run_name,
        checkpoint_basename="model_" + run_name,
        save_every=5000,
    )

    # raw: b,c,h,w
    # labels: b,c,h,w
    # pred_mask: b,c,h,w
    pipeline += gp.Squeeze([raw], axis=1)
    pipeline += gp.Squeeze(
        [raw, gt_affs, pred_affs, gt_lsds, pred_lsds, labels, aff_weights, lsd_weights]
    )

    pipeline += gp.Snapshot(
        output_filename="batch_{iteration}.zarr",
        output_dir="snapshots/" + run_name,
        dataset_names={
            raw: "raw",
            labels: "labels",
            gt_affs: "gt_affs",
            gt_lsds: "gt_lsds",
            pred_affs: "pred_affs",
            pred_lsds: "pred_lsds",
            #    aff_weights: "aff_weights",
            #    lsd_weights: "lsd_weights",
        },
        every=100,
    )

    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)


if __name__ == "__main__":
    train(10000)
