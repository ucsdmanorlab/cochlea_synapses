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

base_dir = '../../../01_data/zarrs/glu/training'

samples = glob.glob(base_dir + '/*.zarr')

voxel_size = gp.Coordinate((1,)*3)
#         zarr.open(base_dir+'/sample_0.zarr')['raw'].attrs['resolution']
# )

input_size = gp.Coordinate((64,)*3) * voxel_size
output_size = gp.Coordinate((24,)*3) * voxel_size

batch_size = 10

def calc_max_padding(
        output_size,
        voxel_size):

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)
    diagz = np.sqrt(output_size[1]**2 + output_size[0]**2)
    max_padding = np.ceil(
            [i/2 + j for i,j in zip(
                [diag, diag, diag],
                list(voxel_size)
                )]
            )

    return gp.Coordinate(max_padding)


class ToDtype(gp.BatchFilter):

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def process(self, batch, request):

        data = batch[self.array].data

        batch[self.array].data = data.astype(self.dtype)


def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    pred_mask = gp.ArrayKey("PRED_MASK")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(pred_mask, output_size)

    num_fmaps=12

    ds_fact = [(2,2,2),(2,2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3,3),(3,3,3)]]*num_levels
    ksu = [[(3,3,3),(3,3,3)]]*(num_levels-1)

    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*3], activation=None))

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:'3d/raw',
                    labels:'3d/seg',
                    labels_mask:'3d/mask'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    labels:gp.ArraySpec(interpolatable=False),
                    labels_mask:gp.ArraySpec(interpolatable=False)
                }) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Pad(labels_mask, padding) +
                gp.Normalize(raw) +
                gp.RandomLocation(mask=labels_mask) #+
                #gp.Reject(mask=labels_mask, min_masked=0.1)
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()

    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(16,)*3,
                    jitter_sigma=(5.0,)*3,
                    rotation_interval=(0,math.pi/2))

    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    # pipeline += ToDtype(labels, np.float32)

    pipeline += gp.Unsqueeze([raw, labels])

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
            0: pred_mask
        },
        loss_inputs={
            0: pred_mask,
            1: labels
        },
        log_dir='log',
        save_every=5000)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w
    # labels_mask: b,d,h,w
    # pred_mask: b,c,d,h,w

#    pipeline += gp.Squeeze([raw, labels, labels_mask])

    # raw: c,d,h,w
    # labels: c,d,h,w
    # labels_mask: d,h,w

#    pipeline += gp.Squeeze([raw, labels])

    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            dataset_names={
                raw: "raw",
                labels: "labels",
                labels_mask: "labels_mask",
                pred_mask: "pred_mask"
            },
            every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(10000)
