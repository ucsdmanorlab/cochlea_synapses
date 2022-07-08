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

base_dir = '../../../01_data/zarrs/ctbp2/8-bit/training'

zarrlist = glob.glob(base_dir + '/*.zarr')

voxel_size = gp.Coordinate((1,)*2)

input_size = gp.Coordinate((140,)*2) * voxel_size
output_size = gp.Coordinate((100,)*2) * voxel_size

samples = {}
for i in range(len(zarrlist)):
    samples[zarrlist[i]] = len(glob.glob(zarrlist[i]+'/2d/raw/*'))

batch_size = 32

def calc_max_padding(
        output_size,
        voxel_size):

    diag = np.sqrt(output_size[0]**2 + output_size[1]**2)
    
    max_padding = np.ceil(
            [i/2 + j for i,j in zip(
                [output_size[0], diag, diag],
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

    num_fmaps=20

    ds_fact = [(2,2), (2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3),(3,3)]]*num_levels
    ksu = [[(3,3),(3,3)]]*(num_levels-1)

    unet = UNet(
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*2], activation='Sigmoid'))

    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=5e-5, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:f'2d/raw/{i}',
                    labels:f'2d/seg/{i}',
                    labels_mask:f'2d/mask/{i}'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True),
                    labels:gp.ArraySpec(interpolatable=False),
                    labels_mask:gp.ArraySpec(interpolatable=False)
                }) +
                gp.Squeeze([raw,labels,labels_mask]) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.Pad(labels_mask, padding) +
                gp.Normalize(raw) +
                gp.RandomLocation(mask=labels_mask)
                for sample, num_sections in samples.items() 
                for i in range(num_sections))

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()

    pipeline += gp.IntensityAugment(raw, 0.6, 1.1, -0.2, 0.2)
    
    pipeline += gp.NoiseAugment(raw)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*2,
                    jitter_sigma=(5.0,)*2,
                    rotation_interval=(0,math.pi/2))

    # raw: h,w
    # labels: h,w
    # labels_mask: h,w

    # pipeline += ToDtype(labels, np.float32)

    pipeline += gp.Unsqueeze([raw, labels])

    # raw: c,h,w
    # labels: c,h,w
    # labels_mask: c,h,w

    pipeline += gp.Stack(batch_size)

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w

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

    # raw: b,c,h,w
    # labels: b,c,h,w
    # labels_mask: b,c,h,w
    # pred_mask: b,c,h,w

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

    train(20000)
