import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label
from skimage.filters import threshold_otsu

def predict(
    checkpoint,
    raw_file,
    raw_dataset):

    raw = gp.ArrayKey('RAW')
    pred_mask = gp.ArrayKey('PRED_MASK')

    voxel_size = gp.Coordinate((1,)*2)

    input_size = gp.Coordinate((132,)*2) * voxel_size
    output_size = gp.Coordinate((92,)*2) * voxel_size

    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_mask, output_size)

    num_fmaps=12

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

    source = gp.ZarrSource(
        raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)
            }
        )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred_mask
        })

    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)

    pipeline += gp.Stack(1)
    pipeline += predict
    pipeline += scan

    pipeline += gp.Squeeze([raw, pred_mask])

    pipeline += gp.Squeeze(
        [raw,
         pred_mask])

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(pred_mask, total_output_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_mask].data


if __name__ == '__main__':

    checkpoint = 'model_checkpoint_10000'
    raw_file = '../../../01_data/zarrs/sample_0.zarr'

    out_file = zarr.open('test.zarr', 'a')

    full_raw = zarr.open(raw_file)['3d/raw'][:]

    pred = []

    for i in range(full_raw.shape[0]):

        print(f'predicting on section {i}')

        raw_dataset = f'2d/raw/{i}'

        pred_mask = predict(
            checkpoint,
            raw_file,
            raw_dataset)

        pred.append(pred_mask)

    pred = np.array(pred)

    thresh = threshold_otsu(pred)
    thresholded = pred >= thresh

    labeled = label(thresholded).astype(np.uint64)

    out_file['raw'] = full_raw
    out_file['raw'].attrs['offset'] = [0,]*3
    out_file['raw'].attrs['resolution'] = [1,]*3

    out_file['pred_mask'] = pred
    out_file['pred_mask'].attrs['offset'] = [0,10,10]
    out_file['pred_mask'].attrs['resolution'] = [1,]*3

    out_file['labeled'] = labeled
    out_file['labeled'].attrs['offset'] = [0,10,10]
    out_file['labeled'].attrs['resolution'] = [1,]*3
