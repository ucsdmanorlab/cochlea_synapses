import glob
import gunpowder as gp
import numpy as np
import math
import os
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label
from skimage.filters import threshold_otsu
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from torch.nn.functional import binary_cross_entropy

def save_out(outfile, 
        array, 
        key, 
        res=None,
        offset=None):

    if res is None:
        res = [1,]*len(array.shape)
    if offset is None:
        offset = [0,]*len(array.shape)

    out_file[key] = array
    out_file[key].attrs['offset'] = offset
    out_file[key].attrs['resolution'] = res

def predict(
    checkpoint,
    raw_file,
    raw_dataset):

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    voxel_size = gp.Coordinate((1,)*2)
    input_size = gp.Coordinate((140,)*2) * voxel_size
    output_size = gp.Coordinate((100,)*2) * voxel_size

    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    num_fmaps=30

    ds_fact = [(2,2), (2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3),(3,3)]]*num_levels
    ksu = [[(3,3),(3,3)]]*(num_levels-1)

    unet = UNet(
        in_channels=5,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 2, [[1,]*2], activation='Sigmoid'))
    
    source = gp.ZarrSource(
        raw_file,
        datasets={
            raw: raw_dataset,
        },
        array_specs={
            raw: gp.ArraySpec(interpolatable=True),
            })

    with gp.build(source):
        total_output_roi = source.spec[raw].roi
        total_input_roi = total_output_roi.grow(context, context)

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred
        })

    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)
    pipeline += gp.Pad(raw, context)

    # h, w
    
    pipeline += gp.Stack(1)
    # b, h, w
    pipeline += predict
    pipeline += scan

    pipeline += gp.Squeeze([raw, pred])
    # h, w
    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi
    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred].data


if __name__ == '__main__':

    checkpoint = sys.argv[1] #'model_2022-09-28_17-03-31_affs_checkpoint_20000'
    checkpoint_num = checkpoint[checkpoint.rfind('_')::]
    model = checkpoint[0:checkpoint.rfind('_')]
    print(model)

    raw_files = glob.glob('../../../01_data/zarrs/validate/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('01_data/zarrs/validate', '03_predict/2p5d/'+model))

        full_raw = zarr.open(raw_file)['3d/raw']
        save_out(out_file, full_raw[:], 'raw')

        pred = []

        for i in range(full_raw.shape[0]):

            print(f'predicting on section {i}')

            raw_dataset = f'2p5d/raw/{i}'

            pred_mask = predict(
                checkpoint,
                raw_file,
                raw_dataset)

            pred.append(pred_mask)

        pred = np.array(pred)


        save_out(out_file, pred, 'pred'+checkpoint_num)
        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')


