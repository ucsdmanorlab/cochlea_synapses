import glob
import gunpowder as gp
import numpy as np
import math
import os
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label
from skimage.filters import threshold_otsu
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from torch.nn.functional import binary_cross_entropy
from skimage.transform import resize

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
        in_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)
    
    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1]*2], activation=None), 
            torch.nn.Sigmoid())

    torch.cuda.set_device(1) 
    
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

    checkpoint = glob.glob('model_*_8000')[0]
    raw_files = glob.glob('../../../01_data/zarrs/validate/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('validate', checkpoint))
        full_raw = zarr.open(raw_file)['3d/raw']
        
        if "confocal" in raw_file:
            raw_open = zarr.open(raw_file, 'a')

            scale = np.array([1, 2, 2])
            in_shape = full_raw.shape
            out_shape = np.round(np.asarray(in_shape) * scale).astype('int64')
            raw_dtype = full_raw.dtype
            
            print(in_shape)
            raw_resize = resize(np.asarray(full_raw), out_shape,
                        preserve_range=True).astype(raw_dtype)
            print(raw_resize.shape)
            
            for i in range(full_raw.shape[0]):
                print(f'saving rescaled section {i}')
                
                raw_open[f'2d/raw_resize/{i}'] = np.expand_dims(raw_resize[i], axis=0)
                raw_open[f'2d/raw_resize/{i}'].attrs['offset'] = [0,]*2
                raw_open[f'2d/raw_resize/{i}'].attrs['resolution'] = [1,]*2
        
            pred_on = '2d/raw_resize/'
            save_out(out_file, raw_resize, 'raw')
            
            labels = raw_open['3d/labeled']

            save_out(out_file, resize(np.asarray(labels[:]), out_shape, order=0, 
                preserve_range=True, anti_aliasing=False), 'gt_labels')
        else:
            pred_on = '2d/raw/'
            save_out(out_file, full_raw[:], 'raw')
            save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')

        pred = []

        for i in range(full_raw.shape[0]):

            print(f'predicting on section {i}')

            raw_dataset = f'{pred_on}{i}'

            pred_mask = predict(
                checkpoint,
                raw_file,
                raw_dataset)

            pred.append(pred_mask)

        pred = np.array(pred)
        
        save_out(out_file, pred, 'pred')
