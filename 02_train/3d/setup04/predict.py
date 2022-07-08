import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import torch
import zarr
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from time import time

from funlib.learn.torch.models import UNet, ConvPass

def predict(
    checkpoint,
    raw_file,
    raw_dataset):

    raw = gp.ArrayKey('RAW')
    pred_mask = gp.ArrayKey('PRED_MASK')

    input_shape = gp.Coordinate((36,172,172))
    output_shape = gp.Coordinate((16,80,80))

    voxel_size = gp.Coordinate((5,1,1))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_shape - output_shape) / 2
    context = context * voxel_size

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_mask, output_size)

    in_channels = 1
    num_fmaps = 12
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

    num_levels = len(downsample_factors) + 1

    unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1]*3], activation='Tanh'))
    
    source = gp.ZarrSource(
        raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)
            })

    source += gp.Normalize(raw)
    
    with gp.build(source):
        total_output_roi = source.spec[raw].roi
        total_input_roi = total_output_roi.grow(context,context)

        print(total_input_roi)
        print(total_output_roi)

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
    pipeline += gp.Pad(raw, context)
    # d, h, w
    
    pipeline += gp.Unsqueeze([raw])
    # d,h,w
    pipeline += gp.IntensityScaleShift(raw, 2,-1)
    pipeline += gp.Stack(1)
    # b,d,h,w
     
    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw, pred_mask])
    pipeline += gp.Squeeze([raw, pred_mask])
    # d,h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end()) #total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi #total_input_roi
    predict_request.add(pred_mask, total_output_roi.get_end())
    predict_request[pred_mask].roi = total_output_roi
    
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_mask].data


if __name__ == '__main__':

    checkpoint = 'model_checkpoint_100000'
    raw_files = sorted(glob.glob('../../../01_data/zarrs/ctbp2/restest/validation/*08.zarr'))
    
    for raw_file in raw_files:
        print(raw_file)
        out_file_path = (raw_file.replace('validation', 'out'))
        out_file = zarr.open(out_file_path)


        raw_dataset = f'raw' #zarr.open(raw_file)['3d/raw'][:]
        
        pred = predict(
                checkpoint,
                raw_file,
                raw_dataset,
                ) #[]

        pred = np.array(pred)
                
        thresh = 0.05 #threshold_otsu(pred)
        tic = time()
        thresholded = pred > 0.5
        pred_gaus = pred #gaussian(pred, sigma=1.5, preserve_range=True)
        markers = h_maxima(pred_gaus*thresholded, thresh) #label(thresholded)

        labeled = watershed(np.multiply(pred, -1), label(markers*thresholded), mask = thresholded)


        out_file['label_sdt_noaug'] = labeled.astype(np.uint64)
        out_file['label_sdt_noaug'].attrs['offset'] = [0,]*3 #[10, 46, 46] #[0,]*3
        out_file['label_sdt_noaug'].attrs['resolution'] = [1,]*3

        out_file['raw'] = zarr.open(raw_file)['raw'][:]
        out_file['raw'].attrs['offset'] = [0,]*3
        out_file['raw'].attrs['resolution'] = [1,]*3

        out_file['pred_sdt_noaug'] = pred
        out_file['pred_sdt_noaug'].attrs['offset'] = [0,]*3 #[10, 46, 46] #[0,]*3
        out_file['pred_sdt_noaug'].attrs['resolution'] = [1,]*3

