import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import torch
import zarr
import rescale_cmm2
from skimage.transform import resize
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
    #source += rescale_cmm2.ScaleAugment(raw, tuple(scalefactor))
    #source += gp.Pad(raw, context)
    #source += gp.Unsqueeze([raw])
    
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
    # pipeline += gp.Normalize(raw)
    #pipeline += rescale_cmm2.ScaleAugment(raw, tuple(scalefactor))
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
    #pipeline += rescale_cmm2.ScaleAugment(pred_mask, tuple(1/scalefactor))
    # d,h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end()) #total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi #total_input_roi
    predict_request.add(pred_mask, total_output_roi.get_end())
    predict_request[pred_mask].roi = total_output_roi
    
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    print('raw and pred shapes:')
    print(np.asarray(batch[raw].data).shape)
    print(np.asarray(batch[pred_mask].data).shape)

    return batch[pred_mask].data


if __name__ == '__main__':

    checkpoint = 'model_checkpoint_50000'
#    raw_file = '../../../01_data/zarrs/sample_0.zarr'
    raw_files = sorted(glob.glob('../../../01_data/zarrs/nih/ctbp2_restest/*.zarr'))
    
    for raw_file in raw_files:
        print(raw_file)
        out_file_path = (raw_file.replace('restest', 'prediction'))
        out_file = zarr.open(out_file_path)

        scalefactor = np.asarray(zarr.open(raw_file)['3d/raw'].attrs['scalefactor'])
        whichscale =  ~np.multiply(scalefactor > 0.8, scalefactor < 1.2) # don't bother scaling differences <20%
        scalefactor = (np.multiply(scalefactor, whichscale) +
                np.multiply(1, ~whichscale))
        
        inshape = zarr.open(raw_file)['3d/raw'].shape
        outshape = np.round(np.asarray(inshape) * scalefactor).astype('int64')
        raw_min = np.min(zarr.open(raw_file)['3d/raw'])
        raw_max = np.max(zarr.open(raw_file)['3d/raw'])
        
        print(inshape)
        print(outshape)

        raw_resize = resize(zarr.open(raw_file)['3d/raw'], outshape)
        raw_resize = (((raw_max-raw_min) / (np.max(raw_resize) - np.min(raw_resize))) 
            * (raw_resize - np.min(raw_resize))) + raw_min

        raw_resize = raw_resize.astype(zarr.open(raw_file)['3d/raw'].dtype)
        
        out_file['raw_resize'] = raw_resize
        out_file['raw_resize'].attrs['offset'] = [0,]*3
        out_file['raw_resize'].attrs['resolution'] = [1,]*3
        del raw_resize

        raw_dataset = f'raw_resize' #f'3d/raw' #zarr.open(raw_file)['3d/raw'][:]
        
        pred = predict(
                checkpoint,
                out_file_path,
                raw_dataset,
                ) #[]

        pred = np.array(pred)
                
        pred_min = np.min(pred)
        pred_max = np.max(pred)
        print(inshape, pred.shape)
        print(pred_min, pred_max)
        
        pred = resize(pred, inshape)

        pred = (((pred_max-pred_min) / (np.max(pred) - np.min(pred))) * 
                (pred - np.min(pred))) + pred_min

        out_file['raw'] = zarr.open(raw_file)['3d/raw'][:]
        out_file['raw'].attrs['offset'] = [0,]*3
        out_file['raw'].attrs['resolution'] = [1,]*3

        out_file['pred_mask'] = pred
        out_file['pred_mask'].attrs['offset'] = [0,]*3 #[10, 46, 46] #[0,]*3
        out_file['pred_mask'].attrs['resolution'] = [1,]*3

