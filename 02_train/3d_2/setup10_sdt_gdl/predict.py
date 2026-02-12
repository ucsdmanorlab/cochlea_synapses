import glob
import gunpowder as gp
import numpy as np
import sys
import torch
import zarr
import time
from funlib.learn.torch.models import UNet, ConvPass

import os
import sys
project_root = os.path.expanduser('~/workspace/cochlea_synapses/')
sys.path.append(project_root)
from utils import save_out

def predict(
    checkpoint,
    raw_file,
    raw_dataset):

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    voxel_size = gp.Coordinate((4,1,1))

    input_shape = gp.Coordinate((44,172,172))
    output_shape = gp.Coordinate((24,80,80))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
 
    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    in_channels = 1
    num_fmaps = 12
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
    torch.cuda.set_device(0)

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
            ConvPass(num_fmaps, 1, [[1,]*3], activation='Tanh'))
    
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
    source += gp.Pad(raw, context)
    source += gp.Normalize(raw)
    source += gp.Unsqueeze([raw])

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
            0: pred
        })

    scan = gp.Scan(scan_request)

    pipeline = source
    # d,h,w

    pipeline += gp.Stack(1)
    # b,d,h,w

    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw, pred])
    pipeline += gp.Squeeze([raw, pred])

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi
    
    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred].data


if __name__ == '__main__':
    start_t = time.time()

    model = 'model_full_run_MSE1_GDL5__cpu12_cx2_nwx20_MSE1.0GDL5.0' #model_2024-11-19_11-35-27_3d_sdt_IntWgt_b1_libcil' #'model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_dil1'
    checkpoint = model+'_checkpoint_3000'
    raw_files = glob.glob('../../../01_data/zarrs/2812*_a_*.zarr')#train/spinning*.zarr')#validate/*.zarr') 
    #validate/spinning*.zarr') #validate/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('01_data/zarrs/',#/validate', 
            '03_predict/3d/0'+checkpoint))

        raw_dataset = f'raw' #zarr.open(raw_file)['3d/raw'][:]
        raw_res = zarr.open(raw_file)[raw_dataset].attrs['resolution']

        pred = predict(
                checkpoint,
                raw_file,
                raw_dataset) 

        pred = np.array(pred)

        save_out(out_file, zarr.open(raw_file)['raw'][:], 'raw', save2d=False, res=raw_res)
        save_out(out_file, pred, 'pred', save2d=False, res=raw_res)
        #save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels',save2d=False)
    
    print("elapsed time: "+str(time.time()-start_t))
