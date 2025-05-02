import glob
import gunpowder as gp
import numpy as np
import os
import pandas as pd
import torch
import zarr
import time
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label, regionprops_table
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import calc_errors, save_out

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
    torch.cuda.set_device(1)

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

    #model = '2025-03-13_12-58-35_MSE0GDL1_rej1_end0.8_step0.01_every1'
    model = '2025-03-14_22-55-23_MSE1GDL20_rej1_end0.8_step0.01_every10_1k4k'
    #model = '2025-03-14_14-41-23_MSE0.05GDL1_rej1_end0.8_step0.01_every1_1k4k'
    
    raw_files = glob.glob('../../../01_data/zarrs/validate/*.zarr')
    
    # parameters: ###########
    mask_thresh = 0.0
    peak_thresh = 0.1
    blur_sig = [0.5, 0.7, 0.7]
    save_zarrs = False
    save_csvs = True
    gauss_blur = False

    AllResults = pd.DataFrame()
    count = 0
    checkpoints = glob.glob('./models/model_'+model+'_checkpoint_*')

    for raw_file in raw_files:
        print(raw_file)
        if save_zarrs:
            out_file = zarr.open(raw_file.replace('01_data/zarrs','03_predict/3d/'+model))
            save_out(out_file, zarr.open(raw_file)['raw'][:], 'raw', save2d=False)
            save_out(out_file, zarr.open(raw_file)['labeled'][:], 'gt_labels',save2d=False)
        
        if save_csvs:
            gt = zarr.open(raw_file)['labeled'][:]
            gt_xyz = regionprops_table(gt, properties=('centroid','label'))
            gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T

        for checkpoint in checkpoints:
            print(checkpoint)
            checkpoint_num = checkpoint[checkpoint.rfind('_')::]

            pred = predict(
                    checkpoint,
                    raw_file,
                    f'raw') 
            pred = np.array(pred)
            
            if save_zarrs:
                save_out(out_file, pred, 'pred'+checkpoint_num , save2d=False)

            
            dist_map = pred
            if gauss_blur:
                dist_map = gaussian_filter(pred, blur_sig)
            coords = peak_local_max(
                    dist_map,
                    footprint=np.ones((3, 3, 3)),
                    threshold_abs=peak_thresh,
                    min_distance=2,
                    )
            mask = np.zeros(dist_map.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers = label(mask)
            mask = pred>(mask_thresh)
            markers = markers*mask
            segmentation = watershed(-dist_map, markers, mask=mask)

            if save_zarrs:
                save_out(out_file, segmentation, 'pred_labels'+checkpoint_num, save2d=False)
            if save_csvs:
                results= {"file": os.path.split(raw_file)[1],
                             "mask_thresh": mask_thresh,
                             "peak_thresh": peak_thresh,
                             "blur": [blur_sig] if gauss_blur else [(0,0,0)],
                             "checkpoint": int(checkpoint_num[1::]),
                             }
                results.update(
                        calc_errors(
                            segmentation,
                            gt_xyz,
                            )
                        )
                AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
            count = count + 1
    if save_csvs:
        AllResults.to_csv('stats/val_stats'+model+'.csv',encoding='utf-8-sig')
        
    print("elapsed time: "+str(time.time()-start_t))
