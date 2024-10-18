#!python
#cython: language_level=3

import shutil
import pandas as pd
import waterz
import glob
import gunpowder as gp
import numpy as np
import math
import os
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import regionprops_table, label
#from skimage.filters import threshold_otsu
#from gunpowder import *
#from gunpowder.ext import torch
#from gunpowder.torch import *
#from torch.nn.functional import binary_cross_entropy
from natsort import natsorted 

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

def calc_errors(labels, gt_xyz, mode='2d'):

    gtp = gt_xyz.shape[0]
    Npoints_pred = len(np.unique(labels))-1

    pred_idcs = np.zeros(gtp).astype(np.int32)

    for i in range(gtp):

        cent = np.around(gt_xyz[i,:]).astype(np.int16)
        if mode=='2d':
            pred_idcs[i] = labels[cent[1], cent[0]]
        elif mode=='3d':
            pred_idcs[i] = labels[cent[2], cent[1], cent[0]]

    idc, counts = np.unique(pred_idcs[pred_idcs>0], return_counts=True)

    tp = np.sum(counts==1)
    fm = np.sum(counts>1)
    fn = np.sum(pred_idcs==0)
    fp = Npoints_pred - len(idc)   
    
    ap = tp/(tp+fn+fp+fm)
    f1 = 2*tp/(2*tp + fp + fn + fm)
    
    prec = tp/(tp+fp)
    rec  = tp/gtp

    results = {
        "gtp": gtp,
        "tp": tp,
        "fn": fn,
        "fm": fm, 
        "fp": fp, 
        "ap": ap,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "mergerate": fm/gtp,
    }
    
    return results 



if __name__ == '__main__':

    model = 'model_2022-09-13_17-26-28_twoclass_wgtsurr3px'
    
    checkpoints = natsorted('./'+model+'_checkpoint_*')
    raw_fi_dir = '../../../01_data/zarrs/validate/'
    out_dir = '../../../03_predict/2d/'+model+'/'

    raw_files = glob.glob(raw_fi_dir+'*.zarr')

    thresh = 0.5
    save_pred_labels = False
    
    AllResults = pd.DataFrame()
    count = 0
    
    for raw_file in raw_files:
        print('Predicting on %s...', raw_file)
        out_file = zarr.open(raw_file.replace(raw_fi_dir, out_dir))

        full_raw = zarr.open(raw_file)['3d/raw']
        save_out(out_file, full_raw[:], 'raw')

        for checkpoint in tqdm(checkpoints):
            pred = []

            for i in range(full_raw.shape[0]):

                raw_dataset = f'2d/raw/{i}'
                
                pred_mask = predict(
                    checkpoint,
                    raw_file,
                    raw_dataset)

                pred.append(pred_mask)

            pred = np.array(pred)
            
            checkpoint_num = checkpoint[checkpoint.rfind('_')::]
            save_out(out_file, pred, 'pred'+checkpoint_num)

        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')
        
        img = zarr.open(out_file)
        gt = img['gt_labels'][:]

        print('getting centroids...')
        gt_xyz = regionprops_table(gt, properties=('centroid','label'))
        gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T
        pred_list = natsorted(glob.glob(out_file+'/pred_*'))
    
        for pred_fi in pred_list
            pred_fi = pred_fi[pred_fi.rfind('pred')::]
            pred = img[pred_fi][:]
            print(pred_fi)
    
            segmentation = label(pred>thresh)
    
            #if save_pred_labels:
            #    print('saving labels...')
            #    if os.path.exists(os.path.join(out_file, 'pred_labels')):
            #        shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))
    
            #    print('saving')
            #    img['pred_labels'] = segmentation.astype(np.uint64)
            #    img['pred_labels'].attrs['offset'] = [0,]*3
            #    img['pred_labels'].attrs['resolution'] = [1,]*3
    
            print('calculating stats...')
            results= {"file": out_file,
                     "thresh": thresh,
                     "iter": pred_fi[pred_fi.rfind('_')+1::],
                     }
            results.update(
                calc_errors(
                    segmentation, 
                    gt_xyz, 
                    mode='3d')
            )
            AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
            count = count + 1

    
    AllResults.to_csv(os.path.join(out_dir,'val_stats.csv'),encoding='utf-8-sig')
    
    bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    AllResults.iloc[bestlist].to_csv(os.path.join(out_dir,'best_stats.csv'), encoding='utf-8-sig')


