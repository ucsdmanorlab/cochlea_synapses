#!python
#cython: language_level=3

import numpy as np
import shutil
import pandas as pd
import waterz
import zarr
import os

from skimage.measure import regionprops_table, label
val_dir = '../../../03_predict/3d/model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_checkpoint_10000'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]

thresh_list = [0, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7]
save_pred_labels = False

AllResults = pd.DataFrame()
count = 0

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

for fi in val_samples:
    print(fi)
    img = zarr.open(os.path.join(val_dir, fi))
    gt = img['gt_labels'][:]
    pred = img['pred'][:]
    
    print('getting centroids...')
    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T

    for thresh in thresh_list:
        print(thresh)

        segmentation = label(pred>thresh)

        if save_pred_labels:
            print('saving labels...')
            if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

            print('saving')
            img['pred_labels'] = segmentation.astype(np.uint64)
            img['pred_labels'].attrs['offset'] = [0,]*3
            img['pred_labels'].attrs['resolution'] = [1,]*3

        print('calculating stats...')
        results= {"file": fi,
                 "thresh": thresh,
                 }
        results.update(
            calc_errors(
                segmentation, 
                gt_xyz, 
                mode='3d')
        )
        AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
        count = count + 1

AllResults.to_csv(os.path.join(val_dir,'val_stats.csv'),encoding='utf-8-sig')

bestlist = list(AllResults.groupby('file').idxmax()['f1'])
AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')


