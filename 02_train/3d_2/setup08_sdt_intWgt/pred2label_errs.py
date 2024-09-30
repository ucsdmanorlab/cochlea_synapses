#!python
#cython: language_level=3

import numpy as np
import shutil
import pandas as pd
import waterz
import zarr
import time
import os

from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops_table, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.util import map_array

import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import calc_errors, save_out

start_t = time.time()

val_dir = '../../../03_predict/3d/model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_checkpoint_10000' #cilcare5/sdt' #model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_checkpoint_10000'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]

thresh_list = [0.25] #[0.3,0.4, 0.5,0.6, 0.7]
save_pred_labels = True #False
save_csvs = False #True
blur_sig = [0.5, 0.7, 0.7]
size_thresh = 4

AllResults = pd.DataFrame()
count = 0

for fi in val_samples:
    print(fi)
    img = zarr.open(os.path.join(val_dir, fi))
    gt = img['gt_labels'][:]
    pred = img['pred'][:]
    
    print('getting centroids...')
    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T
    
    y_mask = np.zeros_like(pred)

    if '3526-L-04' in fi:
        print('4')
        y_mask[:, 0:350, :] = 1
        y_mask[:, 700::, :] = 1
    elif '3532-L-16' in fi:
        print('16')
        y_mask[:, 0:150, :] = 1
        y_mask[:, 650::, :] = 1
    elif '6385-L-25' in fi:
        print('25')
        y_mask[:, 0:320, :] = 1
        y_mask[:, 720::, :] = 1
    elif '6382-L-32' in fi:
        print('32')
        y_mask[:, 0:400, :] = 1
        y_mask[:, 700::, :] = 1
    
    pred[y_mask>0] = -1

    for thresh in thresh_list:
        print(thresh)
        
        segmentation = watershed(
            gaussian_filter(pred.min()+pred.max()-pred, blur_sig),
                #markers=pks, 
                mask=(pred>thresh))

        #segmentation = label(pred>thresh)
        seg_props = regionprops_table(segmentation, properties=('label', 'num_pixels'))
        in_labels = seg_props['label']
        out_labels = in_labels
        out_labels[seg_props['num_pixels']<size_thresh] = 0
        segmentation = map_array(segmentation, in_labels, out_labels)
        errs, img_err = calc_errors(
                    segmentation,
                    gt_xyz,
                    return_img=True)
        false_neg_ids = gt[img_err==4]
        for i in false_neg_ids:
            img_err[gt==i] = 4

        if save_pred_labels:
            print('saving labels...')
            if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

            print('saving')
            img['pred_labels'] = segmentation.astype(np.uint64)
            img['pred_labels'].attrs['offset'] = [0,]*3
            img['pred_labels'].attrs['resolution'] = [1,]*3
            
            save_out(img, img_err.astype(np.uint8), 'pred_errs')
        
        if save_csvs:
            print('calculating stats...')
            results= {"file": fi,
                     "thresh": thresh,
                     }
            results.update(
                errs
            )
            AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
        count = count + 1
if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats2.csv'),encoding='utf-8-sig')
    
    bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
