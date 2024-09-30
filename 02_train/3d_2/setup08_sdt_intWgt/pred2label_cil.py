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
from utils import calc_errors

start_t = time.time()

val_dir = '../../../03_predict/3d/cilcare_all/sdt' #model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_checkpoint_10000'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]
#3653-L-16-kHz.zarr
thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #-0.05, 0, 0.05, 0.1, 0.15] #[0.3,0.4, 0.5,0.6, 0.7]
save_pred_labels = False #True
save_csvs = True
blur_sig = [0.5, 0.7, 0.7]

AllResults = pd.DataFrame()
count = 0
#val_samples = [i for i in val_samples if '3532' in i]

for fi in val_samples:
    print("starting ", fi, " ...getting centroids...")
    img = zarr.open(os.path.join(val_dir, fi))
    gt = img['gt_labels'][:]
    pred = img['pred_11k'][:]
    
    mask = img['convex'][:]
    pred[mask==0] = -1
    gt[mask==0] = 0
    dist_map = gaussian_filter(pred, blur_sig)

    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T
    
    for thresh in thresh_list:
        mask_thresh = 0.0
        peak_thresh = thresh
        
        print("thresh: ", thresh, " (", np.where(thresh==np.array(thresh_list))[0][0]+1, " out of ", len(thresh_list), ")")
        
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
        if save_pred_labels:
            print('saving labels...')
            if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

            # img['dist_map'] = dist_map
            # img['dist_map'].attrs['offset'] = [0,]*3
            # img['dist_map'].attrs['resolution'] = [1,]*3
            # img['markers'] = markers
            # img['markers'].attrs['offset'] = [0,]*3
            # img['markers'].attrs['resolution'] = [1,]*3
            img['pred_labels_11k'] = segmentation.astype(np.uint64)
            img['pred_labels_11k'].attrs['offset'] = [0,]*3
            img['pred_labels_11k'].attrs['resolution'] = [1,]*3
        if save_csvs:
            print('calculating stats...')
            try:
                results= {"file": fi,
                         "thresh": thresh,
                         }
                results.update(
                    calc_errors(
                        segmentation, 
                        gt_xyz, 
                        )
                )
                print(np.mean(results['f1']))
            except:
                continue
            AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
        count = count + 1
if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats_11k.csv'),encoding='utf-8-sig')
    
    #bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    #AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
