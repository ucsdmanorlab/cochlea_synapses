import numpy as np
import shutil
import pandas as pd
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

val_dir = '/media/caylamiller/DataDrive/synapses_tscc/model_full_run_MSE1_GDL5__cpu12_cx2_nwx20_MSE1.0GDL5.0/' #'../../../01_data/zarrs/pssr'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]

gt_dir = '../../../01_data/zarrs/validate'

gt_samples = [i for i in os.listdir(gt_dir) if i.endswith('.zarr')]

thresh_list = [0.0, 0.1, 0.2, 0.4, 0.6]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]#[0.25] #[ 0.15, 0.2, 0.25, 0.3, 0.35]
save_pred_labels = False
save_csvs = True
size_filt_list = [1] #, 100, 150] #[1,2,3,4,5,7,10,15]
blur_sig = [0.5, 0.7, 0.7]

AllResults = pd.DataFrame()
count = 0

for fi in val_samples:
    print(fi)
    if fi not in gt_samples: #if 'spinningdisk_'+fi not in gt_samples:
        print("can't find GT")
        continue
    img = zarr.open(os.path.join(val_dir, fi))
    gt_img = zarr.open(os.path.join(gt_dir, fi))
    gt = gt_img['3d/labeled'][:]
    pred = img['pred_2000'][:]

    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T

    for thresh in thresh_list:
        print("thresh: ", thresh) #, " (", np.where(thresh==np.array(thresh_list))[0][0]+1, " out of ", len(thresh_list), ")")
        peak_thresh = thresh
        mask_thresh = 0.0
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

        segmentation_filt = segmentation
        if save_pred_labels:
            print('saving labels...')
            if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

            print('saving')
            img['gt_labels'] = gt
            img['gt_labels'].attrs['offset'] = [0,]*3
            img['gt_labels'].attrs['resolution'] = [1,]*3

            img['pred_labels'] = segmentation_filt.astype(np.uint64)
            img['pred_labels'].attrs['offset'] = [0,]*3
            img['pred_labels'].attrs['resolution'] = [1,]*3
        if save_csvs:
            print('calculating stats...')
            results= {"file": fi,
                        "thresh": thresh,
                        "size_thresh": 1,
                        }
            results.update(
                calc_errors(
                    segmentation_filt, 
                    gt_xyz, 
                    )
            )
            intersection = np.logical_and(gt > 0, segmentation_filt > 0).sum()
            union = np.logical_or(gt > 0, segmentation_filt > 0).sum()
            iou = intersection / union if union != 0 else 0.0

            gt_sum = (gt > 0).sum()
            pred_sum = (segmentation_filt > 0).sum()
            dice = (2.0 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) != 0 else 0.0
            
            results['iou'] = iou
            results['dice'] = dice
            print(results['f1'])
            AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
        count = count + 1
if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats_msk0.csv'),encoding='utf-8-sig')
    
elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
