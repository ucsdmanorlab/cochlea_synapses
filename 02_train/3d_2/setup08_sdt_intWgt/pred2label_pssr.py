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

val_dir = '../../../03_predict/3d/pssr_control' #'../../../01_data/zarrs/pssr'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]

cand_dirs=['../../../01_data/zarrs/train', '../../../01_data/zarrs/validate']

gt_samples = [(cand_dir, i) for cand_dir in cand_dirs for i in os.listdir(cand_dir) if i.startswith('spinning')]
gt_dirs = [i for (i,j) in gt_samples]
gt_samples = [j for (i,j) in gt_samples]

thresh_list = [-0.1, 0.0, 0.1, 0.2, 0.4, 0.6]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]#[0.25] #[ 0.15, 0.2, 0.25, 0.3, 0.35]
save_pred_labels = False
save_csvs = True
size_filt_list = [1] #, 100, 150] #[1,2,3,4,5,7,10,15]
blur_sig = [0.5, 0.7, 0.7]

AllResults = pd.DataFrame()
count = 0

for fi in gt_samples:
    print(fi, gt_samples.index(fi), gt_dirs[gt_samples.index(fi)])

for fi in val_samples:
    print(fi)
    if fi not in gt_samples: #if 'spinningdisk_'+fi not in gt_samples:
        continue
    gt_id = gt_samples.index(fi) #'spinningdisk_'+fi)
    print("starting ", fi, " ...getting centroids...")
    img = zarr.open(os.path.join(val_dir, fi))
    gt_img = zarr.open(os.path.join(gt_dirs[gt_id], gt_samples[gt_id]))
    gt = gt_img['3d/labeled'][:]
    pred = img['pred'][:]
    gt_scale = [pred.shape[i]/gt.shape[i] for i in range(len(gt.shape))]
    gt_scale.reverse()

    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T
    gt_xyz = gt_xyz * gt_scale

    for thresh in thresh_list:
        print("thresh: ", thresh) #, " (", np.where(thresh==np.array(thresh_list))[0][0]+1, " out of ", len(thresh_list), ")")
        peak_thresh = thresh
        mask_thresh = peak_thresh-0.1
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

        for size_thresh in size_filt_list:
            if size_thresh>1:
                print(size_thresh)
                out_labels = in_labels
                out_labels[seg_props['num_pixels']<size_thresh] = 0
                segmentation_filt = map_array(segmentation, in_labels, out_labels)

            else:
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
                         "size_thresh": size_thresh,
                         }
                results.update(
                    calc_errors(
                        segmentation_filt, 
                        gt_xyz, 
                        )
                )
                print(results['f1'])
                AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
        count = count + 1
if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats.csv'),encoding='utf-8-sig')
    
    #bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    #AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
