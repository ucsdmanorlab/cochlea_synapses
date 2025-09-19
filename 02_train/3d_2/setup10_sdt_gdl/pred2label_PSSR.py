import numpy as np
import shutil
import pandas as pd
import zarr
import time
import os

from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops, regionprops_table, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.util import map_array

import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import calc_errors

start_t = time.time()

val_dir = os.path.join(project_root, '01_data/zarrs/pssr/')
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]

cand_dirs=[os.path.join(project_root, '01_data/zarrs/train'), os.path.join(project_root, '01_data/zarrs/validate')]
gt_samples = [(cand_dir, i) for cand_dir in cand_dirs for i in os.listdir(cand_dir) if i.startswith('spin')]
gt_dirs = [i for (i,j) in gt_samples]
gt_samples = [j for (i,j) in gt_samples]
print(gt_dirs, gt_samples)
pred_name = 'pred_new'

msk_thresh_list = [0.0, 0.1, 0.2, 0.3] #[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
pk_thresh_list = [0.2]
save_pred_labels = False
save_csvs = True
size_filt_list = [1] #, 100, 150] #[1,2,3,4,5,7,10,15]
blur_sig = [0.5, 0.7, 0.7]
strict_peak_thresh = True

AllResults = pd.DataFrame()
count = 0

for fi in val_samples:
    if 'spinningdisk_'+fi not in gt_samples:
        print("skipping ", fi, " ...not in gt_samples")
        continue
    else:
        print("starting ", fi, " ...getting centroids...")
    gt_id = gt_samples.index('spinningdisk_'+fi)

    img = zarr.open(os.path.join(val_dir, fi))
    gt_img = zarr.open(os.path.join(gt_dirs[gt_id], gt_samples[gt_id]))
    gt = gt_img['labeled'][:]
    pred = img[pred_name][:]
    gt_scale = [pred.shape[i]/gt.shape[i] for i in range(len(gt.shape))]
    gt_scale.reverse()
    
    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T
    gt_xyz = gt_xyz * gt_scale
    
    for thresh in msk_thresh_list:
        for pk_thresh in pk_thresh_list:
            print("thresh: ", thresh, thresh+pk_thresh) 
            peak_thresh = thresh+pk_thresh
            mask_thresh = thresh
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

            if strict_peak_thresh:
                    # find labels with peaks > pk_thresh
                    strict_labels = np.unique(segmentation[tuple(coords.T)])
                    # remove labels with no peaks
                    for i in np.unique(segmentation):
                        if i == 0:
                            continue
                        if i not in strict_labels:
                            segmentation[segmentation==i] = 0

            if save_pred_labels:
                print('saving labels...')
                if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                    shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

                print('saving')
                img['gt_labels'] = gt
                img['gt_labels'].attrs['offset'] = [0,]*3
                img['gt_labels'].attrs['resolution'] = [1,]*3

                img['pred_labels'] = segmentation.astype(np.uint64)
                img['pred_labels'].attrs['offset'] = [0,]*3
                img['pred_labels'].attrs['resolution'] = [1,]*3
            if save_csvs:
                print('calculating stats...')
                #dice = np.sum((segmentation>0) & (gt>0)) * 2.0 / (np.sum(segmentation>0) + np.sum(gt>0))
                results= {"file": fi,
                            "mask_thresh": thresh,
                            "peak_thresh": thresh+pk_thresh,
                            #"size_thresh": size_thresh,
                            #"dice": dice,
                            }
                results.update(
                    calc_errors(
                        segmentation, 
                        gt_xyz, 
                        )
                )
                print(results['f1'])
                AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
        count = count + 1
if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats_2025.csv'),encoding='utf-8-sig')
    
    #bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    #AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
