#!python
#cython: language_level=3

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
project_root = os.path.expanduser('~/workspace/cochlea_synapses/')
sys.path.append(project_root)
from utils import save_out

start_t = time.time()

val_dir = project_root+'03_predict/3d/model_2025-02-21_10-32-24_MSE0GDL1_checkpoint_2000'#model_2024-11-19_11-35-27_3d_sdt_IntWgt_b1_libcil_checkpoint_10000' #_predict/3d/cilcare5/sdt' #model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_checkpoint_10000'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('ctbp2.zarr')]

mask_thresh = 0.0 #-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
peak_thresh = 0.1 #, 0.2, 0.4]
size_thresh = 1
strict_peak_thresh = True
blur_sig = [0.5, 0.7, 0.7]

for fi in val_samples:
    if os.path.exists(os.path.join(fi, 'pred_labels')):
        #print("skipping ", fi)
        continue
    print("starting ", fi, " ...getting centroids...")
    if not os.path.exists(os.path.join(fi, 'pred')):
         print("no pred found, skipping ", fi)
         continue
    img = zarr.open(os.path.join(val_dir, fi))
    pred = img['pred'][:]
    
    dist_map = gaussian_filter(pred, blur_sig) 
    coords = peak_local_max(
            dist_map, 
            footprint=np.ones((3, 3, 3)), 
            threshold_abs=peak_thresh, 
            min_distance=2,
            )
    markers = np.zeros(dist_map.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers = label(markers)

    mask = pred>(mask_thresh)
    markers = markers*mask
    segmentation = watershed(-dist_map, markers, mask=mask)

    if strict_peak_thresh: # remove any masked regions without a peak  over peak threshold
         strict_labels = np.unique(segmentation[tuple(coords.T)])
         for i in np.unique(segmentation):
             if i == 0:
                 continue
             if i not in strict_labels:
                 segmentation[segmentation==i] = 0
                 
    if size_thresh>1:
        print(size_thresh)
        seg_props = regionprops_table(segmentation, properties=('label', 'num_pixels'))
        in_labels = seg_props['label']
        out_labels = in_labels
        out_labels[seg_props['num_pixels']<size_thresh] = 0
        segmentation_filt = map_array(segmentation, in_labels, out_labels)

    else:
        segmentation_filt = segmentation
    print('saving labels...')
    if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
        shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

    print('saving')

    save_out(img, segmentation_filt, 'pred_labels', save2d=False)    

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
