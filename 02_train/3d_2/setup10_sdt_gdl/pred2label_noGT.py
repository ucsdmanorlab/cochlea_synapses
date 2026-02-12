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
from utils import save_out, sdt_to_labels

start_t = time.time()

val_dir = project_root+'03_predict/3d/model_2025-02-21_10-32-24_MSE0GDL1_checkpoint_2000'#model_2024-11-19_11-35-27_3d_sdt_IntWgt_b1_libcil_checkpoint_10000' #_predict/3d/cilcare5/sdt' #model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1_checkpoint_10000'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('ctbp2.zarr')]

mask_thresh = 0.0 #-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
peak_thresh = 0.1 #, 0.2, 0.4]
size_thresh = 1
strict_peak_thresh = True

for fi in val_samples:
    if os.path.exists(os.path.join(fi, 'pred_labels')):
        #print("skipping ", fi)
        continue
    if not os.path.exists(os.path.join(fi, 'pred')):
         print("no pred found, skipping ", fi)
         continue
    img = zarr.open(os.path.join(val_dir, fi))
    pred = img['pred'][:]
    
    segmentation = sdt_to_labels(
                    pred, 
                    peak_thresh=peak_thresh,
                    mask_thresh=mask_thresh,
                    strict_peak_thresh=strict_peak_thresh,
                    size_filt=size_thresh)
                
    print('saving labels...')
    if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
        shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))
    save_out(img, segmentation, 'pred_labels', save2d=False)    

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
