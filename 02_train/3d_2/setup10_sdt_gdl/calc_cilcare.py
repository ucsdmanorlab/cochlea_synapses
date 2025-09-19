import numpy as np
import shutil
import pandas as pd
import zarr
import time
import os
import tifffile as tiff
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

val_dir = os.path.expanduser('~/Desktop/cilcare5/cayla_anno/labels')
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.tif')]
gt_dir = os.path.expanduser('~/Desktop/cilcare5/cilcare_anno/labels')

save_csvs = True
blur_sig = [0.5, 0.7, 0.7]
strict_peak_thresh = True

AllResults = pd.DataFrame()
count = 0

for fi in val_samples:
    print("starting ", fi, " ...getting centroids...")
    cayla = tiff.imread(os.path.join(val_dir, fi))
    cilcare = tiff.imread(os.path.join(gt_dir, fi))
    print(cayla.shape, cilcare.shape)
    
    y_mask = np.zeros(cayla.shape, dtype=int) 
    if '3526-L-04' in fi:
        print('4')
        y_mask[:, 0:350, :] = 1
        y_mask[:, 700::, :] = 1
    elif '3532-L-16' in fi:
        print('16')
        y_mask[:, 0:150, :] = 1
        y_mask[:, 650::, :] = 1
        y_mask[:, :, 0:30] = 1
        y_mask[:, :, 958::] = 1
    elif '6385-L-25' in fi:
        print('25')
        y_mask[:, 0:320, :] = 1
        y_mask[:, 720::, :] = 1
        y_mask[:, :, 0:60] = 1
        y_mask[:, :, 1000::] =1
    elif '6382-L-32' in fi:
        print('32')
        y_mask[:, 0:400, :] = 1
        y_mask[:, 700::, :] = 1
    
    cayla[y_mask>0] = 0
    cilcare[y_mask>0] = 0
    gt_xyz = regionprops_table(cayla, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T

    if save_csvs:
        print('calculating stats...')
        dice = np.sum((cilcare>0) & (cayla>0)) * 2.0 / (np.sum(cilcare>0) + np.sum(cayla>0))
        results= {"file": fi,
                    "dice": dice,
                    }
        results.update(
            calc_errors(
                cilcare, 
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
