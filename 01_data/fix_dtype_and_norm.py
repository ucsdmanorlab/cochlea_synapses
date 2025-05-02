import glob
import time
import numpy as np
import zarr
from skimage.io import imread
import os.path
import shutil
import warnings
import csv
import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import normalize, save_out

check_dir = 'zarrs/validate/'
check_files = sorted(glob.glob(check_dir + '*.zarr'))

for check_file in check_files:
    f = zarr.open(check_file)
    print(check_file)

    raw = f['3d/raw'][:]
    raw = normalize(raw.astype(np.uint16), 
                    maxval=(2**16-1)).astype(np.uint16)
    
    save_out(f, raw, 'raw', save2d=False)

    labels = f['3d/labeled']
    save_out(f, labels.astype(np.int64), 'labeled', save2d=False)