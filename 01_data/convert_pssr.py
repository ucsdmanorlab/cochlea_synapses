import time
import glob
import numpy as np
import zarr
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import convex_hull_image
import os.path
import shutil
import warnings
import csv
import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import split_files, normalize

start_t = time.time()
#### Parameters to set: ####
#rawdir = '/home/caylamiller/Desktop/cilcare5/' #spinningdisk/raw/'
rawdir = '../../../Downloads/mice_preds/'
############################
 
outdir = './zarrs/pssr/' #rawdir.replace('tifs','zarrs')

count = 0 
raw_files = sorted(glob.glob(rawdir + '*.tif'))

for rawfile in raw_files:
    print(count, 'out of ', len(raw_files), rawfile)
    
    raw = imread(rawfile)[:,:,:] 
    print(raw.shape)
    raw = normalize(raw.astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)

    fileroot = rawfile[len(rawdir):-4].replace('Myosin7a405_CtBP2568_GluR2488_NF647_', '') 
    print(fileroot)
    outpath = os.path.join(outdir, fileroot+'.zarr')

    out = zarr.open(outpath, 'a')

    for ds_name, data in [
            ('raw', raw),
            #('convex', convex),
            #('labeled', labeled)
            ]:

        # write 3d data
        out[f'3d/{ds_name}'] = data
        out[f'3d/{ds_name}'].attrs['offset'] = [0,]*3
        out[f'3d/{ds_name}'].attrs['resolution'] = [1,]*3
        
        # write 2d data
        # for z in range(data.shape[0]):
        #     out[f'2d/{ds_name}/{z}'] = np.expand_dims(data[z], axis=0)
        #     out[f'2d/{ds_name}/{z}'].attrs['offset'] = [0,]*2
        #     out[f'2d/{ds_name}/{z}'].attrs['resolution'] = [1,]*2

    count += 1

print("Elapsed time: "+str(time.time()-start_t)+" s")
