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
from utils import split_files, normalize, save_out

start_t = time.time()
#### Parameters to set: ####
rawdir = 'tifs/overfit/' #spinningdisk/raw/'
raw_files = sorted(glob.glob(rawdir + '*raw.tif'))

val_frac = 0.15
test_frac = 0.1

outdir = rawdir.replace('tifs','zarrs')
outdir = outdir.replace('raw','')
############################

fi_groups = [i[i.rfind('/')+1::] for i in raw_files]
fi_groups = [i[:i.find('_')] for i in fi_groups]

fi_split = split_files(raw_files,
                       #grouplist = fi_groups,
                       val_frac=val_frac,
                       test_frac=test_frac)

splits = ['train', 'validate', 'test']
for i in np.unique(fi_split):
    outpath = os.path.join(outdir, splits[i])
    if not os.path.exists(outpath):
        os.mkdir(outpath)

count = 0 
last_t = time.time()
for rawfile in raw_files:
    print(count, 'of', len(raw_files) rawfile, fi_groups[count], splits[fi_split[count]])
    raw = imread(rawfile) #[:,:,:,2] for spinning disk data
    raw = normalize(raw.astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)

    fileroot = rawfile[len(rawdir):-len('_raw.tif')]

    labeled = np.array(imread(rawfile.replace('raw','labels'))).astype(np.uint64)

    outpath = os.path.join(outdir, splits[fi_split[count]], fileroot+'.zarr')

    out = zarr.open(outpath, 'a')

    for ds_name, data in [
            ('raw', raw),
            ('labeled', labeled)]:
        
        # write 3d data
        save_out(out, data, ds_name, save2d=False)

    count += 1
    print("Image conversion time: "+str(time.time()-last_t)+" s")
    last_t = time.time()

print("Total elapsed time: "+str(time.time()-start_t)+" s")

