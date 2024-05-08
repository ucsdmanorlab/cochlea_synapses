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

#### Parameters to set: ####
rawdir = 'tifs/' #spinningdisk/raw/'
############################

outdir = rawdir.replace('tifs','zarrs')
outdir = outdir.replace('raw','')

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data.max()
    print(scale_fact)
    data_norm = data * scale_fact
    return data_norm

count = 0 
raw_files = sorted(glob.glob(rawdir + '*.tif'))

for rawfile in raw_files:
    print(count, rawfile) #, fi_groups[count], splits[fi_split[count]])
    raw = imread(rawfile) #[:,:,:,2] for spinning disk data
    raw = normalize(raw.astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)

    fileroot = rawfile[len(rawdir):-len('_raw.tif')]

    # labeled = np.array(imread(rawfile.replace('raw','labels'))).astype(np.uint64)

    # convex = np.zeros_like(labeled)
    # for i in range(labeled.shape[0]):
    #     convex[i] = convex_hull_image((labeled[i])>0).astype(np.uint8)

    outpath = os.path.join(outdir, 'noGT', fileroot+'.zarr')

    out = zarr.open(outpath, 'a')

    for ds_name, data in [
            ('raw', raw),
            #('mask', convex),
            #('labeled', labeled)
            ]:

        # write 3d data
        out[f'3d/{ds_name}'] = data
        out[f'3d/{ds_name}'].attrs['offset'] = [0,]*3
        out[f'3d/{ds_name}'].attrs['resolution'] = [1,]*3
        
        # write 2d data
        #for z in range(data.shape[0]):
        #    out[f'2d/{ds_name}/{z}'] = np.expand_dims(data[z], axis=0)
        #    out[f'2d/{ds_name}/{z}'].attrs['offset'] = [0,]*2
        #    out[f'2d/{ds_name}/{z}'].attrs['resolution'] = [1,]*2

    count += 1
