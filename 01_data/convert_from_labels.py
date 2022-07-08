import glob
import numpy as np
import zarr
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import convex_hull_image
import os.path
import shutil
import csv

# 08 Feb 2022:
# Script looks for segmentation data & if it exists automatically splits into training & validation
# Raw data is also now converted to 8-bit.

def normalize(data, maxval=1.):
    return (data - np.min(data)) * maxval / (np.max(data) - np.min(data))

#### Parameters to set: ####
rawdir = 'tifs/training/raw/'
############################

outdir = rawdir.replace('tifs','zarrs')
outdir = outdir.replace('raw','')
count = 0 

for rawfile in sorted(glob.glob(rawdir + '*.tif')):
    print(count, rawfile)
    raw = imread(rawfile)
    raw = normalize(raw, 255.).astype(np.uint8)
    
    fileroot = rawfile[len(rawdir):-len('_raw.tif')]

    scalefactor = (1, 1, 1)

    labeled = np.array(imread(rawfile.replace('raw','labels'))).astype(np.uint64)

    convex = np.zeros_like(labeled)
    for i in range(labeled.shape[0]):
        convex[i] = convex_hull_image((labeled[i])>0).astype(np.uint8)


    out = zarr.open(outdir+fileroot+'.zarr', 'a')

    for ds_name, data in [
            ('raw', raw),
            ('mask', convex),
            ('labeled', labeled)]:

        # write 3d data
        out[f'{ds_name}'] = data
        out[f'{ds_name}'].attrs['offset'] = [0,]*3
        out[f'{ds_name}'].attrs['resolution'] = [1,]*3

    out[f'raw'].attrs['scalefactor'] = scalefactor
    count += 1
