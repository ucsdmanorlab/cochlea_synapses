import glob
import numpy as np
import zarr
from tqdm import tqdm
import os.path
import shutil
import warnings
import csv

#### Parameters to set: ####
zarr2ddir = 'zarrs/validate/' #spinningdisk/raw/'
############################

outdir = zarr2ddir
    
count = 0 
zarrfiles = sorted(glob.glob(zarr2ddir + '*.zarr'))

for zarrfile in zarrfiles:
    print(count, zarrfile)
    
    zarrfile = zarr.open(zarrfile)
    raw = zarrfile['3d/raw']
    slices = raw.shape[0]
    raw = np.pad(raw, ((2,2), (0,0), (0,0)))
    labeled = zarrfile['3d/labeled']
    xy = raw.shape[1::]

    for z in tqdm(range(slices)):
            zarrfile[f'2p5d/raw/{z}'] = raw[z:z+5, :, :] #.reshape([5,  xy[0], xy[1]])
            zarrfile[f'2p5d/labeled/{z}'] = labeled[z, :, :] #.reshape([1, xy[0], xy[1]])

            for ds_name in ['raw', 'labeled']:
                zarrfile[f'2p5d/{ds_name}/{z}'].attrs['offset'] = [0,]*2
                zarrfile[f'2p5d/{ds_name}/{z}'].attrs['resolution'] = [1,]*2
    
    count += 1
