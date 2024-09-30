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
rawdir = '/media/caylamiller/DataDrive/BPHO Dropbox/Manor Lab/ManorLab/File requests/cilcare_data/images/'
############################
 
outdir = './zarrs/cilcare/' #rawdir.replace('tifs','zarrs')
#outdir = outdir.replace('raw','')

count = 0 
raw_files = sorted(glob.glob(rawdir + '*.tif'))
label_files = sorted(glob.glob(rawdir.replace('images', 'labels')+'*.tif'))

# splits = ['train', 'validate', 'test']
# for i in np.unique(fi_split):
#     outpath = os.path.join(outdir, splits[i])
#     if not os.path.exists(outpath):
#         os.mkdir(outpath)
# 
for label_file in label_files: #rawfile in raw_files:
    print(count, 'out of ', len(label_files), label_file)
    rawfile = label_file.replace('labels', 'images')

    #raw = imread(rawfile)[:,:,:,0] #for spinning disk data
    #print(raw.shape)
    #raw = normalize(raw.astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)

    fileroot = rawfile[len(rawdir):-len(' Marion.tif')]

    labeled = np.array(imread(label_file)).astype(np.uint64)

    convex = np.zeros_like(labeled, dtype=np.uint8)
    for i in range(labeled.shape[0]):
        if np.sum(labeled[i])>0:
            convex[i] = convex_hull_image((labeled[i])>0).astype(np.uint8)

    outpath = os.path.join(outdir, fileroot+'.zarr')

    out = zarr.open(outpath, 'a')

    for ds_name, data in [
            #('raw', raw),
            ('convex', convex),
            ('labeled', labeled)]:

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
