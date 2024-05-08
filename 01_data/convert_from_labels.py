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
rawdir = 'tifs/overfit/' #spinningdisk/raw/'
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

def splitfiles(
        imfis, 
        grouplist = None, 
        val_frac = 0.15,
        test_frac = 0.1, 
        ):
    
    if grouplist == None:
        grouplist = [1,]*len(imfis)
    
    groups, counts = np.unique(grouplist, return_counts=True)
    
    fi_split = [0,]*len(imfis)

    for (gr, c) in zip(groups, counts):
        
        if test_frac > 0:
            n_test = int(np.max([1, np.round(test_frac * c)]))
        else:
            n_test = 0 
        if val_frac > 0 and (c-n_test) > 0:
            n_val = int(np.max([1, np.round(val_frac * c)]))
        else:
            n_val = 0

        if (c-n_val-n_test) < 1:
            warnings.warn('All data going to test/val!')
        
        im_ids = np.where(np.array(grouplist) == gr)[0]
        testval = np.random.choice(im_ids, n_val + n_test, replace=False)
        val = testval[0:n_val]
        test = testval[n_val:(n_val + n_test)]
        
        for x in val:
            fi_split[x] = 1
        for x in test:
            fi_split[x] = 2

    return fi_split

count = 0 
raw_files = sorted(glob.glob(rawdir + '*raw.tif'))
fi_groups = [i[i.rfind('/')+1::] for i in raw_files]
fi_groups = [i[:i.find('_')] for i in fi_groups]

fi_split = splitfiles(raw_files)#, grouplist = fi_groups)

splits = ['train', 'validate', 'test']
for i in np.unique(fi_split):
    outpath = os.path.join(outdir, splits[i])
    if not os.path.exists(outpath):
        os.mkdir(outpath)

for rawfile in raw_files:
    print(count, rawfile, fi_groups[count], splits[fi_split[count]])
    raw = imread(rawfile) #[:,:,:,2] for spinning disk data
    raw = normalize(raw.astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)

    fileroot = rawfile[len(rawdir):-len('_raw.tif')]

    labeled = np.array(imread(rawfile.replace('raw','labels'))).astype(np.uint64)

    convex = np.zeros_like(labeled)
    for i in range(labeled.shape[0]):
        convex[i] = convex_hull_image((labeled[i])>0).astype(np.uint8)

    outpath = os.path.join(outdir, splits[fi_split[count]], fileroot+'.zarr')

    out = zarr.open(outpath, 'a')

    for ds_name, data in [
            ('raw', raw),
            ('mask', convex),
            ('labeled', labeled)]:

        # write 3d data
        out[f'3d/{ds_name}'] = data
        out[f'3d/{ds_name}'].attrs['offset'] = [0,]*3
        out[f'3d/{ds_name}'].attrs['resolution'] = [1,]*3
        
        # write 2d data
        for z in range(data.shape[0]):
            out[f'2d/{ds_name}/{z}'] = np.expand_dims(data[z], axis=0)
            out[f'2d/{ds_name}/{z}'].attrs['offset'] = [0,]*2
            out[f'2d/{ds_name}/{z}'].attrs['resolution'] = [1,]*2

    count += 1
