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

# todo: add resolution [3462,1139,1139]

def normalize(data, maxval=1.):
    return (data - np.min(data)) * maxval / (np.max(data) - np.min(data))

#### Parameters to set: ####
segmentsuf  = '_seg'
trainfrac = 0.7
rawdir = 'tifs/elena-2examples/'
############################

outdir = rawdir.replace('tifs','zarrs')
outdir = outdir.replace('ctbp2/', 'ctbp2_restest/')
filelog = []
count = 0 

for rawfile in sorted(glob.glob(rawdir + '*.tif')):
    print(count, rawfile)
    filelog.append([str(count), rawfile, 'validation'])
    raw = imread(rawfile)
    raw = normalize(raw, 255.).astype(np.uint8)
    
    fileroot = rawfile[len(rawdir):-len('.tif')]

    resfile = os.path.join(rawdir +'resolution/', fileroot + "_resolution.txt")

    if os.path.exists(resfile):
        f = open(resfile, "r")
        resolution = f.readline() # skip first line
        resolution = f.readline()
        resolution = np.asarray(resolution.split()[0:3]).astype(float)
        scalefactor = np.divide(np.asarray(resolution), [0.2, 0.05, 0.05]).tolist()
        f.close()
        print(resolution)
        print(scalefactor)
    else:
        scalefactor = (1, 1, 1)
        print('Warning: resolution file missing: '+resfile+'\n Using arbitrary resolution')

    segdir = rawdir.replace('raw','seg') + fileroot + segmentsuf
    
    if os.path.isdir(segdir):
        seg = np.array([imread(i) for i in sorted(glob.glob(segdir+'/*.tif'))])

        convex = np.zeros_like(seg)
        for i in range(seg.shape[0]):
            convex[i] = convex_hull_image(seg[i]).astype(np.uint8)

        labeled = label(seg).astype(np.uint64)
        seg = normalize(seg).astype(np.float32)

        out = zarr.open(outdir+'temp/'+'sample_'+str(count)+'.zarr', 'a')

        for ds_name, data in [
                ('seg', seg),
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

    else:
        out = zarr.open(outdir+'sample_'+str(count)+'.zarr', 'a')
        
    # write 3d data
    out[f'3d/raw'] = raw
    out[f'3d/raw'].attrs['offset'] = [0,]*3
    out[f'3d/raw'].attrs['resolution'] = [1,]*3
    out[f'3d/raw'].attrs['scalefactor'] = scalefactor

    # write 2d data
    for z in range(raw.shape[0]):
        out[f'2d/raw/{z}'] = np.expand_dims(raw[z], axis=0)
        out[f'2d/raw/{z}'].attrs['offset'] = [0,]*2
        out[f'2d/raw/{z}'].attrs['resolution'] = [1,]*2
        out[f'2d/raw/{z}'].attrs['scalefactor'] = scalefactor[1:2]

    count += 1

# split into training and validation:
if os.path.isdir(outdir+'temp/'):
    dirlist = glob.glob(outdir+'temp/*.zarr')
    
    trainN = np.around(trainfrac * len(dirlist)).astype(int)
    trainSamples = np.random.choice(len(dirlist), trainN, replace=False)
        
    for i in trainSamples:
        rootname = dirlist[i].replace(outdir + 'temp/','')
        imnum = int(rootname.replace('sample_','').replace('.zarr', ''))
        filelog[imnum][2] = 'training'
        shutil.move(dirlist[i], outdir + 'training/' + rootname)
    
    for zarrleft in glob.glob(outdir+'temp/*.zarr'):
        rootname = zarrleft.replace(outdir + 'temp/','')
        shutil.move(zarrleft, outdir + 'validation/' + rootname)

    os.rmdir(outdir+'temp/')
print(outdir)        
with open(outdir + "filelog.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(filelog)
