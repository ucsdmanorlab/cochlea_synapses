import glob
import numpy as np
import zarr
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import convex_hull_image

# todo: add resolution [3462,1139,1139]
# todo: wrap in loop for each sample

def normalize(data):

    return (data - np.min(data)) / (np.max(data) - np.min(data))

raw = imread('tifs/raw/*.tif')
seg = np.array([imread(i) for i in sorted(glob.glob('tifs/seg/*.tif'))])

convex = np.zeros_like(seg)

for i in range(seg.shape[0]):
    convex[i] = convex_hull_image(seg[i]).astype(np.uint8)

labeled = label(seg).astype(np.uint64)

seg = normalize(seg).astype(np.float32)

out = zarr.open('zarrs/sample_0.zarr', 'a')

for ds_name, data in [
        ('raw', raw),
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

