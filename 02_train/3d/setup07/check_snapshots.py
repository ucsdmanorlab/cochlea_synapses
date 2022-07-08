import glob
import numpy as np
import zarr

files = sorted(glob.glob('snapshots/batch_4?????.zarr'))

for f in files:
    labels = zarr.open(f)['labels'][:]
    a, b = np.unique(labels, return_counts=True)
    if len(b)>0:
        print(f, len(b)-1)

