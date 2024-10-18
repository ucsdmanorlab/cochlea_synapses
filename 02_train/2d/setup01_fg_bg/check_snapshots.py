import glob
import numpy as np
import zarr

files = sorted(glob.glob('snapshots/batch_*.zarr'))

for f in files:
    labels = zarr.open(f)['labels'][:]
    a, b = np.unique(labels, return_counts=True)
    if len(b)>1:
        print(f, len(b)-1)

