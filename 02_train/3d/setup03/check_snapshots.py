import glob
import numpy as np
import zarr

files = sorted(glob.glob('snapshots/*.zarr'))

for f in files:
    labels = zarr.open(f)['labels'][:]
    a, b = np.unique(labels, return_counts=True)
    print(f, len(b))

