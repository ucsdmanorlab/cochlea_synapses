import glob
import numpy as np
import zarr

files = sorted(glob.glob('snapshots/*.zarr'))

for f in files:
    labels = zarr.open(f)['labels'][:]
    a = np.unique(labels)
    if len(a)>1:
        print(f, len(a)-1)

