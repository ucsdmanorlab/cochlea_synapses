import glob
import numpy as np
import zarr
import sys
from natsort import natsorted

if len(sys.argv)>1:
    snapdir = sys.argv[1]
else:
    snapdir = 'snapshots/*/'

files = sorted(glob.glob(snapdir+'*.zarr'))

for f in natsorted(files):
    labels = zarr.open(f)['labels'][:]
    a, b = np.unique(labels, return_counts=True)
    if len(b)>1:
        print(f, len(b)-1)

