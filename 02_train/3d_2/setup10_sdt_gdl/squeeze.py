import zarr
import numpy
import glob
import os

snap_dir = 'snapshots/2025-03-14_14-41-23_MSE0.05GDL1_rej1_end0.8_step0.01_every1_1k4k/'
snap_files = glob.glob(snap_dir+'*.zarr')

input_size = (44,172,172)

for snap_file in snap_files:
    print(snap_file)
    f = zarr.open(snap_file)
    for key in os.listdir(snap_file):
        if key.startswith('.'):
            continue
        while f[key].shape[0] == 1:
            f[key] = numpy.squeeze(f[key], axis=0)

        zyx_shape = f[key].shape[-3::]
        f[key].attrs['offset'] = [(j-i)/2 for i, j in zip(zyx_shape, input_size)]
        print(key, f[key].attrs['offset'])
        f[key].attrs['resolution'] = [1,1,1]
        