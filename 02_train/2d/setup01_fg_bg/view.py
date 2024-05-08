import neuroglancer
import numpy as np
import sys
import webbrowser
import zarr

neuroglancer.set_server_bind_address('10.3.12.57')

f = zarr.open(sys.argv[1])

raw = f['raw'][:]
labels = f['labels'][:]
pred_mask = f['pred_mask'][:]

offset = f['labels'].attrs['offset']
res = f['labels'].attrs['resolution']

offset = [0,]*2 + [int(i/j) for i,j in zip(offset, res)]

viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['c','z^','y','x'],
        units='nm',
        scales=res+res)

with viewer.txn() as s:

    s.layers['raw'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=raw,
                voxel_offset=[0,]*4,
                dimensions=dims))

    s.layers['labels'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=labels,
                voxel_offset=offset,
                dimensions=dims))

    s.layers['pred_mask'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=pred_mask,
                voxel_offset=offset,
                dimensions=dims))

    s.layout = 'yz'

url = str(viewer)
print(url)

try:
    webbrowser.open_new_tab(url)
except:
    webbrowser.open_new(url)
