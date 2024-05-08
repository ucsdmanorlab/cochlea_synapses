import neuroglancer
import numpy as np
import sys
import os
import webbrowser
import zarr

neuroglancer.set_server_bind_address('132.239.70.80')

viewer = neuroglancer.Viewer()

container = zarr.open(sys.argv[1])
datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]

#ds = zarr.open(sys.argv[1])['pred_mask']

with viewer.txn() as s:

    for dataset in datasets:
        data = container[dataset][:]
        if data.dtype == 'int64':
            print(dataset)
            data = data.astype(np.uint64)

        offset = container[dataset].attrs['offset']
        shape = len(data.shape)
        while len(offset) < shape:
            offset.insert(0, 0) 

        if shape==5:
            names = ['b^', 'c^', 'z', 'y', 'x']
        elif shape==4:
            names = ['c^', 'z', 'y', 'x']
        else: 
            names = ['z', 'y', 'x']
        
        dims = neuroglancer.CoordinateSpace(
                names = names, 
                units = 'nm', 
                scales = [1,]*shape)

        vol = neuroglancer.LocalVolume(
                data = data,
                voxel_offset=offset, 
                dimensions=dims)

        if shape==4:
            s.layers[dataset] = neuroglancer.ImageLayer(
                    source=vol,
                    shader="""
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}
""")
        elif dataset == 'labels':
            s.layers[dataset] = neuroglancer.SegmentationLayer(
                    source=vol)
        else:
            s.layers[dataset] = neuroglancer.ImageLayer(
                    source=vol
                    )

url = str(viewer)
print(url)

try:
    webbrowser.open_new_tab(url)
except:
    webbrowser.open_new(url)

input("Press ENTER to quit\n")
#print(viewer)
