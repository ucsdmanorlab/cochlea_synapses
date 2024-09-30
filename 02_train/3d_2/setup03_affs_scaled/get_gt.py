import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import shutil
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.transform import resize
import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import save_out, normalize

#def normalize(data, maxval=1., dtype=np.uint16):
#    data = data.astype(dtype)
#    data_norm = data - data.min()
#    scale_fact = maxval/data.max()
#    print(scale_fact)
#    data_norm = data * scale_fact
#    return data_norm
#
#def save_out(out_file,
#        array,
#        key,
#        res=None,
#        offset=None):
#
#    if res is None:
#        res = [1,]*len(array.shape)
#    if offset is None:
#        offset = [0,]*len(array.shape)
#
#    out_file[key] = array
#    out_file[key].attrs['offset'] = offset
#    out_file[key].attrs['resolution'] = res

def seg_to_affgraph(seg, nhood):
    nhood = np.array(nhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:
        for e in range(nEdge):
            aff[
                e,
                max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
            ] = (
                (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    ]
                    == seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    ]
                )
                * (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    ]
                    > 0
                )
                * (
                    seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    ]
                    > 0
                )
            )

    elif dims == 3:
        for e in range(nEdge):
            aff[
                e,
                max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
            ] = (
                (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                        max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                    ]
                    == seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                        max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                    ]
                )
                * (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                        max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                    ]
                    > 0
                )
                * (
                    seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                        max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                    ]
                    > 0
                )
            )

    else:
        raise RuntimeError(f"AddAffinities works only in 2 or 3 dimensions, not {dims}")

    return aff


if __name__ == '__main__':

    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
    
    raw_files = glob.glob('../../../01_data/zarrs/validate/airy*.zarr')
    checkpoint = 'model_2024-04-30_17-13-39_3d_affs_IntWgt_b1_scaled_down_2e-5_rej1k_checkpoint_10000'    
    
    for raw_file in raw_files:
        print(raw_file)

        in_shape = zarr.open(raw_file)['3d/raw'].shape
        in_type = zarr.open(raw_file)['3d/raw'].dtype

        out_path = raw_file.replace('01_data/zarras/validate', '03_predict/3d/'+checkpoint)
        
        gt = seg_to_affgraph(
                zarr.open(raw_file)['3d/labeled'].astype(np.int32),
                    neighborhood)
        
        gt = np.array(gt)
        
        out_file = zarr.open(out_path)
        save_out(out_file, gt, 'gt')
