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

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data.max()
    print(scale_fact)
    data_norm = data * scale_fact
    return data_norm

def save_out(out_file,
        array,
        key,
        res=None,
        offset=None):

    if res is None:
        res = [1,]*len(array.shape)
    if offset is None:
        offset = [0,]*len(array.shape)

    out_file[key] = array
    out_file[key].attrs['offset'] = offset
    out_file[key].attrs['resolution'] = res

def predict(
    raw_file):

    labels_orig = gp.ArrayKey("LABELS_ORIG")
    labels = gp.ArrayKey("LABELS")
    gt = gp.ArrayKey("GT")
    
    def create_source(sample):
        if "airyscan" in sample:
            vox = (16,4,4)
        elif "spinning" in sample:
            vox = (24, 9, 9)
        else: #"confocal"
            vox = (36,16,16)

        source = gp.ZarrSource(
                sample,
                datasets={
                    labels_orig:f'3d/labeled',
                },
                array_specs={
                    labels_orig:gp.ArraySpec(interpolatable=False, voxel_size=vox),
                })
        source += gp.Resample(labels_orig, target_vox, labels, ndim=3)

        return source 
    
    source = create_source(raw_file)
    
    with gp.build(source):
        total_label_roi = source.spec[labels].roi
    
    pipeline = source 
    # c,d,h,w

    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
    pipeline += gp.AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels,
            affinities=gt,
            dtype=np.float32)
    
    # c,d,h,w

    request = gp.BatchRequest()
    request.add(labels, total_label_roi.get_shape())
    request[labels].roi = total_label_roi
    request.add(gt, total_label_roi.get_shape())
    request[gt].roi = total_label_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    return batch[gt].data


if __name__ == '__main__':

    target_vox=(36,16,16)
    raw_files = glob.glob('../../../01_data/zarrs/validate/airy*.zarr')
    checkpoint = 'model_2024-04-30_17-13-39_3d_affs_IntWgt_b1_scaled_down_2e-5_rej1k_checkpoint_10000'    
    for raw_file in raw_files:
        print(raw_file)

        in_shape = zarr.open(raw_file)['3d/raw'].shape
        in_type = zarr.open(raw_file)['3d/raw'].dtype

        out_path = raw_file.replace('validate', checkpoint)
        
        gt = predict(
                raw_file)
        
        gt = np.array(gt)
        
        out_file = zarr.open(out_path)
        save_out(out_file, gt, 'gt_resize')
