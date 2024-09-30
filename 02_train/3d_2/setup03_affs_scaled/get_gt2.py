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

def predict(raw_file):

    gt = gp.ArrayKey('GT')
    #labels = gp.ArrayKey('LABELS')
    labels_orig = gp.ArrayKey('LABELS_ORIG')
    
    source = gp.ZarrSource(
            raw_file,
            datasets={
                labels_orig: f'3d/labeled',
                },
            array_specs={
                labels_orig:gp.ArraySpec(interpolatable=False, voxel_size=(16,4,4)),#vox),
                })
            
    with gp.build(source):
        total_input_roi = source.spec[labels_orig].roi

    pipeline = source
    pipeline += gp.AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels_orig,
            affinities=gt,
            dtype=np.float32)
    
    request = gp.BatchRequest()
    request[labels_orig] = total_input_roi
    request[gt] = total_input_roi
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    return batch[gt].data

if __name__ == '__main__':

    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
    
    raw_files = glob.glob('../../../01_data/zarrs/validate/airy*.zarr')
    checkpoint = 'model_2024-04-30_17-13-39_3d_affs_IntWgt_b1_scaled_down_2e-5_rej1k_checkpoint_10000'    
    
    for raw_file in raw_files:
        print(raw_file)

        in_shape = zarr.open(raw_file)['3d/raw'].shape
        in_type = zarr.open(raw_file)['3d/raw'].dtype

        out_path = raw_file.replace('01_data/zarras/validate', '03_predict/3d/'+checkpoint)
        
        gt = predict(raw_file) #seg_to_affgraph(
             #   zarr.open(raw_file)['3d/labeled'].astype(np.int32),
             #       neighborhood)
        
        gt = np.array(gt)
        
        out_file = zarr.open(out_path)
        save_out(out_file, gt, 'gt')
