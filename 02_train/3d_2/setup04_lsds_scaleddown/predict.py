import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import shutil
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.transform import resize

target_vox = (24,9,9)
input_shape = [44, 172, 172]
voxel_size = gp.Coordinate(target_vox)

class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out,
            constant_upsample,
            dims,
            num_heads,
            ):

        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            num_fmaps_out=num_fmaps_out,
            constant_upsample=constant_upsample,
            num_heads = num_heads, 
            )

        self.lsd_head = ConvPass(num_fmaps_out, dims, [[1]*3], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps_out, 3, [[1]*3], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        lsds = self.lsd_head(z[0])
        affs = self.aff_head(z[0])

        return lsds, affs

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
    checkpoint,
    raw_file,
    raw_dataset):
    
    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    
    in_channels = 1
    num_fmaps=12
    num_fmaps_out = 14
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    model = MtlsdModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            num_fmaps_out,
            constant_upsample=True,
            dims=4,
            num_heads=2,)

    output_shape = model.forward(torch.empty(size=[1,1] + input_shape))[0].shape[2:]
    input_size = gp.Coordinate(input_shape) * target_vox
    output_size = gp.Coordinate(output_shape) * target_vox
    
    context = (input_size - output_size) / 2
    
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)
    scan_request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
        raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)
            })
    source += gp.Normalize(raw)
    #source += gp.Resample(raw_orig, target_vox, raw, ndim=3)

    source += gp.Pad(raw,context)
    source += gp.Unsqueeze([raw])

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred_lsds,
            1: pred, # affs
        })

    scan = gp.Scan(scan_request)

    pipeline = source 
    # c,d,h,w

    pipeline += gp.Stack(1)
    # b,c,d,h,w

    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw, pred])
    # c,d,h,w

    pipeline += gp.Squeeze(
        [raw,])
    # d,h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi

    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred].data


if __name__ == '__main__':

    checkpoint = 'model_2023-05-04_13-01-15_3d_scaled_mid_r0p9_efficiency_checkpoint_10000'
    raw_files = glob.glob('../../../01_data/zarrs/validate/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)

        raw_dataset = f'3d/raw' 
        if "airyscan" in raw_file:
            vox = (16,4,4)
        elif "spinning" in raw_file:
            vox = (24, 9, 9)
        else:
            vox = (36,16,16)
        
        scale_fact = np.divide(vox, target_vox)

        in_shape = zarr.open(raw_file)['3d/raw'].shape
        out_shape = np.multiply(in_shape, scale_fact)
        in_type = zarr.open(raw_file)['3d/raw'].dtype

        out_path = raw_file.replace('validate', checkpoint)
        
        save_out(
                zarr.open(out_path), 
                resize(zarr.open(raw_file)['3d/raw'][:], out_shape, preserve_range=True).astype(in_type),
                'raw_resize')

        pred = predict(
                checkpoint,
                out_path,
                'raw_resize') 
        pred = np.array(pred)

        pred_reshape = resize(pred, 
                np.concatenate(([3],in_shape)), 
                preserve_range=True) 
        
        shutil.rmtree(os.path.join(out_path,'raw_resize'))

        out_file = zarr.open(out_path)
        save_out(out_file, zarr.open(raw_file)['3d/raw'][:], 'raw')
        save_out(out_file, pred_reshape, 'pred')
        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')

