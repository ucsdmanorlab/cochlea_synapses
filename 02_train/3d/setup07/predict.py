import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import torch
import zarr
from skimage.transform import resize
from funlib.learn.torch.models import UNet, ConvPass


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
            dims):

        super().__init__()

        self.unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        num_fmaps_out=num_fmaps_out,
        constant_upsample=constant_upsample)

        self.lsd_head = ConvPass(num_fmaps_out, dims, [[1]*3], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps_out, 3, [[1]*3], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)

        return lsds, affs

def save_out(out_file,
        array, 
        key,
        res=[1,1,1],
        offset=[0,0,0]):

    out_file[key] = array
    out_file[key].attrs['offset'] = offset
    out_file[key].attrs['resolution'] = res

def predict(
    checkpoint,
    raw_file,
    raw_dataset):

    raw = gp.ArrayKey('RAW')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')

    input_shape = gp.Coordinate((36,172,172))
    output_shape = gp.Coordinate((16,80,80))

    voxel_size = gp.Coordinate((4,1,1))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    context = (input_shape - output_shape) / 2
    context = context * voxel_size

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_affs, output_size)
    scan_request.add(pred_lsds, output_size)

    in_channels = 1
    num_fmaps = 12
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
            dims=4)

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
    
    with gp.build(source):
        total_output_roi = source.spec[raw].roi
        total_input_roi = total_output_roi.grow(context,context)

    model.eval()
    
    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred_lsds,
            1: pred_affs
        })

    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Pad(raw, context)
    
    pipeline += gp.Unsqueeze([raw])
    # d,h,w
    pipeline += gp.IntensityScaleShift(raw, 2,-1)
    pipeline += gp.Stack(1)
    # b,d,h,w
     
    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw])
    pipeline += gp.Squeeze([raw, pred_lsds, pred_affs])
    # d,h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end()) #total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi #total_input_roi
    predict_request.add(pred_affs, total_output_roi.get_end())
    predict_request[pred_affs].roi = total_output_roi
    predict_request.add(pred_lsds, total_output_roi.get_end())
    predict_request[pred_lsds].roi = total_output_roi
    
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred_affs].data, batch[pred_lsds].data


if __name__ == '__main__':

    checkpoint = 'model_checkpoint_400000'
    raw_files = sorted(glob.glob('../../../01_data/zarrs/elena-sd/ctbp2/*.zarr'))

    for raw_file in raw_files:
        print(raw_file)
        out_file_path = raw_file.replace('validation', 'out')
        out_file = zarr.open(out_file_path)

        raw_dataset = f'3d/raw' 
        
        pred_affs, pred_lsds = predict(
                checkpoint,
                raw_file,
                raw_dataset,
                )

        save_out(out_file, np.asarray(zarr.open(raw_file+'/3d/raw')), 'raw')
        save_out(out_file, np.asarray(pred_affs), 'pred_affs_400')
        save_out(out_file, np.asarray(pred_lsds), 'pred_lsds_400')


