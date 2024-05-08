import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label
from skimage.filters import threshold_otsu

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

def save_out(outfile,
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

    voxel_size = gp.Coordinate((4,1,1,))

    input_size = gp.Coordinate((44,172,172)) * voxel_size
    output_size = gp.Coordinate((24,80,80)) * voxel_size

    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)
    scan_request.add(pred_lsds, output_size)

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
    
    source += gp.Pad(raw,context)
    source += gp.Normalize(raw)
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
    #pipeline += gp.Normalize(raw)
    
    # d,h,w

    pipeline += gp.Stack(1)
    # b,d,h,w

    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw, pred])
    # d,h,w

    pipeline += gp.Squeeze(
        [raw,])
    # ???
    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi

    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred].data


if __name__ == '__main__':

    checkpoint = 'model_2023-03-21_16-17-41_3d_lsds_rej0p8_checkpoint_10000'
#    raw_file = '../../../01_data/zarrs/sample_0.zarr'
    raw_files = glob.glob('../../../01_data/zarrs/validate/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('validate', checkpoint))

        raw_dataset = f'3d/raw' #zarr.open(raw_file)['3d/raw'][:]

        pred = predict(
                checkpoint,
                raw_file,
                raw_dataset) #[]

#        for i in range(full_raw.shape[0]):
#
#            print(f'predicting on section {i}')
#
#            raw_dataset = f'2d/raw/{i}'
#
#            pred_mask = predict(
#                checkpoint,
#                raw_file,
#                raw_dataset)
#
#            pred.append(pred_mask)
#
        pred = np.array(pred)
        
        save_out(out_file, zarr.open(raw_file)['3d/raw'][:].astype(np.uint8), 'raw')
        save_out(out_file, pred, 'pred')
        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')

