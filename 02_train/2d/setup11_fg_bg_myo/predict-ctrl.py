import glob
import gunpowder as gp
import numpy as np
import math
import os
import torch
import zarr
from funlib.learn.torch.models import UNet, ConvPass
from skimage.measure import label
from skimage.filters import threshold_otsu
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from torch.nn.functional import binary_cross_entropy
from natsort import natsorted 

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

class AddChannel(gp.BatchFilter):
    def __init__(
            self,
            in_array,
            add_array,
            ch_axis=0,
            ):

        self.in_array = in_array
        self.add_array = add_array
        self.ch_axis = ch_axis

    #def prepare(self, request):

    def process(self, batch, request):

        batch[self.in_array].data = np.stack((batch[self.in_array].data, batch[self.add_array].data), axis = self.ch_axis)


def predict(
    checkpoint,
    raw_file,
    raw_dataset, ): #myo_dataset):

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')
#    myo = gp.ArrayKey('MYO')

    voxel_size = gp.Coordinate((1,)*2)
    input_size = gp.Coordinate((140,)*2) * voxel_size
    output_size = gp.Coordinate((100,)*2) * voxel_size

    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)
 #   scan_request.add(myo, input_size)

    num_fmaps=30

    ds_fact = [(2,2), (2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3),(3,3)]]*num_levels
    ksu = [[(3,3),(3,3)]]*(num_levels-1)

    unet = UNet(
        in_channels=2,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)
    
    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1]*2], activation=None), 
            torch.nn.Sigmoid())

    torch.cuda.set_device(1) 
    
    source = gp.ZarrSource(
        raw_file,
        datasets={
            raw: raw_dataset,
  #          myo: myo_dataset,
        },
        array_specs={
            raw: gp.ArraySpec(interpolatable=True),
  #          myo: gp.ArraySpec(interpolatable=True),
            })

    with gp.build(source):
        total_output_roi = source.spec[raw].roi
        total_input_roi = total_output_roi.grow(context, context)

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred
        })

    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)
    pipeline += gp.Squeeze([raw,])# myo])
    #pipeline += gp.Normalize(myo)
    pipeline += gp.Pad(raw, context)
    #pipeline += gp.Pad(myo, context)
    pipeline += AddChannel(raw, raw) #myo)

    # h, w
    
    pipeline += gp.Stack(1)
    # b, h, w
    pipeline += predict
    pipeline += scan

    pipeline += gp.Squeeze([raw, pred])
    pipeline += gp.Squeeze([ pred])
    # h, w
    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi
    
    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    #predict_request.add(myo, total_input_roi.get_end())
    #predict_request[myo].roi = total_input_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred].data


if __name__ == '__main__':

    model = 'model_2024-04-05_15-37_2class_myo_ctrl_checkpoint_10000'
    
    #checkpoints = natsorted('./'+model+'_checkpoint_*')
    raw_files = glob.glob('../../../01_data/zarrs/validate/spinning*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('01_data/zarrs/validate', '03_predict/2d/'+model))

        full_raw = zarr.open(raw_file)['3d/raw']
        save_out(out_file, full_raw[:], 'raw')

        #for checkpoint in checkpoints:
        checkpoint = model
        pred = []

        for i in range(full_raw.shape[0]):

            print(f'predicting on section {i}')

            raw_dataset = f'2d/raw/{i}'
            #myo_dataset = f'2d/myo/{i}'
            
            pred_mask = predict(
                checkpoint,
                raw_file,
                raw_dataset,)# myo_dataset)

            pred.append(pred_mask)

        pred = np.array(pred)
        
        #checkpoint_num = checkpoint[checkpoint.rfind('_')::]
        save_out(out_file, pred, 'pred')#+checkpoint_num)

        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')

