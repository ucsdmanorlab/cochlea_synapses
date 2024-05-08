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

input_shape = [44, 172, 172]

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
    checkpoint,
    raw_file,
    raw_dataset):

    # raw_orig = gp.ArrayKey("RAW_ORIG") 
    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')
    
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

    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=True)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 3, [[1,]*3], activation='Sigmoid'),
            )
    
    output_shape = model.forward(torch.empty(size=[1,1]+input_shape))[0].shape[1:]
    input_size = gp.Coordinate(input_shape) * target_vox
    output_size = gp.Coordinate(output_shape) * target_vox
    
    context = (input_size - output_size) / 2
    
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)
    
    # if "airyscan" in raw_file:
    #     vox= (16, 4, 4)
    # elif "spinning" in raw_file:
    #     vox = (24, 9, 9)
    # else: #"confocal"
    #     vox = (36, 16, 16)
    # print(vox)
    def create_source(sample):
        # if "airyscan" in sample:
        #     vox = (16,4,4)
        # elif "spinning" in sample:
        #     vox = (24, 9, 9)
        # else: #"confocal"
        #     vox = (36,16,16)

        vox = gp.Coordinate(target_vox)

        source = gp.ZarrSource(
                sample,
                datasets={
                    raw: raw_dataset #f'3d/raw',
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=vox),
                })
        source += gp.Normalize(raw) #_orig)
        #source += gp.Resample(raw_orig, target_vox, raw, ndim=3)

        source +=  gp.Pad(raw, context)
        return source 
    
    source = create_source(raw_file)
    
    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)
        print(total_input_roi, total_output_roi)
    
    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
            'input': raw
        },
        outputs = {
            0: pred,
        })

    scan = gp.Scan(scan_request)

    pipeline = source 
    # c,d,h,w

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)
    # b,c,d,h,w

    pipeline += predict
    pipeline += scan
    
    pipeline += gp.Squeeze([raw, pred])
    # c,d,h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request[raw].roi = total_input_roi

    predict_request.add(pred, total_output_roi.get_end())
    predict_request[pred].roi = total_output_roi

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[pred].data


if __name__ == '__main__':

    checkpoint = 'model_2024-05-06_14-05-07_3d_affs_IntWgt_b1_scaled_mid_2e-5_rej1k_checkpoint_10000' #model_2024-05-01_17-24-41_3d_affs_IntWgt_b1_scaled_NONE_2e-5_rej1k_checkpoint_10000' #0odel_2024-04-30_17-21-45_3d_affs_IntWgt_b1_scaled_up_2e-5_rej1k_checkpoint_10000'
    target_vox=(24, 9, 9)
    raw_files = glob.glob('../../../01_data/zarrs/noGT/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)

        raw_dataset = f'3d/raw' 
        if "airyscan" in raw_file:
            vox = (16,4,4)
        elif "spinning" in raw_file:
            vox = (24, 9, 9)
        else:
            vox = (36,16,16)
        vox = (11, 4, 4) #target_vox        
        scale_fact = np.divide(vox, target_vox)

        in_shape = zarr.open(raw_file)['3d/raw'].shape
        out_shape = np.multiply(in_shape, scale_fact)
        in_type = zarr.open(raw_file)['3d/raw'].dtype

        out_path = raw_file
        
        save_out(
                zarr.open(out_path), 
                normalize(
                    resize(zarr.open(raw_file)['3d/raw'][:], out_shape, preserve_range=True).astype(np.uint16),
                    maxval=(2**16-1)).astype(np.uint16),
                'raw_resize')
        pred = predict(
                checkpoint,
                out_path, #raw_file,
                'raw_resize') #f'3d/raw') 
        
        pred = np.array(pred)
        
        pred_reshape = resize(pred, 
                np.concatenate(([3],in_shape)), 
                preserve_range=True) 
        
        #shutil.rmtree(os.path.join(out_path,'raw_resize'))

        out_file = zarr.open(out_path)
        save_out(out_file, pred, 'pred_resize')
        save_out(out_file, zarr.open(raw_file)['3d/raw'][:], 'raw')
        save_out(out_file, pred_reshape, 'pred')
        #save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')

