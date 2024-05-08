import argparser
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

parser = argparse.ArgumentParser()

parser.add_argument("-z", "--zarrdir", dest="zarrdir", default='../../../01_data/zarrs/validate/', help='Input directory')
parser.add_argument("checkpoint")
parser.add_argument("-csv", "--save_csvs", dest="save_csvs", default=True)
parser.add_argument("-l", "--save_labels", dest="save_pred_labels", default=False)

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

    voxel_size = gp.Coordinate((4,1,1,))

    input_size = gp.Coordinate((44,172,172)) * voxel_size
    output_size = gp.Coordinate((24,80,80)) * voxel_size

    context = (input_size - output_size) / 2

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    num_fmaps=12
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
        in_channels=1, #in_channels,
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
            0: pred
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

    checkpoint = 'model_2023-03-30_17-38-07_3d_affs_IntWgt_b5_checkpoint_10000'
#    raw_file = '../../../01_data/zarrs/sample_0.zarr'
    raw_files = glob.glob('../../../01_data/zarrs/validate/*.zarr')
    
    for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('01_data/zarrs/validate', '03_predict/'+checkpoint))

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
        
        save_out(out_file, zarr.open(raw_file)['3d/raw'][:], 'raw')
        save_out(out_file, pred, 'pred')
        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')

#!python
#cython: language_level=3

import numpy as np
import shutil
import pandas as pd
import waterz
import zarr
import os

from scipy.ndimage import label, gaussian_filter, maximum_filter, distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import regionprops_table

val_dir = '../../../03_predict/model_2023-03-30_17-38-07_3d_affs_IntWgt_b5_checkpoint_10000'
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]

aff_thresh_list = [0.4, 0.5, 0.7, 0.9]
merge_thresh_list = [0.1, 0.5, 0.7]
save_pred_labels = False
save_csvs = True

AllResults = pd.DataFrame()
count = 0

def calc_errors(labels, gt_xyz, mode='2d'):

    gtp = gt_xyz.shape[0]
    Npoints_pred = len(np.unique(labels))-1

    pred_idcs = np.zeros(gtp).astype(np.int32)

    for i in range(gtp):

        cent = np.around(gt_xyz[i,:]).astype(np.int16)
        if mode=='2d':
            pred_idcs[i] = labels[cent[1], cent[0]]
        elif mode=='3d':
            pred_idcs[i] = labels[cent[2], cent[1], cent[0]]

    idc, counts = np.unique(pred_idcs[pred_idcs>0], return_counts=True)

    tp = np.sum(counts==1)
    fm = np.sum(counts>1)
    fn = np.sum(pred_idcs==0)
    fp = Npoints_pred - len(idc)   
    
    ap = tp/(tp+fn+fp+fm)
    f1 = 2*tp/(2*tp + fp + fn + fm)
    
    prec = tp/(tp+fp)
    rec  = tp/gtp

    results = {
        "gtp": gtp,
        "tp": tp,
        "fn": fn,
        "fm": fm, 
        "fp": fp, 
        "ap": ap,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "mergerate": fm/gtp,
    }
    
    return results 

def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        min_seed_distance=3):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), 0

    fragments = watershed(
        #boundary_distances.max()
        -boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n)

    return ret

def watershed_from_affinities(
        affs,
        thresh=0.55,
        max_affinity_value=1.0,
        min_seed_distance=3,
        labels_mask=None):
    
    mean_affs = np.mean(affs, axis=0)
    depth = mean_affs.shape[0]

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)

    #for z in range(depth):

    boundary_mask = mean_affs>(thresh*max_affinity_value)
    boundary_distances = distance_transform_edt(boundary_mask)

    if labels_mask is not None:
        boundary_mask *= labels_mask.astype(bool)

    ret = watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        min_seed_distance=min_seed_distance)

    fragments = ret[0]

    id_offset = ret[1]

    ret = (fragments, id_offset)

    return ret

def get_segmentation(affinities,
        merge_thresh = 0.5,
        affs_thresh = 0.5,
        labels_mask=None):

    fragments = watershed_from_affinities(
            affinities,
            thresh=affs_thresh,
            labels_mask=labels_mask)[0]
    print('fragments done')

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=[merge_thresh],
    )
    print('generator created')
    segmentation = next(generator)
    print('done')
    
    return segmentation


for fi in val_samples:
    print(fi)
    img = zarr.open(os.path.join(val_dir, fi))
    
    # Get xyz coordinates of GT centroids:
    print('getting centroids...')
    gt = img['gt_labels'][:]
    gt_xyz = pd.DataFrame(regionprops_table(gt, properties=('centroid','area')))
    gt_xyz = gt_xyz[gt_xyz['area']>1]
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T


    pred = img['pred'][:]
    print('pred shape ',pred.shape)
    #pred = gaussian_filter(pred, sigma=(1,0,1,1))
    
    for affs_thresh in aff_thresh_list:
        affs_thresh_scaled = affs_thresh
        print(affs_thresh)

        fragments = watershed_from_affinities(
            pred,
            thresh=affs_thresh_scaled,
            labels_mask=None)[0]

        for merge_thresh in merge_thresh_list:
            
            generator = waterz.agglomerate(
                    affs=pred.astype(np.float32),
                    fragments=fragments,
                    thresholds=[merge_thresh],
                    )

            segmentation = next(generator)

            #segmentation = get_segmentation(pred, merge_thresh=merge_thresh, affs_thresh=affs_thresh_scaled ,)
            if save_pred_labels:
                print('saving labels...')
                if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                    shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

                print('saving')
                img['pred_labels'] = segmentation.astype(np.uint64)
                img['pred_labels'].attrs['offset'] = [0,]*3
                img['pred_labels'].attrs['resolution'] = [1,]*3

            print('calculating stats...')
            results= {"file": fi,
                     "affs_thresh": affs_thresh,
                     "merge_thresh": merge_thresh,}
            results.update(
                calc_errors(
                    segmentation, 
                    gt_xyz, 
                    mode='3d')
            )
            AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
            count = count + 1

if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats.csv'),encoding='utf-8-sig')

    bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')


#for raw_file in raw_files:
#    print(raw_file)
#    out_file = zarr.open(raw_file)
#
#    merge_thresh = 0.5 
#    affs_thresh = 0.5
#    pred = out_file['pred']
#    pred = np.stack([np.zeros_like(pred[:,0,:,:]), 
#        pred[:,0,:,:],
#        pred[:,1,:,:]])
#    print(pred.shape)
#    segmentation = get_segmentation(pred, merge_thresh=merge_thresh, affs_thresh=affs_thresh ,)# labels_mask=mask)
#    print(segmentation.shape)
#    out_file['pred_labels'] = segmentation #labeled.astype(np.uint64)
#    out_file['pred_labels'].attrs['offset'] = out_file['raw'].attrs['offset']
#    out_file['pred_labels'].attrs['resolution'] = out_file['raw'].attrs['resolution']
#
