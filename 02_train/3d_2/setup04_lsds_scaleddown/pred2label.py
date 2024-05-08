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

val_dir = '../../../01_data/zarrs/model_2023-05-04_13-01-24_3d_scaled_mid_r0p9_checkpoint_10000'
#'../../../01_data/zarrs/model_2022-09-28_17-03-31_affs_checkpoint_20000' #'../../../01_data/zarrs/checkpoint'
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
