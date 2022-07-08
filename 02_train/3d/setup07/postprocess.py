import glob
import numpy as np
import waterz
import zarr

from scipy.ndimage import label, gaussian_filter
from scipy.ndimage import measurements
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import resize
from skimage.segmentation import watershed

#raw_files = sorted(glob.glob('../../../01_data/zarrs/nih/ctbp2_prediction_affs_lsds/*3.zarr'))
#raw_files = sorted(glob.glob('../../../01_data/zarrs/ctbp2/prediction_affs_lsds/validation/*.zarr'))
raw_files = sorted(glob.glob('../../../01_data/zarrs/elena-sd/ctbp2/*.zarr')) #hertzano/crops/*.zarr'))
#raw_files = sorted(glob.glob('../../../01_data/zarrs/ctbp2/restest/out/*_8.zarr'))

def renumber(array):

    unq_list = np.unique(array.astype(np.uint64))
    all_list = np.arange(unq_list.max())
    all_list = np.flip(all_list[1:])

    last_n = np.max(unq_list)
    for a in all_list:
        if ~np.isin(a, unq_list):
            array[array >= last_n] -= 1
        last_n = a
    return array

def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        #boundary_distances.max() 
        -boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret

def watershed_from_affinities(
        affs,
        thresh=0.55,
        max_affinity_value=1.0,
        return_seeds=False,
        min_seed_distance=10,
        labels_mask=None):

    #mean_affs = 0.333333*( affs[0] + affs[1] + affs[2])
    mean_affs = np.mean(affs, axis=0) 
    mean_affs = np.mean(gaussian_filter(affs, [0,0.5,1,1]), axis=0)
    depth = mean_affs.shape[0]

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
    if return_seeds:
        seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

    id_offset = 0

    #for z in range(depth):

    boundary_mask = mean_affs>(thresh*max_affinity_value)
    boundary_distances = distance_transform_edt(boundary_mask)

    if labels_mask is not None:
        boundary_mask *= labels_mask.astype(bool)
        
    ret = watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        return_seeds=return_seeds,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance)

    fragments = ret[0]
    if return_seeds:
        seeds = ret[2]

    id_offset = ret[1]

    ret = (fragments, id_offset)
    if return_seeds:
        ret += (seeds,)

    return ret

def get_segmentation(affinities, 
        merge_thresh = 0.5, 
        affs_thresh = 0.5, 
        labels_mask=None):

    fragments = watershed_from_affinities(
            affinities,
            thresh=affs_thresh,
            labels_mask=labels_mask)[0]

    thresholds = [merge_thresh]
    affs = affinities[0:3]
    #affs = gaussian_filter(affs, [0,1,1.5,1.5])
    generator = waterz.agglomerate(
        affs=affs.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    return segmentation

for raw_file in raw_files:
    print(raw_file)
    out_file = zarr.open(raw_file)
    #in_file = raw_file.replace('out', 'images')
    #outshape = zarr.open(in_file)['raw'].shape

    merge_thresh = 0.05 #[0.5, 0.5, 0.5]
    affs_thresh = 0.65
    pred = np.asarray(out_file['pred_affs_400'])
    mask = np.max(pred,axis=0)>0.9 #np.asarray(out_file['pred_lsds_resize'][3])<0.4
    segmentation = get_segmentation(pred, merge_thresh=merge_thresh, affs_thresh=affs_thresh , labels_mask=mask)
    
    #segmentation = resize(segmentation, outshape, order=0, preserve_range=True, anti_aliasing=False)
    #segmentation = renumber(segmentation)
    
    #out_file['raw'] = zarr.open(in_file)['3d/raw']
    #out_file['raw'].attrs['offset'] = zarr.open(in_file)['3d/raw'].attrs['offset']
    #out_file['raw'].attrs['resolution'] = zarr.open(in_file)['3d/raw'].attrs['resolution']

    out_file['affs_seg_400'] = segmentation #labeled.astype(np.uint64)
    out_file['affs_seg_400'].attrs['offset'] = out_file['pred_affs_400'].attrs['offset'] 
    out_file['affs_seg_400'].attrs['resolution'] = out_file['pred_affs_400'].attrs['resolution']

