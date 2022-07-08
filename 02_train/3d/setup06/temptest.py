import glob
import numpy as np
import waterz
import zarr

from scipy.ndimage import label
from scipy.ndimage import measurements
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import resize
from skimage.segmentation import watershed

#raw_files = sorted(glob.glob('../../../01_data/zarrs/nih/ctbp2_prediction_affs_lsds/*3.zarr'))
#raw_files = sorted(glob.glob('../../../01_data/zarrs/ctbp2/prediction_affs_lsds/validation/*.zarr'))
raw_files = sorted(glob.glob('../../../01_data/zarrs/sampled_out/*10.zarr'))


for raw_file in raw_files:
    print(raw_file)
    out_file = zarr.open(raw_file)
    in_file = out_file #raw_file.replace('out', 'images')
    #print(in_file)
    outshape = zarr.open(in_file)['raw'].shape

    thresh = 0.5 #[0.5, 0.5, 0.5]
    pred = np.asarray(out_file['pred_lsds_resize'])

    print(pred.shape)

    max_affs = np.amax(pred, axis=0) #np.fmax(affs[0], affs[1]), affs[2])
    mean_affs = np.mean(pred[0:3], axis=0)
    std_affs = np.std(pred, axis=0)

    print(max_affs.shape, mean_affs.shape, std_affs.shape)

    #segmentation = get_segmentation(pred, thresh) #next(waterz.agglomerate(affs=pred, thresholds=thresh))

    #segmentation = resize(segmentation, outshape, order=0, preserve_range=True, anti_aliasing=False)

    out_file['mean_test'] = mean_affs #labeled.astype(np.uint64)
    out_file['mean_test'].attrs['offset'] = out_file['pred_affs_resize'].attrs['offset']
    out_file['mean_test'].attrs['resolution'] = out_file['pred_affs_resize'].attrs['resolution']

    #out_file['max_test'] = max_affs #labeled.astype(np.uint64)
    #out_file['max_test'].attrs['offset'] = out_file['pred_affs_resize'].attrs['offset']
    #out_file['max_test'].attrs['resolution'] = out_file['pred_affs_resize'].attrs['resolution']

    #out_file['std_test'] = std_affs #labeled.astype(np.uint64)
    #out_file['std_test'].attrs['offset'] = out_file['pred_affs_resize'].attrs['offset']
    #out_file['std_test'].attrs['resolution'] = out_file['pred_affs_resize'].attrs['resolution']

    for ds_name, num in [
            ('lsd1', 0),
            ('lsd2', 1),
            ('lsd3', 2),
            ('lsd4', 3) ]:

        # write 3d data
        out_file[f'{ds_name}'] = pred[num]
        out_file[f'{ds_name}'].attrs['offset'] = [0,]*3
        out_file[f'{ds_name}'].attrs['resolution'] = [1,]*3

