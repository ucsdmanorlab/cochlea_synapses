import numpy as np
import zarr

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data_norm.max()
    data_norm = data_norm * scale_fact
    return data_norm.astype(dtype)

def save_out(
        zarr_file,
        array,
        key,
        res=None,
        offset=None,
        save2d=False):

    if res is None:
        res = [1,]*len(array.shape)
    if offset is None:
        offset = [0,]*len(array.shape)

    if save2d:
        for z in range(array.shape[0]):
            zarr_file[f'3d/{key}/{z}'] = np.expand_dims(array[z], axis=0)
            zarr_file[f'3d/{key}/{z}'].attrs['offset'] = offset[1::]
            zarr_file[f'3d/{key}/{z}'].attrs['resolution'] = res[1::]
        
        zarr_file[f'3d/{key}'] = array
        zarr_file[f'3d/{key}'].attrs['offset'] = offset
        zarr_file[f'3d/{key}'].attrs['resolution'] = res
    
    else:
        zarr_file[key] = array
        zarr_file[key].attrs['offset'] = offset
        zarr_file[key].attrs['resolution'] = res


def split_files(
        imfis,
        grouplist = None,
        val_frac = 0.15,
        test_frac = 0.1,
        ):

    if grouplist == None:
        grouplist = [1,]*len(imfis)

    groups, counts = np.unique(grouplist, return_counts=True)

    fi_split = [0,]*len(imfis)

    for (gr, c) in zip(groups, counts):

        if test_frac > 0:
            n_test = int(np.max([1, np.round(test_frac * c)]))
        else:
            n_test = 0
        if val_frac > 0 and (c-n_test) > 0:
            n_val = int(np.max([1, np.round(val_frac * c)]))
        else:
            n_val = 0

        if (c-n_val-n_test) < 1:
            warnings.warn('All data going to test/val!')

        im_ids = np.where(np.array(grouplist) == gr)[0]
        testval = np.random.choice(im_ids, n_val + n_test, replace=False)
        val = testval[0:n_val]
        test = testval[n_val:(n_val + n_test)]

        for x in val:
            fi_split[x] = 1
        for x in test:
            fi_split[x] = 2

    return fi_split
