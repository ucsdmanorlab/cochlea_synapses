import numpy as np
import zarr

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data_norm.max()
    data_norm = data_norm * scale_fact
    return data_norm.astype(dtype)

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
