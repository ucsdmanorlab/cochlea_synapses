import numpy as np
import zarr
import warnings

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
            zarr_file[f'2d/{key}/{z}'] = np.expand_dims(array[z], axis=0)
            zarr_file[f'2d/{key}/{z}'].attrs['offset'] = offset[1::]
            zarr_file[f'2d/{key}/{z}'].attrs['resolution'] = res[1::]
        
        zarr_file[f'3d/{key}'] = array
        zarr_file[f'3d/{key}'].attrs['offset'] = offset
        zarr_file[f'3d/{key}'].attrs['resolution'] = res
    
    else:
        zarr_file[key] = array
        zarr_file[key].attrs['offset'] = offset
        zarr_file[key].attrs['resolution'] = res

def _read_res(imgpath):
    if imgpath is None:
        return [1, 1, 1]
    elif imgpath.endswith('.tif') or imgpath.endswith('.tiff'):
        return _read_tiff_voxel_size(imgpath)
    elif imgpath.endswith('.lif') or imgpath.endswith('.czi') or imgpath.endswith('.nd2'):
        return _read_aics_voxel_size(imgpath)
    
def _read_tiff_voxel_size(file_path):
    import tifffile
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size
        return [z, y, x]

def _read_aics_voxel_size(file_path):
    import aicsimageio
    try:
        reader = aicsimageio.AICSImage(file_path)
    except Exception as e:
        print('error loading file', file_path, flush=True)
        return [1, 1, 1]

    pixel_size =[np.abs(reader.physical_pixel_sizes.Z), reader.physical_pixel_sizes.Y, reader.physical_pixel_sizes.X]
    return pixel_size

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
