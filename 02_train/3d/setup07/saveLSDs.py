import zarr
import io
import numpy as np
import requests
import local_shape_descriptor

def save_out(out_file,
        array,
        key,
        res=[1,1,1],
        offset=[0,0,0]):

    out_file[key] = array
    out_file[key].attrs['offset'] = offset
    out_file[key].attrs['resolution'] = res

# get image
impath = '../../../01_data/zarrs/ctbp2/restest/validation/sample_3.zarr'
outfile = zarr.open(impath.replace('validation','out'))

labels = np.asarray(zarr.open(impath)['3d/labeled']) #[10:85,0:1740,150:850]

#labels[labels==0] = 500

# calc lsds
lsds = local_shape_descriptor.get_local_shape_descriptors(
              segmentation=labels,
              sigma=(10,)*3,
              voxel_size=[4,1,1])[0:3,:,:,:]

print(np.asarray(lsds).shape)
#comp = ((0,3))
#if type(comp) == int:
#                lsds = lsds[comp]
#else:
#    try:
#        num_els = len([i for j in comp[0:2] for i in j])
#
#        if num_els == 4 and len(comp) == 3:
#            lsds = np.concatenate(
#                    (
#                        lsds[comp[0][0]:comp[0][1]],
#                        lsds[comp[1][0]:comp[1][1]],
#                        lsds[comp[2]]
#                    ), axis=0)
#        elif num_els == 3:
#            lsds = np.concatenate(
#                    (
#                        lsds[comp[0][0]:comp[0][1]],
#                        lsds[comp[1]]
#                    ), axis=0)
#        else:
#            lsds = np.concatenate(
#                    (
#                        lsds[comp[0][0]:comp[0][1]],
#                        lsds[comp[1][0]:comp[1][1]]
#                    ), axis=0)
#    except:
#        lsds = lsds[comp[0]:comp[1]]
#
#print(np.asarray(lsds).shape)
save_out(outfile, 
        np.asarray(lsds).astype('float64'), 
        'gt_lsds',
        offset=[10,0,150])


