import glob
import numpy as np
import zarr
import os.path

import warnings
import pandas as pd

from aicsimageio import AICSImage
from tqdm import tqdm

from scipy.ndimage import gaussian_filter, binary_closing, distance_transform_edt, center_of_mass
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

#### Parameters to set: ####
rawdir = 'amira/' #spinningdisk/raw/'
############################

outdir = rawdir.replace('amira','zarrs')

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data.max()
    print(scale_fact)
    data_norm = data * scale_fact
    return data_norm

def amira_fixshape(data, ch):
    cropx = ch*4
    left_part = data.get_image_dask_data("ZYX", T=0, C=ch)[:,:,cropx::]
    right_part = data.get_image_dask_data("ZYX", T=0, C=ch)[:,:,0:cropx]
    img = np.concatenate((left_part, right_part), axis=2)
    return img

def amira_readpos(csvfi, xy_res, z_res):
    pts = pd.read_csv(csvfi, usecols=['CenterX', 'CenterY', 'CenterZ'])
    pts = [pts['CenterX']/xy_res, pts['CenterY']/xy_res, pts['CenterZ']/z_res]
    pts = np.array(pts)

    return pts

def get_slices(rad_xy, rad_z, loc, shape):
    x1 = max(loc[2] - rad_xy, 0) ;
    x2 = min(loc[2] + rad_xy, shape[2]) ;
    y1 = max(loc[1] - rad_xy, 0) ;
    y2 = min(loc[1] + rad_xy, shape[1]) ;
    z1 = max(loc[0] - rad_z, 0) ;
    z2 = min(loc[0] + rad_z, shape[0]) ;
    relx = loc[2] - x1 ;
    rely = loc[1] - y1 ;
    relz = loc[0] - z1 ;

    return slice(z1,z2), slice(y1,y2), slice(x1,x2), [relz, rely, relx]

def dist_watershed_sep(mask, loc):
    dists = distance_transform_edt(mask, sampling=[4,1,1])

    indices = peak_local_max(dists, labels=mask)
    pks = np.zeros(dists.shape, dtype=bool)
    pks[tuple(indices.T)] = True

    pk_labels = label(pks)
    if pk_labels.max()>1:
        merged_peaks = center_of_mass(pks, pk_labels, index=range(1, pk_labels.max()+1))
        merged_peaks = np.round(merged_peaks).astype('int')

        markers = np.zeros_like(mask, dtype='int')
        for i in range(merged_peaks.shape[0]):
            markers[merged_peaks[i,0], merged_peaks[i,1], merged_peaks[i,2]] = i+1

        labels = watershed(-dists, markers=markers, mask=mask)
        wantLabel = labels[loc[0], loc[1], loc[2]]
        mask_out = labels == wantLabel

    else:
        mask_out = mask

    return mask_out

def pts_to_labels(
    img,
    pts,
    snap_to_max = True,
    snap_rad = 1,
    max_rad_xy = 2,
    max_rad_z = 1,
    rad_xy = 4,
    rad_z = 3,
    blur_sig = [0.5, 0.7, 0.7], # z, y, x
    ):

    x = pts[0,:]
    y = pts[1,:]
    z = pts[2,:]
    n = x.shape[0]

    print("pre-processing image...")
    img_inv = gaussian_filter(img.min()+img.max() - img, blur_sig)

    w = img.shape[2]
    h = img.shape[1]
    d = img.shape[0]

    print("finding peaks...")
    # make markers:
    markers = np.zeros_like(img, dtype='int')

    for j in tqdm(range(n)):
        pos = np.round([z[j], y[j], x[j]]).astype('int')
        if snap_to_max:
            zrange, yrange, xrange, rel_pos = get_slices(snap_rad, snap_rad, pos, img.shape)
            pointIntensity = img_inv[zrange, yrange, xrange]

            shift = np.unravel_index(np.argmin(pointIntensity), pointIntensity.shape)
            shift = np.asarray(shift)-snap_rad

            z[j] = z[j] + shift[0]
            y[j] = y[j] + shift[1]
            x[j] = x[j] + shift[2]

            pos = np.round([z[j], y[j], x[j]]).astype('int')
        markers[pos[0], pos[1], pos[2]] = j+1

    print("masking...")
    # make mask:
    mask = np.zeros_like(img, dtype='bool')

    for j in tqdm(range(n)):
        pos = np.round([z[j], y[j], x[j]]).astype('int')
        pointIntensity = img_inv[pos[0], pos[1], pos[2]]

        # find local min (inverted max) value:
        zrange, yrange, xrange, rel_pos = get_slices(max_rad_xy, max_rad_z, pos, img.shape)
        subim = img_inv[zrange, yrange, xrange]
        local_min = subim.min()

        # get local region to threshold, find local min value:
        zrange, yrange, xrange, rel_pos = get_slices(rad_xy, rad_z, pos, img.shape)
        subim = img_inv[zrange, yrange, xrange]
        local_max = subim.max() # background

        # threshold:
        thresh = 0.5*local_min + 0.5*local_max
        if thresh < pointIntensity:
            print("threshold overriden for spot "+str(j)+" "+str(thresh)+" "+str(pointIntensity))
            thresh = 0.5*local_max + 0.5*pointIntensity
        subim_mask = subim <= thresh

        # check for multiple objects:
        sublabels = label(subim_mask)
        if sublabels.max() > 1:
            wantLabel = sublabels[rel_pos[0], rel_pos[1], rel_pos[2]]
            subim_mask = sublabels == wantLabel

            # recheck max:
            thresh2 = 0.5*subim[subim_mask].min() + 0.5*subim.max()
            if thresh < thresh2:
                subim_mask = subim <= thresh2
                sublabels = label(subim_mask)
                wantLabel = sublabels[rel_pos[0], rel_pos[1], rel_pos[2]]
                subim_mask = sublabels == wantLabel

        pt_solidity = regionprops(subim_mask.astype('int'))[0].solidity

        if pt_solidity < 0.8:
            subim_mask = dist_watershed_sep(subim_mask, rel_pos)

        submask = mask[zrange, yrange, xrange]
        submask = np.logical_or(submask, subim_mask)

        mask[zrange, yrange, xrange] = submask

    outlabels = watershed(img_inv, markers=np.array(markers), mask=mask)

    return outlabels

def splitfiles(
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
        test = testval[n_val::] #(n_val + n_test)]
        
        for x in val:
            fi_split[x] = 1
        for x in test:
            fi_split[x] = 2

    return fi_split

count = 0 
raw_files = sorted(glob.glob(rawdir + '*.am'))
fi_groups = [i[6:13] for i in raw_files]

fi_split = splitfiles(raw_files, grouplist = fi_groups)

splits = ['train', 'validate', 'test']
for i in np.unique(fi_split):
    outpath = os.path.join(outdir, splits[i])
    if not os.path.exists(outpath):
        os.mkdir(outpath)

for rawfile in raw_files:
    print(count, rawfile, fi_groups[count], splits[fi_split[count]])
    data = AICSImage(rawfile)
    
    xy_pix = data.physical_pixel_sizes.X
    z_pix = data.physical_pixel_sizes.Z

    csvfi = rawfile.replace(".am", ".iso45r.csv")
    pts = amira_readpos(csvfi, xy_pix, z_pix)

    img = amira_fixshape(data,2)
    labels = pts_to_labels(img,
        pts, snap_to_max=False)

    raw = normalize(img.astype(np.uint8), maxval=(2**8-1)).astype(np.uint8)

    fileroot = rawfile[len(rawdir):-len('.am')]

    outpath = os.path.join(outdir, splits[fi_split[count]], fileroot.replace('.','_')+'.zarr')

    out = zarr.open(outpath, 'w')

    for ds_name, data in [
            ('raw', np.array(raw, dtype=np.uint8)),
            ('labeled', np.array(labels, dtype=float))]:
        print(f'3d/{ds_name}', data.shape)#, data, outpath)
        # write 3d data
        out[f'3d/{ds_name}'] = data
        out[f'3d/{ds_name}'].attrs['offset'] = [0,]*3
        out[f'3d/{ds_name}'].attrs['resolution'] = [1,]*3
        
        # write 2d data
        for z in range(data.shape[0]):
            out[f'2d/{ds_name}/{z}'] = np.expand_dims(data[z], axis=0)
            out[f'2d/{ds_name}/{z}'].attrs['offset'] = [0,]*2
            out[f'2d/{ds_name}/{z}'].attrs['resolution'] = [1,]*2

    count += 1
