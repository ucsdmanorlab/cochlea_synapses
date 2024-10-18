import zarr
import glob
import numpy as np

zarrlist = glob.glob('../../../01_data/zarrs/train/spinning*.zarr')

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data_norm.max()
    data_norm = data_norm * scale_fact
    print(scale_fact)
    return data_norm


for zarrfi in zarrlist:
    print(zarrfi)
    img = zarr.open(zarrfi)
    norm_data = normalize(np.asarray(img['3d/raw']).astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)
    
    img['3d/raw'] = norm_data
    img['3d/raw'].attrs['offset'] = [0,]*3
    img['3d/raw'].attrs['resolution'] = [1,]*3
 
    for z in range(norm_data.shape[0]):
                img[f'2d/raw/{z}'] = np.expand_dims(norm_data[z], axis=0)
                img[f'2d/raw/{z}'].attrs['offset'] = [0,]*2
                img[f'2d/raw/{z}'].attrs['resolution'] = [1,]*2


