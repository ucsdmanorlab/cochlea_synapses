import os
import numpy as np
import tifffile
import zarr

target_list = ["spinningdisk_F2307V_16kHz_a_100x.zarr",
                "spinningdisk_F2307V_32kHz_a_100x.zarr",
                "spinningdisk_F2307V_8kHz_a_100x.zarr",
                "spinningdisk_F2308V_16kHz_a_100x.zarr",
                "spinningdisk_F2308V_32kHz_a_100x.zarr",
                "spinningdisk_F2308V_8kHz_a_100x.zarr",
                "spinningdisk_F2352DOI_8kHz_a_100x.zarr",
                "spinningdisk_F2352DOI_8kHz_b_100x.zarr",
                "spinningdisk_F2353DOI_16kHz_a_100x.zarr",
                "spinningdisk_F2353DOI_32kHz_a_100x.zarr",
                "spinningdisk_F2354DOI_32kHz_a_100x.zarr",
                "spinningdisk_F2354DOI_8kHz_a_100x.zarr",
                "spinningdisk_F2352DOI_16kHz_a_100x.zarr",
                "spinningdisk_F2352DOI_32kHz_a_100x.zarr",
                "spinningdisk_F2353DOI_8kHz_a_100x.zarr",
                "spinningdisk_F2354DOI_16kHz_a_100x.zarr",]

groups = [0,]*12
for i in [1, 1, 2, 2]:
    groups.append(i)

tif_dir = "/home/caylamiller/workspace/cochlea_synapses/01_data/tifs/myo7a/"
tif_list = os.listdir(tif_dir)

def normalize(data, maxval=1., dtype=np.uint16):
    data = data.astype(dtype)
    data_norm = data - data.min()
    scale_fact = maxval/data_norm.max()
    data_norm = data_norm * scale_fact
    return data_norm

for i in range(len(target_list)):
    target_tif = target_list[i].replace('.zarr', '.tif')
    print(target_tif)
    print(os.path.isfile(tif_dir+target_tif))
    img = tifffile.imread(tif_dir+target_tif)
    #break

    if groups[i] == 0:
        out_dir = "/home/caylamiller/workspace/cochlea_synapses/01_data/myo7a/train/"
    elif groups[i] == 1:
        out_dir = "/home/caylamiller/workspace/cochlea_synapses/01_data/myo7a/validate/"
    elif groups[i] == 2:
        out_dir = "/home/caylamiller/workspace/cochlea_synapses/01_data/myo7a/test/"
        
    out = zarr.open(out_dir+target_list[i]) 
    
    for ds_name, data in [
            ('myo', img[:,0,:,:]),
            ('ctbp2', img[:,2,:,:])]:
        
        data = normalize(data.astype(np.uint16), maxval=(2**16-1)).astype(np.uint16)

        # write 3d data
        out[f'3d/{ds_name}'] = data
        out[f'3d/{ds_name}'].attrs['offset'] = [0,]*3
        out[f'3d/{ds_name}'].attrs['resolution'] = [1,]*3
    
        # write 2d data
        for z in range(data.shape[0]):
            out[f'2d/{ds_name}/{z}'] = np.expand_dims(data[z], axis=0)
            out[f'2d/{ds_name}/{z}'].attrs['offset'] = [0,]*2
            out[f'2d/{ds_name}/{z}'].attrs['resolution'] = [1,]*2

