import zarr
import numpy as np
import glob
import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import normalize, save_out
from skimage.morphology import binary_dilation

check_dir = 'zarrs/train/'
check_files = sorted(glob.glob(check_dir + '*.zarr'))

for check_file in check_files:
    f = zarr.open(check_file)
    print(check_file)
    
    # max project:
    raw_max = normalize(
        np.max(f['raw'][:], axis=0).astype(np.uint16), 
        maxval=(2**16-1)
        ).astype(np.uint16)
    label_mask = np.max(f['labeled'][:], axis=0)>0
    # dilate mask:
    dilate_rounds = 20
    label_mask_dil = label_mask.copy()
    for i in range(dilate_rounds):
        label_mask_dil = binary_dilation(label_mask_dil)
    print(raw_max.shape)
    save_out(f, raw_max, 'raw_max', save2d=False)
    save_out(f, label_mask.astype(float), 'label_max', save2d=False)
    save_out(f, label_mask_dil.astype(float), 'label_max_dilated', save2d=False)
    
