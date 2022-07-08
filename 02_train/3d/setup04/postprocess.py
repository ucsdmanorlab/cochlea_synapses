import glob
import numpy as np
import zarr
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.filters import gaussian
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from time import time

raw_files = sorted(glob.glob('../../../01_data/zarrs/nih/ctbp2_prediction/*.zarr'))

for raw_file in raw_files:
    print(raw_file)
    out_file = zarr.open(raw_file)

    pred = np.asarray(out_file['pred_mask'])
    
    thresh = 0.05 #threshold_otsu(pred)
    tic = time()
    thresholded = pred > 0.3
    pred_gaus = gaussian(pred, sigma=1.5, preserve_range=True)
    markers = h_maxima(pred_gaus*thresholded, thresh) #label(thresholded)
    print(time()-tic)
    print(np.max(label(markers)))

    labeled = watershed(np.multiply(pred, -1), label(markers*thresholded), mask = thresholded)


    out_file['labeled'] = labeled.astype(np.uint64)
    out_file['labeled'].attrs['offset'] = [0,]*3 #[10, 46, 46] #[0,]*3
    out_file['labeled'].attrs['resolution'] = [1,]*3

