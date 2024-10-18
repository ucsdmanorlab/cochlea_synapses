import zarr
import glob
import numpy as np

zarrlist = glob.glob('../../../01_data/zarrs/model*wgtfg*/*.zarr')

for zarrfi in zarrlist:
    img = zarr.open(zarrfi)
    pred = img['pred'][:]

    img['pred'] = np.squeeze(pred)
