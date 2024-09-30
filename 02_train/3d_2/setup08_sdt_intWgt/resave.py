import glob
import zarr
def save_out(outfile,
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

model = 'model_2024-04-15_12-39-03_3d_sdt_IntWgt_b1'
checkpoint = model+'_checkpoint_10000'
raw_files = glob.glob('../../../01_data/zarrs/cilcare/*.zarr')

for raw_file in raw_files:
        print(raw_file)
        out_file = zarr.open(raw_file.replace('01_data/zarrs/cilcare', '03_predict/3d/cilcare_all/sdt'))
        save_out(out_file, zarr.open(raw_file)['3d/labeled'][:], 'gt_labels')
        save_out(out_file, zarr.open(raw_file)['3d/convex'][:], 'convex')
