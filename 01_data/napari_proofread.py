import zarr

a = zarr.open('zarrs/train/WPZ104L_CtBP2_GluR2_NF_Myo7_IHC_11_3_v2.zarr')
viewer.add_image(a['3d/raw'])

working_labels = a['3d/labeled'].astype(int)[:]
viewer.add_labels(working_labels)
working_labels.max()

## make edits
a['3d/labeled'] = working_labels
a['3d/labeled'].attrs['offset'] = [0,]*3
a['3d/labeled'].attrs['resolution'] = [1,]*3

for z in range(working_labels.shape[0]):
    a[f'2d/labeled/{z}'] = np.expand_dims(working_labels[z], axis=0)
    a[f'2d/labeled/{z}'].attrs['offset'] = [0,]*2
    a[f'2d/labeled/{z}'].attrs['resolution'] = [1,]*2

