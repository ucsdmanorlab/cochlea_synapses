import zarr
import napari

def load_labels(zarrfi):
    a = zarr.open(zarrfi)
    viewer = napari.current_viewer()
    viewer.add_image(a['3d/raw'])

    working_labels = a['3d/labeled'].astype(int)[:]
    viewer.add_labels(working_labels)
    print(working_labels.max())
    return working_labels

def check_count(working_labels):
    print(working_labels.max())

def save_labels(zarrfi, working_labels):
    a = zarr.open(zarrfi)

    ## make edits
    a['3d/labeled'] = working_labels
    a['3d/labeled'].attrs['offset'] = [0,]*3
    a['3d/labeled'].attrs['resolution'] = [1,]*3
    
    for z in range(working_labels.shape[0]):
        a[f'2d/labeled/{z}'] = np.expand_dims(working_labels[z], axis=0)
        a[f'2d/labeled/{z}'].attrs['offset'] = [0,]*2
        a[f'2d/labeled/{z}'].attrs['resolution'] = [1,]*2

