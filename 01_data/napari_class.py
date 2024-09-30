import numpy as np
import zarr

class synapse_labels():
    def __init__(self, zarrfi, viewer: "napari.viewer.Viewer"):
        self.viewer = viewer
        self.zarrfi = zarrfi
        self.load_labels()

    def load_labels(self):
        zarrfi = self.zarrfi
        a = zarr.open(zarrfi)
        self.viewer.add_image(a['3d/raw'])

        working_labels = a['3d/labeled'].astype(int)[:]
        self.viewer.add_labels(working_labels)
        self.check_count()
        #print('currently',working_labels.max(),'labels')
        
        self.working_labels = working_labels
    
    def check_count(self):
        print('currently',working_labels.max(),'labels')
    
    def save_labels(self):
        a = zarr.open(self.zarrfi)
    
        ## make edits
        a['3d/labeled'] = self.working_labels
        a['3d/labeled'].attrs['offset'] = [0,]*3
        a['3d/labeled'].attrs['resolution'] = [1,]*3
        
        for z in range(self.working_labels.shape[0]):
            a[f'2d/labeled/{z}'] = np.expand_dims(self.working_labels[z], axis=0)
            a[f'2d/labeled/{z}'].attrs['offset'] = [0,]*2
            a[f'2d/labeled/{z}'].attrs['resolution'] = [1,]*2

