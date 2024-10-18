"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import zarr
import xml.etree.ElementTree as et
# from aicsimageio import AICSImage


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    # otherwise we return the *function* that can read ``path``.
    #if isinstance(path, str) and 
    if path.endswith(".xml"):
        return cellcounter_reader_function
    # elif path.endswith(".czi"):
    #     return czi_reader_function
    elif path.endswith(".csv"):
        return amira_csv_reader_function
    elif path.endswith(".zarr"):
        return zarr_reader_function
    elif path.endswith(".npy"):
        return npy_reader_function
    else:
        return None

# def czi_reader_function(file):
#     #nch = 
#     img = AICSImage(file).get_image_dask_data()#.get_image_dask_data("ZYX", C=1).compute()
#     add_kwargs = {}
#     layer_type = "image"  
#     return [(img, add_kwargs, layer_type)]

def amira_csv_reader_function(csvfile):
    
    #zyxres = read_tiff_voxel_size(imfi)

    #cilroot = imroot[0:imroot.find('-')]
    #csvfi = glob.glob(cil_roi_dir+cilroot+"*.csv")[0]
    xyz = np.genfromtxt(csvfile, delimiter=",", skip_header=1)[:,-3::]
    x = xyz[:,0] #/zyxres[2]
    y = xyz[:,1] #/zyxres[1]
    z = xyz[:,2] #/zyxres[0]
    data = np.stack((z,y,x), axis=1)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "points"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

def cellcounter_reader_function(xmlfile):
    try:
        tree = et.parse(xmlfile)
    except OSError:
        print('Failed to read XML file {}.'.format(xmlfile))
    
    root = tree.getroot()
    img_props = root[0]
    markers = root[1]
    
    x = np.array([int(elem.text) for elem in markers.findall(".//MarkerX")])
    y = np.array([int(elem.text) for elem in markers.findall(".//MarkerY")])
    z = np.array([int(elem.text) for elem in markers.findall(".//MarkerZ")])
    #n = len(x)
    #imfi = img_props.find("Image_Filename").text
    
    data = np.stack((z,y,x), axis=1)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "points"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

def npy_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def zarr_reader_function(path):
    """Take a path and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # optional kwargs for the corresponding viewer.add_* method
    img_kwargs = {}
    label_kwargs = {}

    zarr_fi = zarr.open(path)
    img = (zarr_fi['3d/raw'], img_kwargs, "image")
    labels = (zarr_fi['3d/labeled'].astype(int)[:], label_kwargs, "labels")

    return [img, labels]
