import logging
import math
import numpy as np
import random
from scipy import ndimage

from gunpowder import * #.batch_filter import * #BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.ext import augment
from gunpowder.roi import Roi
from gunpowder.array import ArrayKey
from skimage.transform import rescale, resize
from skimage.util import crop
logger = logging.getLogger(__name__)

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(lambda a, da: a+da, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

class ReScale(BatchFilter):
    def __init__(
        self,
        array,
        scale_factor=(1.0, 1.0, 1.0),
        order = 1,
        preserve_range = True,
        anti_aliasing = True,
    ):
        self.array = array
        self.scale_factor = scale_factor
        self.order = order
        self.preserve_range = preserve_range
        self.anti_aliasing = anti_aliasing

    def process(self, batch, request):
        
        voxel_size = self.spec[self.array].voxel_size
        data_roi = request[self.array].roi
        data_roi_vox = data_roi / voxel_size

        out_shape = np.asarray(data_roi_vox.get_shape())

        source_shape = np.ceil(out_shape / np.asarray(self.scale_factor))
        context      = tuple(np.ceil(np.ceil(source_shape - out_shape)/2)*np.asarray(voxel_size))
        #context1 = tuple(np.floor(context/2)*np.asarray(voxel_size))
        #context2 = tuple(np.ceil(context/2)*np.asarray(voxel_size))
        source_roi = data_roi.grow(context, context)
        data = batch[self.array].crop(source_roi)
        
        temp_shape = np.ceil(
                np.asarray((source_roi / voxel_size).get_shape()) *
                np.asarray(self.scale_factor))
        scaleddata = resize(data.data, 
                tuple(temp_shape), 
                order=self.order,
                preserve_range=self.preserve_range,
                anti_aliasing=self.anti_aliasing
                ).astype(data.data.dtype)

        context = out_shape - temp_shape
        #print(out_shape, temp_shape)
        #print(scaleddata.shape())
        #start = np.floor(context/2)
        #end = temp_shape - np.ceil(context/2)
        scaleddata = cropND(scaleddata, tuple(out_shape))
        #print(scaleddata.shape())

        batch[self.array].data = scaleddata

