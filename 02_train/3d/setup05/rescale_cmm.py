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


class ScaleNode(BatchFilter):
    def __init__(
        self,
        array,
        scale_factor=(1.0, 1.0, 1.0),
        order = 1,
    ):
        self.array = array
        self.scale_factor = scale_factor
        self.order = order

    def process(self, batch, request):
        
        voxel_size = self.spec[self.array].voxel_size
        data_roi = request[self.array].roi
        data_roi_vox = data_roi / voxel_size

        out_shape = np.asarray(data_roi_vox.get_shape())
        source_shape = out_shape / np.asarray(self.scale_factor)
        context      = np.ceil(source_shape - out_shape)
        context1 = tuple(np.floor(context/2)*np.asarray(voxel_size))
        context2 = tuple(np.ceil(context/2)*np.asarray(voxel_size))
        source_roi = data_roi.grow(context1, context2)
        data = batch[self.array].crop(source_roi)
        
        scaleddata = resize(data.data, tuple(out_shape), order=self.order)
        if self.order != 0:

            convertfactor = (np.max(data.data) - np.min(data.data))/ (np.max(scaleddata) - np.min(scaleddata))
            scaleddata = (scaleddata - np.min(scaleddata)) * convertfactor + np.min(data.data)

        batch[self.array].data = scaleddata.astype(data.data.dtype)

