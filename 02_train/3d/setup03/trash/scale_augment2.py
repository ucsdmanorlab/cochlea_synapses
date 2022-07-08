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


class ScaleAugment(BatchFilter):
    """Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.

    Args:


        rotation_interval (``tuple`` of two ``floats``):

            Interval to randomly sample rotation angles from (0, 2PI).

        scale_interval (``tuple`` of two ``floats``):

            Interval to randomly sample scale factors from.

        prob_slip (``float``):

            Probability of a section to "slip", i.e., be independently moved in
            x-y.

        prob_shift (``float``):

            Probability of a section and all following sections to move in x-y.


        subsample (``int``):

            Instead of creating an elastic transformation on the full
            resolution, create one subsampled by the given factor, and linearly
            interpolate to obtain the full resolution transformation. This can
            significantly speed up this node, at the expense of having visible
            piecewise linear deformations for large factors. Usually, a factor
            of 4 can savely by used without noticable changes. However, the
            default is 1 (i.e., no subsampling).

        spatial_dims (``int``):

            The number of spatial dimensions in arrays. Spatial dimensions are
            assumed to be the last ones and cannot be more than 3 (default).
            Set this value here to avoid treating channels as spacial
            dimension. If, for example, your array is indexed as ``(c,y,x)``
            (2D plus channels), you would want to set ``spatial_dims=2`` to
            perform the elastic deformation only on x and y.

        use_fast_points_transform (``bool``):

            By solving for all of your points simultaneously with the following
            3 step proceedure:
            1) Rasterize nodes into numpy array
            2) Apply elastic transform to array
            3) Read out nodes via center of mass of transformed points
            You can gain substantial speed up as opposed to calculating the
            elastic transform for each point individually. However this may
            lead to nodes being lost during the transform.

        recompute_missing_points (``bool``):

            Whether or not to compute the elastic transform node wise for nodes
            that were lossed during the fast elastic transform process.
    """

    def __init__(
        self,
        scale_factor=(1.0, 1.0, 1.0),
        spatial_dims = 3,
    ):
        self.scale_factor = scale_factor
        self.spatial_dims = 3

    def prepare(self, request):
        
        voxel_size = (5, 1, 1) 
        # get the total ROI of all requests
        total_roi = request.get_total_roi()
        # get master ROI
        master_roi = Roi(
            total_roi.get_begin()[-self.spatial_dims :],
            total_roi.get_shape()[-self.spatial_dims :],
        )
        self.spatial_dims = master_roi.dims()
        master_roi_voxels = master_roi / voxel_size

        self.target_rois = {}
        deps = BatchRequest()

        for key, spec in request.items():
            spec = spec.copy()

            target_roi = Roi(
                spec.roi.get_begin()[-self.spatial_dims :],
                spec.roi.get_shape()[-self.spatial_dims :],
            )

            voxel_size = self.spec[key].voxel_size

            target_roi_vox = target_roi / voxel_size
            target_roi_vox = target_roi_vox - master_roi_voxels.get_begin()
            
            self.target_rois[key] = target_roi_vox
            
            source_shape = np.asarray(target_roi_vox.get_shape()) / np.asarray(self.scale_factor)
            context      = np.ceil(source_shape - np.asarray(target_roi_vox.get_shape()))
            context1 = tuple(np.floor(context/2))
            context2 = tuple(np.ceil(context/2))

            source_roi_vox = target_roi_vox.grow(context1, context2)
            source_roi_vox = source_roi_vox + master_roi_voxels.get_begin()
            source_roi = source_roi_vox * voxel_size

            # update upstream request
            spec.roi = Roi(
                    spec.roi.get_begin()[: -self.spatial_dims]
                    + source_roi.get_begin()[: -self.spatial_dims], 
                    spec.roi.get_shape()[: -self.spatial_dims]
                    + source_roi.get_shape()[: -self.spatial_dims],
                    )

            deps[key] = spec
        '''# voxel size
        voxel_size = self.spec[self.array].voxel_size
        # roi:
        roi = request[self.array].roi
        roi = roi.snap_to_grid(voxel_size, mode="grow")
        roi_voxels = roi / voxel_size
        #print('roi')
        #print(roi)
        #print(roi_voxels)
        #print(self.scale_factor)
        # needed area:
        inshape = np.asarray(roi_voxels.get_shape()) / np.asarray(self.scale_factor)
        context = np.ceil(inshape - np.asarray(roi_voxels.get_shape()))
        context1 = tuple(np.floor(context/2))
        context2 = tuple(np.ceil(context/2))
        needed_roi = roi_voxels.grow(context1, context2)
        #print(needed_roi)
        needed_roi = needed_roi * voxel_size
        needed_roi = needed_roi.snap_to_grid(voxel_size, mode="grow")
        #print(needed_roi)
        # new request:
        deps = BatchRequest()
        deps[self.array] = needed_roi'''
        return deps

    def process(self, batch, request):
        
        for (array_key, array) in batch.arrays.items():
            
            if array_key not in self.target_rois:
                continue
            #voxel_size = self.spec[array_key].voxel_size
            data_roi = self.target_rois[array_key] #/ voxel_size #request[array_key].roi / voxel_size

            # rescale:
            #data = batch[self.array].data
            data = array.data
            print(data.shape)
            scaleddata = resize(data, tuple(data_roi.get_shape())).astype(data.dtype) #rescale(data, self.scale_factor)
            print(scaleddata.shape)

            array.data = scaleddata 
            array.spec.roi = request[array_key].roi
            #batch[self.array].data = scaleddata #rescale(data, self.scale_factor)
        # crop:
        #batch[self.array] = batch[self.array].crop(request[self.array].roi)

