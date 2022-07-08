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
from skimage.transform import rescale
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
        scale_interval=(1.0, 1.0),
        scale_factor=(1.0, 1.0, 1.0),
        subsample=1,
        spatial_dims=3,
    ):

        self.scale_min = scale_interval[0]
        self.scale_max = scale_interval[1]
        self.scale_factor = scale_factor
        self.subsample = subsample
        self.spatial_dims = spatial_dims

    def prepare(self, request):
        # get the voxel size
        self.voxel_size = self.__get_common_voxel_size(request)

        # get the total ROI of all requests
        total_roi = request.get_total_roi()
        logger.debug("total ROI is %s" % total_roi)

        # First, get the total ROI of the request in spatial dimensions only.
        # Channels and time don't matter. This is our master ROI.

        # get master ROI
        master_roi = Roi(
            total_roi.get_begin()[-self.spatial_dims :],
            total_roi.get_shape()[-self.spatial_dims :],
        )
        self.spatial_dims = master_roi.dims()
        logger.debug("master ROI is %s" % master_roi)

        # make sure the master ROI aligns with the voxel size
        master_roi = master_roi.snap_to_grid(self.voxel_size, mode="grow")
        logger.debug("master ROI aligned with voxel size is %s" % master_roi)
        
        # get master roi in voxels
        master_roi_voxels = master_roi / self.voxel_size
        logger.debug("master ROI in voxels is %s" % master_roi_voxels)

        # Second, create a master transformation. This is a transformation that
        # covers all voxels of the all requested ROIs. The master transformation
        # is zero-based.

        # create a transformation with the size of the master ROI in voxels
        self.master_transformation = self.__create_transformation(
            master_roi_voxels.get_shape()
        )

        # Third, crop out parts of the master transformation for each of the
        # smaller requested ROIs. Since these ROIs now have to align with the
        # voxel size (which for points does not have to be the case), we also
        # remember these smaller ROIs as target_rois in global world units.

        # crop the parts corresponding to the requested ROIs
        self.transformations = {}
        self.target_rois = {}
        deps = BatchRequest()
        for key, spec in request.items():

            spec = spec.copy()

            if spec.roi is None:
                continue

            target_roi = Roi(
                spec.roi.get_begin()[-self.spatial_dims :],
                spec.roi.get_shape()[-self.spatial_dims :],
            )
            logger.debug("downstream request spatial ROI for %s is %s", key, target_roi)

            # make sure the target ROI aligns with the voxel grid (which might
            # not be the case for points)
            target_roi = target_roi.snap_to_grid(self.voxel_size, mode="grow")
            logger.debug(
                "downstream request spatial ROI aligned with voxel grid for %s "
                "is %s",
                key,
                target_roi,
            )

            # remember target ROI (this is where the transformation will project
            # to)
            self.target_rois[key] = target_roi

            # get ROI in voxels
            target_roi_voxels = target_roi / self.voxel_size

            # get ROI relative to master ROI
            target_roi_in_master_roi_voxels = (
                target_roi_voxels - master_roi_voxels.get_begin()
            )

            # crop out relevant part of transformation for this request
            transformation = np.copy(
                self.master_transformation[
                    (slice(None),) + target_roi_in_master_roi_voxels.get_bounding_box()
                ]
            )
            self.transformations[key] = transformation

            # get ROI of all voxels necessary to perfrom transformation
            #
            # for that we follow the same transformations to get from the
            # request ROI toi the target ROI in master ROI in voxels, just in
            # reverse
            source_roi_in_master_roi_voxels = self.__get_source_roi(
                    target_roi_in_master_roi_voxels, self.scale_factor)
            source_roi_voxels = (
                source_roi_in_master_roi_voxels + master_roi_voxels.get_begin()
            )
            source_roi = source_roi_voxels * self.voxel_size
            # transformation is still defined on voxels relative to master ROI
            # in voxels (i.e., lowest source coordinate could be 5, but data
            # array we get later starts at 0).
            #
            # shift transformation to be indexed relative to beginning of
            # source_roi_voxels
            self.__shift_transformation(
                -source_roi_in_master_roi_voxels.get_begin(), transformation
            )

            # update upstream request
            spec.roi = Roi(
                spec.roi.get_begin()[: -self.spatial_dims]
                + source_roi.get_begin()[-self.spatial_dims :],
                spec.roi.get_shape()[: -self.spatial_dims]
                + source_roi.get_shape()[-self.spatial_dims :],
            )
            print('spec roi')
            print(spec.roi)

            deps[key] = spec

            logger.debug("upstream request roi for %s = %s" % (key, spec.roi))

        return deps

    def process(self, batch, request):

        for (array_key, array) in batch.arrays.items():

            if array_key not in self.target_rois:
                continue

            # for arrays, the target ROI and the requested ROI should be the
            # same in spatial coordinates
            assert (
                self.target_rois[array_key].get_begin()
                == request[array_key].roi.get_begin()[-self.spatial_dims :]
            ), "Target roi offset {} does not match request roi offset {}".format(
                self.target_rois[array_key].get_begin(),
                request[array_key].roi.get_begin()[-self.spatial_dims :],
            )

            assert (
                self.target_rois[array_key].get_shape()
                == request[array_key].roi.get_shape()[-self.spatial_dims :]
            ), "Target roi offset {} does not match request roi offset {}".format(
                self.target_rois[array_key].get_shape(),
                request[array_key].roi.get_shape()[-self.spatial_dims :],
            )

            shape = array.data.shape
            channel_shape = shape[: -self.spatial_dims]
            data = array.data.reshape((-1,) + shape[-self.spatial_dims :])

            # apply transformation on each channel
            data = np.array(
                [
                    rescale(data[c], self.scale_factor)
                    for c in range(data.shape[0])
                ]
            )

            data = data.reshape((channel_shape + data.shape[-self.spatial_dims:]))
            
            data_roi = request[array_key].roi/self.spec[array_key].voxel_size

            cropeds = np.asarray(data.shape) - np.asarray(data_roi.get_shape()[-self.spatial_dims:])
            crop1 = np.floor(cropeds/2).astype(int)
            crop2 = np.ceil(cropeds/2).astype(int)

            array.data = crop(data, list(zip(crop1, crop2))) #.reshape(channel_shape + data_roi.get_shape()[-self.spatial_dims:])

            # restore original ROIs
            array.spec.roi = request[array_key].roi
            print('resized array')

        for (graph_key, graph) in batch.graphs.items():

            nodes = list(graph.nodes)

            for node in nodes:
                # logger.debug("projecting %s", node.location)

                # get location relative to beginning of upstream ROI
                location = node.location - graph.spec.roi.get_begin()
                logger.debug("relative to upstream ROI: %s", location)

                # get spatial coordinates of node in voxels
                location_voxels = location[-self.spatial_dims :] / self.voxel_size

                # get projected location in transformation data space, this
                # yields voxel coordinates relative to target ROI
                print('location_voxels')
                print(location_voxels)
                projected_voxels = self.__project(
                    self.transformations[graph_key], location_voxels
                )
                print('projected voxels:')
                print(projected_voxels)
                logger.debug(
                    "projected in voxels, relative to target ROI: %s", projected_voxels
                )

                if projected_voxels is None:
                    logger.debug("node outside of target, skipping")
                    graph.remove_node(node, retain_connectivity=True)
                    continue

                # convert to world units (now in float again)
                projected = projected_voxels * np.array(self.voxel_size)

                logger.debug(
                    "projected in world units, relative to target ROI: %s",
                    projected)

                # get global coordinates
                projected += np.array(self.target_rois[graph_key].get_begin())

                # update spatial coordinates of node location
                node.location[-self.spatial_dims:] = projected

                logger.debug("final location: %s", node.location)

                # finally, it can happen that a node no longer is contained in
                # the requested ROI (because larger ROIs than necessary have
                # been requested upstream)
                if not request[graph_key].roi.contains(node.location):
                    logger.debug("node outside of target, skipping")
                    graph.remove_node(node, retain_connectivity=True)
                    continue

            # restore original ROIs
            graph.spec.roi = request[graph_key].roi

    def __get_common_voxel_size(self, request):

        voxel_size = None
        prev = None
        for array_key in request.array_specs.keys():
            if voxel_size is None:
                voxel_size = self.spec[array_key].voxel_size[-self.spatial_dims:]
            elif self.spec[array_key].voxel_size is not None:
                assert voxel_size == self.spec[array_key].voxel_size[-self.spatial_dims:], \
                        "ElasticAugment can only be used with arrays of same voxel sizes, " \
                            "but %s has %s, and %s has %s."%(
                                array_key, self.spec[array_key].voxel_size,
                                prev, self.spec[prev].voxel_size)
            prev = array_key

        if voxel_size is None:
            raise RuntimeError("voxel size must not be None")
        return voxel_size

    def __create_transformation(self, target_shape):

        scale = self.scale_min + random.random()*(
            self.scale_max - self.scale_min
        )

        transformation = augment.create_identity_transformation(
            target_shape, subsample=self.subsample, scale=scale
        )
        
        return transformation

    def __project(self, transformation, location):
        """Find the projection of location given by transformation. Returns None
        if projection lies outside of transformation."""

        dims = len(location)

        # subtract location from transformation
        diff = transformation.copy()
        for d in range(dims):
            diff[d] -= location[d]

        # square
        diff2 = diff * diff

        # sum
        dist = diff2.sum(axis=0)

        # find grid point closes to location
        center_grid = Coordinate(np.unravel_index(dist.argmin(), dist.shape))
        center_source = self.__source_at(transformation, center_grid)

        logger.debug("projecting %s onto grid", location)
        logger.debug("grid shape: %s", transformation.shape[1:])
        logger.debug("grid projection: %s", center_grid)
        logger.debug("dist shape: %s", dist.shape)
        logger.debug("dist.argmin(): %s", dist.argmin())
        logger.debug("dist[argmin]: %s", dist[center_grid])
        logger.debug("transform[argmin]: %s",
                     transformation[(slice(None),) + center_grid])
        logger.debug("min dist: %s", dist.min())
        logger.debug("center source: %s", center_source)

        # inspect grid edges incident to center_grid
        for d in range(dims):

            # nothing to do for dimensions without spatial extent
            if transformation.shape[1 + d] == 1:
                continue

            dim_vector = tuple(1 if dd == d else 0 for dd in range(dims))
            pos_grid = center_grid + dim_vector
            neg_grid = center_grid - dim_vector
            logger.debug("interpolating along %s", dim_vector)

            pos_u = -1
            neg_u = -1

            if pos_grid[d] < transformation.shape[1 + d]:
                pos_source = self.__source_at(transformation, pos_grid)
                logger.debug("pos source: %s", pos_source)
                pos_dist = pos_source[d] - center_source[d]
                loc_dist = location[d] - center_source[d]
                if pos_dist != 0:
                    pos_u = loc_dist / pos_dist
                else:
                    pos_u = 0

            if neg_grid[d] >= 0:
                neg_source = self.__source_at(transformation, neg_grid)
                logger.debug("neg source: %s", neg_source)
                neg_dist = neg_source[d] - center_source[d]
                loc_dist = location[d] - center_source[d]
                if neg_dist != 0:
                    neg_u = loc_dist / neg_dist
                else:
                    neg_u = 0

            logger.debug("pos u/neg u: %s/%s", pos_u, neg_u)

            # if a point only falls behind edges, it lies outside of the grid
            if pos_u < 0 and neg_u < 0:
                return None

        return np.array(center_grid, dtype=np.float32)

    def __source_at(self, transformation, index):
        """Read the source point of a transformation at index."""

        slices = (slice(None),) + tuple(slice(i, i + 1) for i in index)
        return transformation[slices].flatten()

    def __get_source_roi(self, target_roi, scale_factor): #self, transformation):
        
        cent = target_roi.get_center() 

        shape = np.ceil(np.asarray(target_roi.get_shape())/np.asarray(scale_factor) + 1) 

        bb_min = np.floor(np.asarray(cent) - shape/2) 
        
        source_roi = Roi(tuple(bb_min), tuple(shape))

        return source_roi

    def __shift_transformation(self, shift, transformation):

        for d in range(transformation.shape[0]):
            transformation[d] += shift[d]


