import glob
import gunpowder as gp
import logging
import random
import math
import numpy as np
import sys
import torch
from gunpowder.torch import Train
from gunpowder.profiling import Timing

from funlib.learn.torch.models import UNet, ConvPass
from datetime import datetime


project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import Reject_CMM, WeightedMSELoss, calc_max_padding, ComputeDT, BalanceLabelsWithIntensity, CheckPred #CheckSnaps #check_snaps

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 1

class Reject(gp.BatchFilter):
    """Reject batches based on the masked-in vs. masked-out ratio.

    If a pipeline also contains a :class:`RandomLocation` node,
    :class:`Reject` needs to be placed downstream of it.

    Args:

        mask (:class:`ArrayKey`, optional):

            The mask to use, if any.

        min_masked (``float``, optional):

            The minimal required ratio of masked-in vs. masked-out voxels.
            Defaults to 0.5.

        ensure_nonempty (:class:`GraphKey`, optional)

            Ensures there is at least one point in the batch.

        reject_probability (``float``, optional):

            The probability by which a batch that is not valid (less than
            min_masked) is actually rejected. Defaults to 1., i.e. strict
            rejection.
    """

    def __init__(
        self, mask=None, min_masked=0.5, ensure_nonempty=None, reject_probability=1.0
    ):
        self.mask = mask
        self.min_masked = min_masked
        self.ensure_nonempty = ensure_nonempty
        self.reject_probability = reject_probability

    def setup(self):
        if self.mask:
            assert self.mask in self.spec, (
                "Reject can only be used if %s is provided" % self.mask
            )
        if self.ensure_nonempty:
            assert self.ensure_nonempty in self.spec, (
                "Reject can only be used if %s is provided" % self.ensure_nonempty
            )
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):
        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()
        if self.mask:
            assert self.mask in request, (
                "Reject can only be used if %s is provided" % self.mask
            )
        if self.ensure_nonempty:
            assert self.ensure_nonempty in request, (
                "Reject can only be used if %s is provided" % self.ensure_nonempty
            )

        have_good_batch = False
        while not have_good_batch:
            batch = self.upstream_provider.request_batch(request)

            if self.mask:
                mask_ratio = (batch.arrays[self.mask].data>0).mean()
            else:
                mask_ratio = None

            if self.ensure_nonempty:
                num_points = len(list(batch.graphs[self.ensure_nonempty].nodes))
            else:
                num_points = None

            have_min_mask = mask_ratio is None or mask_ratio > self.min_masked
            have_points = num_points is None or num_points > 0

            have_good_batch = have_min_mask and have_points

            if not have_good_batch and self.reject_probability < 1.0:
                have_good_batch = random.random() > self.reject_probability

            if not have_good_batch:
                if self.mask:
                    logger.debug(
                        "reject batch with mask ratio %f at %s",
                        mask_ratio,
                        batch.arrays[self.mask].spec.roi,
                    )
                if self.ensure_nonempty:
                    logger.debug(
                        "reject batch with empty points in %s",
                        batch.graphs[self.ensure_nonempty].spec.roi,
                    )
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:
                    logger.warning(
                        "rejected %d batches, been waiting for a good one " "since %ds",
                        num_rejected,
                        report_next_timeout,
                    )
                    report_next_timeout *= 2

            else:
                if self.mask:
                    logger.debug(
                        "accepted batch with mask ratio %f at %s",
                        mask_ratio,
                        batch.arrays[self.mask].spec.roi,
                    )
                if self.ensure_nonempty:
                    logger.debug(
                        "accepted batch with nonempty points in %s",
                        self.ensure_nonempty,
                    )

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

def train(
        iterations,
        samples,
        run_name,
        rej_prob=1.,
        augmentation=True,
    ):
    
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:'raw', #3d/raw',
                    labels:'labeled', #3d/labeled',
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                    labels:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
                }) +
                gp.Normalize(raw) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                gp.RandomLocation() +
                Reject(mask=labels, 
                    min_masked=0.001, 
                    reject_probability=rej_prob)
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    if augmentation:
        pipeline += gp.SimpleAugment(transpose_only=[1,2])

        pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

        pipeline += gp.ElasticAugment(
                        control_point_spacing=(32,)*3,
                        jitter_sigma=(2.,)*3,
                        rotation_interval=(0,math.pi/2),
                        scale_interval=(0.8, 1.2))

        pipeline += gp.NoiseAugment(raw, var=0.01)
    
    pipeline += gp.Stack(batch_size)

    
    pipeline += gp.Snapshot(
            output_filename="batch_{id}.zarr",
            output_dir="snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
            },
            every=1)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":
    base_dir = '../../../01_data/zarrs/train' 
    all_samples = glob.glob(base_dir + '/*.zarr')

    dt = str(datetime.now()).replace(':','-').replace(' ', '_')
    dt = dt[0:dt.rfind('.')]

    for rej_prob in [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1]:
    
        params = {'iterations': 100,
                'samples': all_samples,
                'run_name': dt + '_RejTest_min0.001_withAug' + '_rej' + str(rej_prob),
                'augmentation': True,
                'rej_prob': rej_prob,
                }
        train(**params) 
    # params['rej_prob_end'] = 0.8
    # params['iterations'] = 4001
    # train(**params) 
    
    #iterations=2001, samples=all_samples, run_name=run_name, rej_prob=0.9, device=0, lambda_gdl=15, lambda_mse=1)
