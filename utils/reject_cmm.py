import logging
import numpy as np
import random

from gunpowder.nodes import BatchFilter
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)


class Reject_CMM(BatchFilter):
    '''Reject batches based on the masked-in vs. masked-out ratio.

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
    '''

    def __init__(
            self,
            mask=None,
            min_masked=0.5,
            min_min_masked=None,
            ensure_nonempty=None,
            reject_probability=1.,
            mask_n_steps=None,
            log_scale_steps=False,
            every=100,
            end_reject_probability=None,
            reject_probability_step=0.05,
            ):

        self.mask = mask
        self.start_min_masked = min_masked
        self.min_masked = min_masked
        self.ensure_nonempty = ensure_nonempty
        self.reject_probability = reject_probability
        self.min_min_masked = min_min_masked
        self.mask_n_steps = mask_n_steps
        self.log_scale_steps = log_scale_steps
        self.every = max(1, every)
        self.n = 0
        self.end_reject_probability = end_reject_probability
        self.reject_probability_step = reject_probability_step

    def setup(self):
        if self.mask:
            assert self.mask in self.spec, (
                "Reject can only be used if %s is provided" % self.mask)
        if self.ensure_nonempty:
            assert self.ensure_nonempty in self.spec, (
                "Reject can only be used if %s is provided" %
                self.ensure_nonempty)
        self.upstream_provider = self.get_upstream_provider()

        if self.min_min_masked is not None:
            assert self.min_min_masked < self.min_masked, (
                    "min_min_masked must be less than min_masked!")
            if self.mask_n_steps is not None:
                mask_n_steps = self.mask_n_steps
            else:
                mask_n_steps = 10

            if self.log_scale_steps:
                self.mask_steps = np.geomspace(self.min_masked, self.min_min_masked, num=(mask_n_steps+1))
            else:
                self.mask_steps = np.linspace(self.min_masked, self.min_min_masked, num=(mask_n_steps+1))
                #mask_range = self.min_masked-self.min_min_masked
                #self.mask_step_size = mask_range/mask_n_steps
        

    def provide(self, request):
        random.seed(request.random_seed)

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()
        self.n += 1
        if self.n % self.every == 0 and self.end_reject_probability is not None:
            self.reject_probability = self.reject_probability - self.reject_probability_step
            logger.info( "changing reject to %f", self.reject_probability)
            if self.reject_probability < self.end_reject_probability:
                self.reject_probability = self.end_reject_probability
                self.end_reject_probability = None

        if self.mask:
            assert self.mask in request, (
                "Reject can only be used if %s is provided" % self.mask)
        if self.ensure_nonempty:
            assert self.ensure_nonempty in request, (
                "Reject can only be used if %s is provided" %
                self.ensure_nonempty)

        have_good_batch = False
        i=0

        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)

            if self.mask:
                mask_ratio = (batch.arrays[self.mask].data>0).mean()
            else:
                mask_ratio = None

            if self.ensure_nonempty:
                num_points = len(
                    list(batch.points[self.ensure_nonempty].nodes))
            else:
                num_points = None

            have_min_mask = mask_ratio is None or mask_ratio > self.min_masked
            have_points = num_points is None or num_points > 0

            have_good_batch = have_min_mask and have_points

            if not have_good_batch and self.reject_probability < 1.:
                have_good_batch = random.random() > self.reject_probability

            if not have_good_batch:
                if self.mask:
                    logger.debug(
                        "reject batch with mask ratio %f at %s",
                        mask_ratio, batch.arrays[self.mask].spec.roi)
                if self.ensure_nonempty:
                    logger.debug(
                        "reject batch with empty points in %s",
                        batch.points[self.ensure_nonempty].spec.roi)
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning(
                        "rejected %d batches, been waiting for a good one "
                        "since %ds", num_rejected, report_next_timeout)
                    report_next_timeout *= 2
                if self.min_min_masked:
                    if self.min_masked > self.min_min_masked:
                        self.min_masked = self.mask_steps[i+1]
                        i = i+1
                        #self.min_masked = self.min_masked - self.mask_step_size
                        
                        #if self.min_masked < self.min_min_masked:
                        #    self.min_masked = self.min_min_masked

                        logger.debug(
                                "reducing min_masked to %f", self.min_masked)

            else:
                if self.mask:
                    logger.debug(
                        "accepted batch with mask ratio %f at %s",
                        mask_ratio, batch.arrays[self.mask].spec.roi)
                if self.ensure_nonempty:
                    logger.debug(
                        "accepted batch with nonempty points in %s",
                        self.ensure_nonempty)

        timing.stop()
        batch.profiling_stats.add(timing)
        return batch