import numpy as np
import skimage

from gunpowder.batch_request import BatchRequest

from gunpowder import BatchFilter


class NoiseAugmentRand(BatchFilter):
    """Add random noise to an array. Uses the scikit-image function skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by clipping values in the end, see
            scikit-image documentation
    """

    def __init__(self, array, mode="gaussian", clip=True, **kwargs):
        self.array = array
        self.mode = mode
        self.clip = clip
        self.kwargs = kwargs
        self.origkwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Noise augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= -1 and raw.data.max() <= 1
            ), "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."

        if 'var' in self.origkwargs:
            #randomize noise in each process pass:
            self.kwargs['var'] = np.random.rand()*self.origkwargs.get('var')

        seed = request.random_seed

        try:
            raw.data = skimage.util.random_noise(
                raw.data, mode=self.mode, rng=seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)

        except ValueError:
            # legacy version of skimage random_noise
            raw.data = skimage.util.random_noise(
                raw.data, mode=self.mode, seed=seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)
