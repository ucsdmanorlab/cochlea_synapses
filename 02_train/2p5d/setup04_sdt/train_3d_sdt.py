import glob
import gunpowder as gp
import logging
import math
import numpy as np
import operator

import torch
from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

base_dir = '../../../01_data/zarrs/train' #cilcare'

samples = glob.glob(base_dir + '/*.zarr')

input_shape = gp.Coordinate((5,140,140))
output_shape = gp.Coordinate((1,100,100))

voxel_size = gp.Coordinate((1,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 64

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]
run_name = dt+'_2p5d_sdt_IntWgt'

class WeightedMSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):
        
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss


def calc_max_padding(
        output_size,
        voxel_size,
        mode='shrink'):

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = gp.Roi(
            (gp.Coordinate(
                [i/2 for i in [output_size[0], diag, diag]]) +
                voxel_size),
            (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin() 

class ComputeDT(gp.BatchFilter):

    def __init__(
            self,
            labels,
            sdt,
            constant=0.5,
            dtype=np.float32,
            mode='3d',
            dilate_iterations=None,
            scale=None,
            mask=None,
            labels_mask=None,
            unlabelled=None):

        self.labels = labels
        self.sdt = sdt
        self.constant = constant
        self.dtype = dtype
        self.mode = mode
        self.dilate_iterations = dilate_iterations
        self.scale = scale
        self.mask = mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled

    def setup(self):

        spec = self.spec[self.labels].copy()

        self.provides(self.sdt,spec)

        if self.mask:
            self.provides(self.mask, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.labels] = request[self.sdt].copy()

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.labels].copy()

        if self.unlabelled:
            deps[self.unlabelled] = deps[self.labels].copy()

        return deps

    def _compute_dt(self, data):
        dist_func = distance_transform_edt

        if self.dilate_iterations:
            data = binary_dilation(
                    data,
                    iterations=self.dilate_iterations)

        if self.scale:
            inner = dist_func(binary_erosion(data))
            outer = dist_func(np.logical_not(data))

            distance = (inner - outer) + self.constant

            distance = np.tanh(distance / self.scale)

        else:

            inner = dist_func(data) - self.constant
            outer = -(dist_func(1-np.logical_not(data)) - self.constant)

            distance = np.where(data, inner, outer)

        return distance.astype(self.dtype)

    def process(self, batch, request):

        outputs = gp.Batch()

        labels_data = batch[self.labels].data
        distance = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.sdt].roi.copy()
        spec.dtype = np.float32

        labels_data = labels_data != 0

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:

            if self.mode == '3d':
                distance = self._compute_dt(labels_data)

            elif self.mode == '2d':
                for z in range(labels_data.shape[0]):
                    distance[z] = self._compute_dt(labels_data[z])
            else:
                raise ValueError('Only implemented for 2d or 3d labels')
                return

        if self.mask and self.mask in request:

            if self.labels_mask:
                mask = batch[self.labels_mask].data
            else:
                mask = (distance>0).astype(self.dtype) #(labels_data!=0).astype(self.dtype)

            if self.unlabelled:
                unlabelled_mask = batch[self.unlabelled].data
                mask *= unlabelled_mask

            outputs[self.mask] = gp.Array(
                    mask.astype(self.dtype),
                    spec)

        outputs[self.sdt] =  gp.Array(distance, spec)

        return outputs

class NormalizeDT(gp.BatchFilter):

    # use if computing dt without scaling (eg not sdt)

    def __init__(self, distance):
        self.distance = distance

    def normalize(self, data):

        return (data - data.min()) / (data.max() - data.min())

    def process(self, batch, request):

        data = batch[self.distance].data

        # don't normalize zero batches
        if len(np.unique(data)) > 1:
            data = self.normalize(data)

        batch[self.distance].data = data

class BalanceLabelsWithIntensity(gp.BatchFilter):

    def __init__(
            self,
            raw,
            labels,
            scales,
            num_bins = 10,
            bin_min = 0.,
            bin_max = 1.,
            dilate_iter = None,
            ):

        self.labels = labels
        self.raw = raw
        self.scales = scales
        self.bins = np.linspace(bin_min, bin_max, num=num_bins+1)
        self.dilate_iter = dilate_iter

    def setup(self):

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.labels] = request[self.scales]
        deps[self.raw] = request[self.raw]

        return deps

    def process(self, batch, request):

        labels = batch.arrays[self.labels]

        raw = batch.arrays[self.raw]

        cropped_roi = labels.spec.roi.get_shape()
        vox_size = labels.spec.voxel_size
        raw_cropped = self.__cropND(raw.data, cropped_roi/voxel_size)

        #raw_cropped = np.expand_dims(raw_cropped, 0)
        #raw_cropped = np.repeat(raw_cropped, 3, 0)

        num_classes = len(self.bins) + 2

        binned = np.digitize(raw_cropped, self.bins) + 1
        # add one so values fall from 1 to num_bins+1 (if outside range 0-1)

        # set foreground labels to be class 0:
        #print(labels.shape, binned.shape)
        if self.dilate_iter is not None:
            binned[binary_dilation(labels.data>0, iterations=self.dilate_iter)] = 0
        else:
            binned[labels.data>0] = 0

        # initialize scales:
        scale_data = np.ones(labels.data.shape, dtype=np.float32)

        # calculate the fraction of per-class samples:
        classes, counts = np.unique(binned, return_counts=True)
        fracs = counts.astype(float) / scale_data.sum()

        # prevent background from being weighted over foreground:
        fg = np.where(classes==0)[0]
        if len(fg)>0:
            fracs = np.clip(fracs, fracs[fg], None)

        # calculate the class weights
        w_sparse = 1.0 / float(num_classes) / fracs
        w = np.zeros(num_classes)
        w[classes] = w_sparse

        # scale the scale with the class weights
        scale_data *= np.take(w, binned)

        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi
        batch.arrays[self.scales] = gp.Array(scale_data, spec)

    def __cropND(self, img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]

def train(iterations,
        rej_prob=1.):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    pred = gp.ArrayKey("PRED")
    gt = gp.ArrayKey("GT")
    surr_mask = gp.ArrayKey("SURR_MASK")
    mask_weights = gp.ArrayKey("MASK_WEIGHTS")

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(pred, output_size)
    request.add(gt, output_size)
    request.add(mask_weights, output_size)
    request.add(surr_mask, output_size)

    in_channels = 1
    num_fmaps = 30
    num_fmaps_out = 14
    fmap_inc_factor = 5
    downsample_factors = [(1,2,2),(1,2,2),(2,2,2)]

    kernel_size_down = [
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3],
                [(1,3,3), (1,3,3)]]

    kernel_size_up = [
                [(1,3,3), (1,3,3)],
                [(3,)*3, (3,)*3],
                [(3,)*3, (3,)*3]]

    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors, 
        kernel_size_down=kernel_size_down, 
        kernel_size_up=kernel_size_up,
        constant_upsample=True)
    
    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*3], activation='Tanh'))
    # ct = 0
    # for child in unet.children():
    #     ct += 1
    #     print('child '+str(ct))
    #     print(child)
    torch.cuda.set_device(1)

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=5e-5, #5e-6, #5e-5
            params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:'3d/raw',
                    labels:'3d/labeled',
                #    labels_mask:'3d/mask'
                },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                    labels:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
                #    labels_mask:gp.ArraySpec(interpolatable=False, voxel_size=voxel_size)
                }) +
                gp.Normalize(raw) +
                gp.Pad(raw, None) +
                gp.Pad(labels, padding) +
                #gp.Pad(labels_mask, padding) +
                gp.RandomLocation() +
                gp.Reject_CMM(mask=labels, 
                    min_masked=0.005, 
                    min_min_masked=0.001,
                    reject_probability=rej_prob)
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    pipeline += gp.ElasticAugment(
                    control_point_spacing=(32,)*3,
                    jitter_sigma=(2.,)*3,
                    rotation_interval=(0,math.pi/2),
                    scale_interval=(0.8, 1.2))

    pipeline += gp.NoiseAugment(raw, var=0.01)

    pipeline += ComputeDT(
            labels,
            gt,
            mode='3d',
            dilate_iterations=None, #1
            scale=1,
            mask=surr_mask,
            )

    pipeline += BalanceLabelsWithIntensity(
            raw,
            surr_mask,
            mask_weights,
            dilate_iter=2,
            )
    
    # raw: d,h,w
    # labels: d,h,w

    #pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw, gt, mask_weights])

    # raw: c,d,h,w
    # labels: c,d,h,w
    
    pipeline += gp.Stack(batch_size)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w

    pipeline += gp.PreCache(
            cache_size=40,
            num_workers=10)

    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred
        },
        loss_inputs={
            0: pred,
            1: gt,
            2: mask_weights
        },
        log_dir='log/'+run_name,
        checkpoint_basename='model_'+run_name,
        save_every=1000)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w
    # pred_mask: b,c,d,h,w

    pipeline += gp.Squeeze([raw, labels, mask_weights])

    # raw: c,d,h,w
    # labels: c,d,h,w

    #pipeline += gp.Squeeze([raw, labels, pred])

    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w
    #pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    
    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir="snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                pred: "pred",
                gt: "gt",
                mask_weights: "mask_weights",
            },
            every=250)
    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":

    train(1001, rej_prob=1)
    #train(9001, rej_prob=0.9)
