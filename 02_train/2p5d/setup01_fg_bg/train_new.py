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
from torch.nn.functional import binary_cross_entropy
from scipy.ndimage import binary_dilation

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = '../../../01_data/zarrs/train'

samples = glob.glob(base_dir + '/*.zarr')
voxel_size = gp.Coordinate((1,)*3)

input_size = gp.Coordinate((5, 140, 140)) * voxel_size
output_size = gp.Coordinate((1, 100, 100)) * voxel_size

batch_size = np.uint64(2)

dt = str(datetime.now()).replace(':','-').replace(' ', '_')
dt = dt[0:dt.rfind('.')]

run_name = dt+'_2class2p5d_intWgt'

def calc_max_padding(
        output_size,
        voxel_size):

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = np.ceil(
            [i/2 + j for i,j in zip(
                [output_size[0], diag, diag],
                list(voxel_size)
                )]
            )
    max_padding[0] = 0
    print(max_padding)

    return gp.Coordinate(max_padding)

class WeightedBCELoss(torch.nn.BCELoss):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, prediction, target, weights):
        loss = binary_cross_entropy(prediction, target, weight=weights)
        return loss

class CreateMask(gp.BatchFilter):

    def __init__(
            self, 
            in_labels, 
            out_mask, 
            dtype=np.uint8, 
            dilate_pix=0,
            dilate_struct=None
            ):
        self.in_labels = in_labels
        self.out_mask = out_mask
        self.dtype = dtype
        self.dilate_pix = dilate_pix
        self.dilate_struct = dilate_struct

    def setup(self):

        self.provides(
            self.out_mask,
            self.spec[self.in_labels].copy())

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.in_labels] = request[self.out_mask].copy()

        return deps

    def process(self, batch, request):
        
        outputs = gp.Batch()

        labels_data = batch[self.in_labels].data
        mask_data = np.zeros_like(labels_data).astype(self.dtype)
        
        spec = batch[self.in_labels].spec.copy()
        spec.roi = request[self.out_mask].roi.copy()
        spec.dtype = self.dtype

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:
            mask_data = (labels_data>0).astype(self.dtype)
        
        if self.dilate_pix >0:
            mask_data = binary_dilation(
                    mask_data, 
                    structure = self.dilate_struct, 
                    iterations = self.dilate_pix).astype(self.dtype)

        outputs[self.out_mask] = gp.Array(mask_data, spec)

        return outputs

class BalanceLabelsWithIntensity(gp.BatchFilter):

    def __init__(
            self,
            raw,
            labels,
            scales,
            num_bins = 10,
            bin_min = 0.,
            bin_max = 1.,
            ):

        self.labels = labels
        self.raw = raw
        self.scales = scales
        self.bins = np.linspace(bin_min, bin_max, num=num_bins+1)

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
        #raw_cropped = self.__cropND(raw.data, cropped_roi/voxel_size)

        num_classes = len(self.bins) + 2

        binned = np.digitize(raw.data, self.bins) + 1
        binned = self.__cropND(binned, cropped_roi/voxel_size)
        # add one so values fall from 1 to num_bins+1 (if outside range 0-1)

        # set foreground labels to be class 0:
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
        out_dims = bounding
        while img.ndim > len(bounding):
            bounding = np.append(1, bounding)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices].reshape(out_dims)

def train(iterations):

    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    # labels_mask = gp.ArrayKey("LABELS_MASK")
    # gt = gp.ArrayKey("GT")
    # pred = gp.ArrayKey("PRED")
    # mask_weights = gp.ArrayKey("MASK_WEIGHTS")
     
    request = gp.BatchRequest()
    request.add(raw, gp.Coordinate(5, 140, 140))
    request.add(labels, gp.Coordinate(1, 100, 100))
    # request.add(labels_mask, output_size)
    # request.add(gt, output_size)
    # request.add(pred, output_size)
    # request.add(mask_weights, output_size)

    num_fmaps=30

    ds_fact = [(2,2), (2,2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3,3),(3,3)]]*num_levels
    ksu = [[(3,3),(3,3)]]*(num_levels-1)
    in_channels = 5
    out_channels = 1

    unet = UNet(
        in_channels=in_channels,
        #out_channels=1,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu)

    model = torch.nn.Sequential(
            unet,
            ConvPass(num_fmaps, 1, [[1,]*2], activation=None),
            torch.nn.Sigmoid()
            )

    #torch.cuda.set_device(1)

    loss = torch.nn.BCELoss() #WeightedBCELoss() 
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    padding = calc_max_padding(output_size, voxel_size)

    sources = tuple(
            gp.ZarrSource(
                sample,
                datasets={
                    raw:f'3d/raw',
                    labels:f'3d/labeled',
                    },
                array_specs={
                    raw:gp.ArraySpec(interpolatable=True, voxel_size=gp.Coordinate(1, 1,1)),
                    labels:gp.ArraySpec(interpolatable=False, voxel_size=gp.Coordinate(1, 1,1)),
                }) +
                # gp.Pad(raw, None) +
                # gp.Pad(labels, padding) + 
                # gp.Normalize(raw) +
                gp.RandomLocation() + 
                gp.Reject_CMM(mask=labels, min_masked=0.005, min_min_masked=0.001, reject_probability=0.99) 
                for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider()

    # pipeline += gp.SimpleAugment(mirror_only=[1,2], transpose_only=[1,2])

    # pipeline += gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2)

    # pipeline += gp.ElasticAugment(
    #                 control_point_spacing=(32,)*2,
    #                 jitter_sigma=(2.,)*2,
    #                 rotation_interval=(0,math.pi/2),
    #                 scale_interval=(0.8, 1.2), 
    #                 spatial_dims=2)
    
    # pipeline += gp.NoiseAugment(raw, var=0.01)
    
    # pipeline += CreateMask(
    #         labels,
    #         gt,
    #         dtype = np.float32)

    # pipeline += BalanceLabelsWithIntensity(
    #         raw,
    #         gt,
    #         mask_weights,
    #         )

    # # raw: h,w
    # # labels: h,w
    # # labels_mask: h,w

    # pipeline += gp.Unsqueeze([gt, mask_weights]) #, gt_fg])

    # # raw: c,h,w
    # # labels: c,h,w
    # # labels_mask: c,h,w

    # pipeline += gp.Stack(batch_size)

    # # raw: b,c,h,w
    # # labels: b,c,h,w
    # # labels_mask: b,c,h,w

    # pipeline += gp.PreCache(
    #         cache_size=40,
    #         num_workers=10)

    # pipeline += Train(
    #     model,
    #     loss,
    #     optimizer,
    #     inputs={
    #         'input': raw
    #     },
    #     outputs={
    #         0: pred
    #     },
    #     loss_inputs={
    #         0: pred,
    #         1: gt,
    #         2: mask_weights
    #     },
    #     log_dir='log/'+run_name,
    #     checkpoint_basename='model_'+run_name,
    #     save_every=500)

    # # raw: b,c,h,w
    # # labels: b,c,h,w
    # # labels_mask: b,c,h,w
    # # pred_mask: b,c,h,w
    
    # pipeline += gp.Squeeze([gt, pred, mask_weights], axis=1)

    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir = 'snapshots/'+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
    #             #labels_mask: "labels_mask",
    #             gt: "gt", 
    #             pred: "pred",
    #             mask_weights: "mask_weights",
            },
            every=1)
    
    # pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)
            print(batch[raw].spec.dtype)
            print(batch[labels].spec.dtype)

if __name__ == "__main__":

    train(10)
