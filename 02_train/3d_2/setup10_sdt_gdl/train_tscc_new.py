import glob
import gunpowder as gp
import logging
import math
import sys
import os
import torch
import argparse
from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
#from datetime import datetime

import time

project_root = '/tscc/nfs/home/cayla/synapses/'
sys.path.append(project_root)
from utils import NoiseAugmentRand, RejectWithinBatch, calc_max_padding, ComputeDT, BalanceLabelsWithIntensity, CheckPred #CheckSnaps #check_snaps

start_time = time.time()

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 10
ncpus = len(os.sched_getaffinity(0))

class CheckShape(gp.BatchFilter):
    def __init__(self, keys):
        self.keys = keys

    def process(self, batch, request):
        for key in self.keys:
            logger.info("%s shape: %s", key, batch[key].data.shape)

class WeightedGDLMSE(torch.nn.Module):
    def __init__(
            self, 
            weights=None, 
            alpha=2,
            lambda_gdl=1,
            lambda_mse=1,
            calc_z = True):
        super(WeightedGDLMSE, self).__init__()
        self.weights = weights
        self.alpha = alpha
        self.lambda_gdl = lambda_gdl
        self.lambda_mse = lambda_mse
        self.calc_z = calc_z

    def forward(self, pred, gt, weights=None):
        if weights is None:
            weights = self.weights
        if self.lambda_mse > 0:
            MSE = weights * (pred - gt) ** 2
            if len(torch.nonzero(MSE)) != 0:
                mask = torch.masked_select(MSE, torch.gt(weights, 0))
                MSE_loss = torch.mean(mask)
            else:
                MSE_loss = torch.mean(MSE)
        else:
            MSE_loss = 0.0
        if self.lambda_gdl > 0:
            grad_pred_y = torch.abs(pred[:, :, :,  1:, :] - pred[:, :, :, :-1, :])
            grad_gt_y   = torch.abs(gt[:, :, :, 1:, :] - gt[:, :, :, :-1, :])
            grad_diff_y = torch.abs(grad_pred_y - grad_gt_y) ** self.alpha

            grad_pred_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
            grad_gt_x = torch.abs(gt[:, :, :, :, 1:] - gt[:, :, :, :, :-1])
            grad_diff_x = torch.abs(grad_pred_x - grad_gt_x) ** self.alpha
            if self.calc_z:
                grad_pred_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
                grad_gt_z = torch.abs(gt[:, :, 1:, :, :] - gt[:, :, :-1, :, :])
                grad_diff_z = torch.abs(grad_pred_z - grad_gt_z) ** self.alpha

            if weights is not None:
                weight_y = weights[:,:,:,1:,:]
                weight_x = weights[:,:,:,:,1:]
                weight_z = weights[:,:,1:,:,:]
                grad_diff_y = grad_diff_y * weight_y
                grad_diff_x = grad_diff_x * weight_x
                if self.calc_z:
                    grad_diff_z = grad_diff_z * weight_z
            if self.calc_z:
                loss_gdl = torch.mean(grad_diff_x) + torch.mean(grad_diff_y) + torch.mean(grad_diff_z)
            else:
                loss_gdl = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
        else:
            loss_gdl = 0.0

        loss = self.lambda_mse*MSE_loss + self.lambda_gdl*loss_gdl

        return loss

class GradientDifferenceLoss(torch.nn.Module):
    def __init__(self, weights=None, alpha=2):
        super(GradientDifferenceLoss, self).__init__()
        self.weights = weights
        self.alpha = alpha  # Controls the exponent as in the paper

    def forward(self, pred, gt, weights=None): 
        """
        Computes the Gradient Difference Loss (GDL) as defined in https://arxiv.org/abs/1511.05440

        Args:
            pred (torch.Tensor): Predicted image tensor (B, C, Z, Y, X)
            gt   (torch.Tensor): Ground truth image tensor (B, C, Z, Y, X)

        Returns:
            torch.Tensor: GDL loss
        """
        if weights is None:
            weights = self.weights

        grad_pred_y = torch.abs(pred[:, :, :,  1:, :] - pred[:, :, :, :-1, :])
        grad_gt_y   = torch.abs(gt[:, :, :, 1:, :] - gt[:, :, :, :-1, :])
        grad_diff_y = torch.abs(grad_pred_y - grad_gt_y) ** self.alpha

        grad_pred_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        grad_gt_x = torch.abs(gt[:, :, :, :, 1:] - gt[:, :, :, :, :-1])
        grad_diff_x = torch.abs(grad_pred_x - grad_gt_x) ** self.alpha
        
        if weights is not None:
            weight_y = weights[:,:,:,1:,:]
            weight_x = weights[:,:,:,:,1:]
            grad_diff_y = grad_diff_y * weight_y
            grad_diff_x = grad_diff_x * weight_x

        loss_gdl = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
        return loss_gdl

def train(
        iterations,
        samples,
        run_name,
        rej_prob=1.,
        amsgrad=False,
        lr=5e-5,
        gt_dilate=1,
        wt_dilate=1,
        scale=2,
        lambda_gdl=0, 
        lambda_mse=1,
        rej_prob_every=100,
        rej_prob_end=0.8,
        rej_prob_step=0.01,
        workers=20, 
        cache_multiplier=2, 
    ):
    
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
    num_fmaps=12
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

    loss = WeightedGDLMSE(lambda_mse=lambda_mse, lambda_gdl=lambda_gdl) #GradientDifferenceLoss() #WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=lr, #5e-5
            params=model.parameters(),
            amsgrad=amsgrad,
            )

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
                gp.RandomLocation() + #mask=labels, min_masked=0.001) #+
                RejectWithinBatch(mask=labels, 
                    min_masked=0.005, #0.01, #0.005
                    min_min_masked=0.001, #0.001
                    reject_probability=rej_prob,
                    log_scale_steps=True,
                    mask_n_steps=100,
                    every=rej_prob_every,
                    end_reject_probability=rej_prob_end,
                    reject_probability_step=rej_prob_step)
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

    pipeline += NoiseAugmentRand(raw, var=0.01)

    pipeline += ComputeDT(
            labels,
            gt,
            mode='3d',
            dilate_iterations=gt_dilate, #1
            scale=scale,
            mask=surr_mask,
            )

    pipeline += BalanceLabelsWithIntensity(
            raw,
            surr_mask,
            mask_weights,
            dilate_iter=wt_dilate,
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
            cache_size=batch_size*cache_multiplier,
            num_workers=workers, #(ncpus-1)*worker_multiplier,
            )

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
        log_dir='/tscc/lustre/ddn/scratch/cayla/synapses_data/log/'+run_name,
        checkpoint_basename='/tscc/lustre/ddn/scratch/cayla/synapses_data/models/model_'+run_name,
        save_every=100)

    # raw: b,c,d,h,w
    # labels: b,c,d,h,w
    # pred_mask: b,c,d,h,w

    pipeline += gp.Squeeze([raw, gt, pred, mask_weights], axis=1)
    if batch_size == 1:
        pipeline += gp.Squeeze([raw, gt, pred, labels, mask_weights])
    #pipeline += CheckShape([raw, gt, pred, labels, mask_weights])
    # raw: c,d,h,w
    # labels: c,d,h,w

    #pipeline += gp.Squeeze([raw, labels, pred])

    # raw: d,h,w
    # labels: d,h,w
    # labels_mask: d,h,w
    #pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    
    pipeline += gp.Snapshot(
            output_filename="batch_{iteration}.zarr",
            output_dir="/tscc/lustre/ddn/scratch/cayla/synapses_data/snapshots/"+run_name,
            dataset_names={
                raw: "raw",
                labels: "labels",
                pred: "pred",
                gt: "gt",
                mask_weights: "mask_weights",
            },
            every=100)
    
    pipeline += CheckPred(pred, every=1)

    pipeline += gp.PrintProfilingStats(every=10) 

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)
            if i==0:
                first_time = time.time()
                print('step' , i, 'time', time.time()-start_time)
            else:
                print('step', i, 'time', time.time()-first_time)
if __name__ == "__main__":
    #base_dir = os.environ["TRAIN"] #
    
    base_dir = '/tscc/lustre/ddn/scratch/cayla/synapses_data/train' 
    all_samples = glob.glob(base_dir + '/*.zarr')

    #dt = str(datetime.now()).replace(':','-').replace(' ', '_')
    #dt = dt[0:dt.rfind('.')]

    parser = argparse.ArgumentParser(description="Train model with weighted GDL and MSE loss.")
    parser.add_argument("--lambda_gdl", "-g", type=float, default=1.0, help="Weight for GDL loss.")
    parser.add_argument("--lambda_mse", "-m", type=float, default=0.0, help="Weight for MSE loss.")
    parser.add_argument("--run_name", "-n", type=str, default='', help="anything to include in run name.")
    parser.add_argument("--cache_multiplier", "-c", type=int, default=2)
    parser.add_argument("--workers", "-w", type=int, default=20)

    args = parser.parse_args()
    
    if len(args.run_name)>0:
        args.run_name = args.run_name+'_'

    run_name = (args.run_name + 
            'cpu'+ str(ncpus) + '_cx' + str(args.cache_multiplier) + '_nwx' + str(args.workers)+ 
            '_MSE' + str(args.lambda_mse) + 'GDL' + str(args.lambda_gdl) + 
            #'_rej' + str(params['rej_prob'])
            '')

    params = {'iterations': 1001,
            'samples': all_samples,
            'run_name': run_name, 
            'rej_prob': 1,
            'lambda_gdl': args.lambda_gdl,
            'lambda_mse': args.lambda_mse,
            'rej_prob_every': 1, # in reality is ~ 1*400 because of precache
            'rej_prob_end': None, #0.8,
            'rej_prob_step': 0.01,
            'workers': args.workers,
            'cache_multiplier': args.cache_multiplier,
            }

    # run_name = args.run_name + 'MSE' + str(params['lambda_mse']) + 'GDL' + str(params['lambda_gdl']) + '_rej' + str(params['rej_prob'])
    # params['run_name'] = run_name
    print(run_name)

        #params['rej_prob_end'] = None
    #try:
    #train(**params) 

    #except:
    #    print('')
    params['rej_prob_end'] = 0.9
    train(**params)

    params['rej_prob_end'] = 0.8
    params['iterations'] = 3001
    train(**params) 

    print(time.time()-start_time)
