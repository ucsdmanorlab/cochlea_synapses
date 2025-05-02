import glob
import gunpowder as gp
import logging
import math
import numpy as np
import sys
import torch
from gunpowder.torch import Train
from funlib.learn.torch.models import UNet, ConvPass
from datetime import datetime


project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import WeightedMSELoss, calc_max_padding, ComputeDT, BalanceLabelsWithIntensity, CheckPred #CheckSnaps #check_snaps

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

input_shape = gp.Coordinate((44,172,172))
output_shape = gp.Coordinate((24,80,80))

voxel_size = gp.Coordinate((4,1,1))

input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

batch_size = 1

def train(
        iterations,
        samples,
        run_name,
        rej_prob=1.,
        amsgrad=False,
        lr=5e-5,
        gt_dilate=1,
        wt_dilate=1,
        device=0,
        scale=2,
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
    torch.cuda.set_device(device)

    loss = WeightedMSELoss()
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
                gp.RandomLocation() +
                gp.Reject_CMM(mask=labels, 
                    min_masked=0.005, #0.01, #0.005
                    min_min_masked=0.001, #0.001
                    reject_probability=rej_prob,
                    log_scale_steps=True,
                    mask_n_steps=100)
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
    
    pipeline += CheckPred(pred, every=1)

    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

if __name__ == "__main__":
    base_dir = '../../../01_data/zarrs/train' 
    orig_samples = glob.glob(base_dir + '/airyscan*.zarr')
    orig_samples += glob.glob(base_dir + '/spinning*.zarr')
    orig_samples += glob.glob(base_dir + '/confocal*.zarr')
    all_samples = glob.glob(base_dir + '/*.zarr')
    cil_samples = orig_samples + glob.glob(base_dir + '/3*kHz.zarr')
    lib_samples = orig_samples + glob.glob(base_dir + '/WPZ*.zarr')

    dt = str(datetime.now()).replace(':','-').replace(' ', '_')
    dt = dt[0:dt.rfind('.')]
    run_name = dt+'_3d_sdt_IntWgt_b1_withcil' 

    # Define the parameter grids
    learning_rates = [2e-5] #[5e-5, 1e-5]
    gt_dilates = [4, 3, 2, 1] #[0, 1, 2]
    wt_dilates = [1] #, 1]
    amsgrads = [False] #, True]
    scale = 1
    device = 0

    dt = str(datetime.now()).replace(':','-').replace(' ', '_')
    dt = dt[0:dt.rfind('.')]
    log_file = open(dt+"run_log.txt", "w")
    
    log_file.write("run name, samples, iterations, reject_prob, device, scale, lr, gt_dil, wgt_dil, amsgrad, exit_status\n")

    # Base parameters
    base_params = {
        'iterations': 1001,
        'samples': "all",
        'run_name': run_name,
        'rej_prob': 1.,
        'device': device,
        'scale': scale, 
    }
    repeats = 10
    samp_str = ""
    # Iterate over the parameter grids
    for _ in range(repeats):
        dt = str(datetime.now()).replace(':','-').replace(' ', '_')
        dt = dt[0:dt.rfind('.')]
        run_name = dt+'_3d_sdt_IntWgt_b1_withall' 
        for lr in learning_rates:
            for gt_dilate in gt_dilates:
                for wt_dilate in wt_dilates:
                    for amsgrad in amsgrads:
                        # Update parameters
                        full_run_name = f"{run_name}_lr{lr}_gt{gt_dilate}_wgt{wt_dilate}_ams{amsgrad}_sc{scale}_d{device}"
                        #full_run_name = "2025-02-18_11-59-32_3d_sdt_IntWgt_b1_withall_lr2e-05_gt4_wgt1_amsFalse_sc2_dev0"
                        train_params = base_params.copy()
                        train_params['lr'] = lr
                        train_params['gt_dilate'] = gt_dilate
                        train_params['wt_dilate'] = gt_dilate + wt_dilate
                        train_params['amsgrad'] = amsgrad
                        train_params['run_name'] = full_run_name
                        samp_str = train_params['samples']
                        if samp_str == "all":
                            train_params['samples'] = all_samples
                        elif samp_str == "cil":
                            train_params['samples'] = cil_samples
                        elif samp_str == "lib":
                            train_params['samples'] = lib_samples
                        else:
                            train_params['samples'] = orig_samples 
                        # Train the model with the current parameters
                        try:
                            train(**train_params)
                            log_file.write(f"{full_run_name}, {samp_str}, {train_params['iterations']}, 
                                           {train_params['reject_prob']}, {train_params['device']}, 
                                           {train_params['scale']}, {train_params['lr']}, 
                                           {train_params['gt_dilate']}, {train_params['wt_dilate']}, 
                                           {train_params['amsgrad']}, OK\n")
                        except Exception as e:
                            logging.error(f"Error during training with parameters {train_params}: {e}")
                            log_file.write(f"{full_run_name}, {samp_str}, {train_params['iterations']}, 
                                           {train_params['reject_prob']}, {train_params['device']}, 
                                           {train_params['scale']}, {train_params['lr']}, 
                                           {train_params['gt_dilate']}, {train_params['wt_dilate']}, 
                                           {train_params['amsgrad']}, {e}\n")
                            continue

    log_file.close()