import numpy as np
import shutil
import pandas as pd
import zarr
import time
import os

from skimage.measure import regionprops, regionprops_table
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

import sys
project_root = '/home/caylamiller/workspace/cochlea_synapses/'
sys.path.append(project_root)
from utils import calc_errors, greedy_match_predictions
from utils import sdt_to_labels, filt_labels_by_size, save_out

start_t = time.time()

val_dir = '/media/caylamiller/DataDrive/synapses_tscc/model_full_run_MSE1_GDL5_p85__cpu12_cx2_nwx20_MSE1.0GDL5.0' #model_full_run_MSE1_GDL5__cpu12_cx2_nwx20_MSE1.0GDL5.0'
#os.path.join(project_root,'03_predict/3d/model_full_run_MSE1_GDL5__cpu12_cx2_nwx20_MSE1.0GDL5.0_checkpoint_5000')
val_samples = [i for i in os.listdir(val_dir) if i.endswith('.zarr')]
gt_dir = os.path.join(project_root,'01_data/zarrs/validate')

msk_thresh_list = [0.0]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
pk_thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
strict_peak_thresh = True
save_pred_labels = False
save_csvs = True
size_filt_list = [1] #, 100, 150] #[1,2,3,4,5,7,10,15]
get_AUC = False

AllResults = pd.DataFrame()
AUCResults = pd.DataFrame()
count = 0

for fi in val_samples:

    preds = [i for i in os.listdir(os.path.join(val_dir,fi)) if i.startswith('pred')]
    print(fi, preds, flush=True)
    if len(preds) == 0:
        continue
    print("starting ", fi, " ...getting centroids...")
    img = zarr.open(os.path.join(val_dir, fi))
    gt_img = zarr.open(os.path.join(gt_dir, fi))
    gt = gt_img['labeled'][:]

    y_mask = np.zeros(gt.shape, dtype=int) 
    if '3526-L-04' in fi:
        print('4')
        y_mask[:, 0:350, :] = 1
        y_mask[:, 700::, :] = 1
    elif '3532-L-16' in fi:
        print('16')
        y_mask[:, 0:150, :] = 1
        y_mask[:, 650::, :] = 1
        y_mask[:, :, 0:30] = 1
        y_mask[:, :, 964::] = 1
    elif '6385-L-25' in fi:
        print('25')
        y_mask[:, 0:320, :] = 1
        y_mask[:, 720::, :] = 1
        y_mask[:, :, 0:60] = 1
        y_mask[:, :, 1000::] =1
    elif '6382-L-32' in fi:
        print('32')
        y_mask[:, 0:400, :] = 1
        y_mask[:, 700::, :] = 1

    gt[y_mask>0] = 0
    gt_xyz = regionprops_table(gt, properties=('centroid','label'))
    gt_xyz = np.asarray([gt_xyz['centroid-2'], gt_xyz['centroid-1'], gt_xyz['centroid-0']]).T
    gt_coords = [i.centroid for i in regionprops(gt)]
    
    for pred_str in preds:
        pred_coords = []
        pred_scores = []
        pred = img[pred_str][:]
        pred_res = img[pred_str].attrs['resolution']
        pred[y_mask>0] = -1
    
        for thresh in msk_thresh_list:
            for pk_thresh in pk_thresh_list:
                print("thresh: ", thresh, thresh+pk_thresh) #, " (", np.where(thresh==np.array(thresh_list))[0][0]+1, " out of ", len(thresh_list), ")")
                peak_thresh = thresh+pk_thresh
                mask_thresh = thresh

                segmentation = sdt_to_labels(
                    pred, 
                    peak_thresh=peak_thresh,
                    mask_thresh=mask_thresh,
                    strict_peak_thresh=strict_peak_thresh)
                
                intersection = np.logical_and(segmentation>0, gt>0).sum()
                union = np.logical_or(segmentation>0, gt>0).sum()
                iou = intersection / union if union != 0 else 0.0    
                dice = intersection * 2.0 / (np.sum(segmentation>0) + np.sum(gt>0)) if (np.sum(segmentation>0) + np.sum(gt>0)) != 0 else 0.0

                for size_thresh in size_filt_list:
                    if size_thresh>1:
                        print(size_thresh)
                        segmentation_filt = filt_labels_by_size(segmentation, size_filt=size_thresh)

                    else:
                        segmentation_filt = segmentation
                    if save_pred_labels:
                        print('saving labels...')
                        if os.path.exists(os.path.join(val_dir, fi, 'pred_labels')):
                            shutil.rmtree(os.path.join(val_dir,fi,'pred_labels'))

                        print('saving')
                        save_out(img, gt, 'gt_labels', save2d=False, res=pred_res)
                        save_out(img, segmentation_filt.astype(np.uint64), 'pred_labels', save2d=False, res=pred_res)

                    if save_csvs:
                        print('calculating stats...')
                        results= {"file": fi,
                                "checkpoint": int(pred_str.replace('pred_','')),
                                "mask_thresh": thresh,
                                "peak_thresh": thresh+pk_thresh,
                                "size_thresh": size_thresh,
                                "dice": dice,
                                "iou": iou,
                                }
                        results.update(
                            calc_errors(
                                segmentation_filt, 
                                gt_xyz, 
                                )
                        )
                        if get_AUC:
                            region_props = regionprops(segmentation_filt, intensity_image=pred)

                            for prop in region_props:
                                if prop.label == 0:
                                    continue
                                pred_coords.append(prop.centroid)  # highest distance transform value
                                pred_scores.append(prop.max_intensity)
                                # Match to GT centroids

                        print(results['f1'])
                        AllResults = pd.concat([AllResults, pd.DataFrame(data=results, index=[count])])
            count = count + 1
        if get_AUC:
            is_tp = greedy_match_predictions(pred_coords, pred_scores, gt_coords, dist_thresh=2.0)
            precision, recall, _ = precision_recall_curve(is_tp, pred_scores)
            ap = average_precision_score(is_tp, pred_scores)
            results= {"file": fi,
                    "checkpoint": int(pred_str.replace('pred_','')),
                    "recall_curve": [recall],
                    "precision_curve": [precision],
                    "ap": ap,
                    }
            AUCResults = pd.concat([AUCResults, pd.DataFrame(data=results, index=[count])])


if save_csvs:
    AllResults.to_csv(os.path.join(val_dir,'val_stats_msk0.csv'),encoding='utf-8-sig')
    if get_AUC:
        AUCResults.to_csv(os.path.join(val_dir,'val_auc_stats_strict2.csv'),encoding='utf-8-sig')
    #bestlist = list(AllResults.groupby('file').idxmax()['f1'])
    #AllResults.iloc[bestlist].to_csv(os.path.join(val_dir,'best_stats.csv'), encoding='utf-8-sig')

elapsed = time.time() - start_t
print("elapsed time: "+str(elapsed))
