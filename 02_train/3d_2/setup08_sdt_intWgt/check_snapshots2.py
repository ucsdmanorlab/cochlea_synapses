import glob
import numpy as np
import zarr
import sys
import pandas as pd
from natsort import natsorted

if len(sys.argv)>1:
    snapdir = sys.argv[1]
else:
    snapdir = 'snapshots/2025-02-18*/'

files = glob.glob(snapdir+'batch_1001.zarr')
print(files)
df = pd.DataFrame([], columns=['data', 'lr', 'gt_dilate', 'wt_dilate', 'amsgrad', 'n_synapses', 'mean_pred_prob', 'range_pred_prob'])

for f in natsorted(files):
    labels = zarr.open(f)['labels'][:]
    pred = zarr.open(f)['pred'][:]
    a, b = np.unique(labels, return_counts=True)

    # pull lr, gt_dilate, wt_dilate from filename:
    lr = float(f.split('lr')[1].split('_')[0])
    wt_dilate = int(f.split('_wgt')[1].split('_')[0])
    gt_dilate = int(f.split('_gt')[1].split('_')[0])
    data = f.split('_lr')[0].split('IntWgt_')[1].replace('b1','').replace('_','')
    if len(data)==0:
        data = 'all'
    ams = f.split('ams')[1].split('/')[0]
    #if len(b)>0:
    print(data, lr, 'gt', gt_dilate, 'wt', wt_dilate, ams, len(b)-1, np.mean(pred), np.max(pred)-np.min(pred))
    
    # put results in a pandas dataframe:
    new_row = pd.DataFrame([{'data': data, 'lr': lr, 'gt_dilate': gt_dilate, 'wt_dilate': wt_dilate, 'amsgrad': ams, 'n_synapses': len(b)-1, 'mean_pred_prob': np.mean(pred), 'range_pred_prob': np.max(pred)-np.min(pred)}])
    df = pd.concat([df, new_row], ignore_index=True)

df.to_csv('results_2025-02-18.csv')
