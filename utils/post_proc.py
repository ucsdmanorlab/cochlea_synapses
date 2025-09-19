import numpy as np
from scipy.spatial import distance_matrix

def greedy_match_predictions(pred_coords, pred_scores, gt_coords, dist_thresh=2.0):
    """
    pred_coords: (N_pred, Ndim) array of predicted object centroids
    pred_scores: (N_pred,) array of confidence scores (higher is better)
    gt_coords:   (N_gt, Ndim) array of GT centroids
    dist_thresh: maximum distance for a valid match

    Returns:
        matches: list of (pred_idx, gt_idx, distance)
        unmatched_preds: list of pred_idx
        unmatched_gts: list of gt_idx
    """
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return [], list(range(len(pred_coords))), list(range(len(gt_coords)))

    dists = distance_matrix(pred_coords, gt_coords)

    # Sort predictions by confidence (descending)
    pred_order = np.argsort(-np.array(pred_scores)) 

    matched_gt = set()
    matches = []

    for pred_idx in pred_order:
        # Get closest unmatched GT
        gt_candidates = [
            (gt_idx, dists[pred_idx, gt_idx]) 
            for gt_idx in range(len(gt_coords)) if gt_idx not in matched_gt
        ]
        if not gt_candidates:
            continue
        # Find nearest one
        gt_idx, dist = min(gt_candidates, key=lambda x: x[1])
        if dist <= dist_thresh:
            matches.append(pred_idx) #, gt_idx, dist))
            matched_gt.add(gt_idx)

    #matched_pred = set([p for p, _, _ in matches])
    #unmatched_preds = [i for i in range(len(pred_coords)) if i not in matched_pred]
    #unmatched_gts = [i for i in range(len(gt_coords)) if i not in matched_gt]

    y = [1 if i in matches else 0 for i in range(len(pred_scores))]
    
    return y #matches, unmatched_preds, unmatched_gts

def calc_errors(labels, gt_xyz, return_img=False):

    gtp = gt_xyz.shape[0]
    Npoints_pred = len(np.unique(labels))-1

    pred_idcs = np.zeros(gtp).astype(np.int32)

    for i in range(gtp):

        cent = np.flip(np.around(gt_xyz[i,:]).astype(np.int16))
        pred_idcs[i] = labels[tuple(cent)]
        #if mode=='2d':
        #    pred_idcs[i] = labels[cent[1], cent[0]]
        #elif mode=='3d':
        #    pred_idcs[i] = labels[cent[2], cent[1], cent[0]]

    idc, counts = np.unique(pred_idcs[pred_idcs>0], return_counts=True)

    tp = len(idc) #np.sum(counts==1)
    fm = np.sum(counts-1)
    fn = np.sum(pred_idcs==0)
    fp = Npoints_pred - len(idc)   
    
    ap = tp/(tp+fn+fp+fm)
    f1 = 2*tp/(2*tp + fp + fn + fm)
    if tp+fp == 0:
        prec = 1
    else:
        prec = tp/(tp+fp)
    rec  = tp/gtp

    results = {
        "gtp": gtp, # ground truth positives
        "tp": tp,   # true positives
        "fn": fn,   # false negatives 
        "fm": fm,   # false merges (2 ground truth positives IDed as 1 spot)
        "fp": fp,   # false potives
        "ap": ap,   # average precision
        "f1": f1,   # f1 score 
        "precision": prec, # precision
        "recall": rec,     # recall
        "mergerate": fm/gtp, # merge rate 
    }
    out = results
    if return_img:
        # 1 = true positive
        # 2 = false positive
        # 3 = false merge
        # 4 = false negative

        #status = np.ones(Npoints_pred, dtype=int)*2 #default is false positive (2)
        labelz = np.unique(labels)[1::]
        
        count = 0
        img_out = np.zeros(labels.shape, dtype=int)

        for id in labelz:
            if id in idc:
                if counts[idc==id]>1:
                    # false merge
                    img_out[labels==id] = 3
                else:
                    # true positive
                    img_out[labels==id] = 1
            else:
                # false positive
                img_out[labels==id] = 2

        # false negatives:
        for i in range(gtp):
            cent = np.flip(np.around(gt_xyz[i,:]).astype(np.int16))
            pred = labels[tuple(cent)]
            if pred == 0:
                img_out[tuple(cent)] = 4
        
        out = (results, img_out)

    return out 


