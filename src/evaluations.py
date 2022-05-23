import numpy as np
import cv2
import os
import json
from tqdm import tqdm


def compute_precision_recall_and_matching_score(gt: np.array, pred: np.array,
                                                prob: np.array, thresh: float, pt=False):
    """Given the groundtruth, prediction, probability arrays and the probability threshold that is
    used to count a prediction as a true match, returns the precision, recall and matching score of
    the prediction. Assumes that keypoints without matches have gt = -1."""
    # Mask out all predictions with prob < thresh
    mask = np.where(prob < thresh)
    masked_pred = np.array(pred)
    masked_pred[mask] = -1
    if pt:
        print(f'gt: {gt}')
        print(f'masked_pred: {masked_pred}')

    total_positive = np.count_nonzero(gt != -1)
    tp = np.count_nonzero(np.logical_and(gt != -1, masked_pred == gt))
    fp = np.count_nonzero(np.logical_and(masked_pred != -1, masked_pred != gt))
    if pt:
        print(f'total positive: {total_positive}, tp: {tp}, fp: {fp}')


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = tp / total_positive if total_positive > 0 else 0.
    matching_score = tp / len(
        gt)  # num_correct / num_kps according to SuperGlue's implementation

    if pt:
        print(f'total positive: {total_positive}, tp: {tp}, fp: {fp}')
        print(f'precision: {precision}, recall: {recall}, matching_score: {matching_score}')
    return {
        'precision': precision,
        'recall': recall,
        'matching_score': matching_score,
        'tp': tp,
        'fp': fp
    }

def evaluate_handcrafted_matching_results(folder: str, img_pairs: str, gt_json: str, desc_type='desc_norm'):
    #TODO: remove pairs that come from the SAME SSS image!!!!!!
    with open(img_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    with open(gt_json, 'r') as f:
        gts = json.load(f)

    matcher = cv2.BFMatcher()
    thresholds = np.arange(0, 1.1, .1)
    precisions = []
    recalls = []
    matching_scores = []

    for p1, p2 in tqdm(pairs):
        patch1, patch2 = f'{p1.split("_")[0]}', f'{p2.split("_")[0]}'
        patch1_feat, patch2_feat = np.load(os.path.join(folder, f'{patch1}.npz')), np.load(os.path.join(folder,
                f'{patch2}.npz'))
        gt = np.array(gts[patch1[5:]][patch2[5:]])

        matches = matcher.knnMatch(patch1_feat[desc_type], patch2_feat[desc_type], k=2)
        pred = np.array([m[0].trainIdx for m in matches])
        # Use 1 - m/n as a score for how good the match is, m = L2 dist for best match, n = L2
        # dist for second best match, i.e. m << n -> very distinct match -> high score
        score = np.array([1 - m[0].distance/m[1].distance for m in matches])

        precision = []
        recall = []
        matching_score = []
        tp = []
        fp = []
        
        pt = False
        for thresh in thresholds:
            if len(recall) > 0 and recall[-1] > .5:
                # pt = True
                print(f'{patch1}, {patch2}')
            res = compute_precision_recall_and_matching_score(gt=gt, pred=pred, prob=score,
                    thresh=thresh, pt=pt)
            precision.append(res['precision'])
            recall.append(res['recall'])
            matching_score.append(res['matching_score'])
            tp.append(res['tp'])
            fp.append(res['fp'])
        np.savez(f'{folder}/{patch1}_{patch2}.npz', precision=precision, recall=recall,
                matching_score=matching_score, pred=pred, prob=score, gt=gt, tp=tp, fp=fp)
        print(f'{patch1}, {patch2}: gt={np.count_nonzero(gt!=-1)}, tp={tp[0]}, fp={fp[0]}')

        precisions.append(precision)
        recalls.append(recall)
        matching_scores.append(matching_score)

    prec_at_thres = 100.*np.array(precisions).mean(axis=0)
    recall_at_thresh = 100.*np.array(recalls).mean(axis=0)
    ms_at_thresh = 100.*np.array(matching_scores).mean(axis=0)

    print(f'Evaluation Results (mean over {len(pairs)} pairs')
    print(f'thresholds: {thresholds}')
    print(f'average precision: {prec_at_thres}')
    print(f'average recall: {recall_at_thresh}')
    print(f'average matching_score: {ms_at_thresh}')
    return precisions, recalls, matching_scores
