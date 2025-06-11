import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def compute_iou(gt_intervals, pred_intervals):
    # Combine GT and Pred to union intervals
    # Compute intersection and union lengths
    def merge_intervals(intervals):
        if not intervals:
            return []
        iv = sorted(intervals, key=lambda x: x[0])
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged
    gt = merge_intervals(gt_intervals)
    pr = merge_intervals(pred_intervals)
    # compute intersection length
    inter = 0.0
    for gs, ge in gt:
        for ps, pe in pr:
            inter += max(0.0, min(ge, pe) - max(gs, ps))
    # compute union length = sum(len(gt)) + sum(len(pr)) - inter
    gt_len = sum(e - s for s, e in gt)
    pr_len = sum(e - s for s, e in pr)
    union = gt_len + pr_len - inter
    return inter / union if union > 0 else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate intro detection using IoU')
    parser.add_argument('--labels', type=Path, required=True,
                        help='cleaned_labels.csv with ground truth intervals')
    parser.add_argument('--predictions', type=Path, required=True,
                        help='intro_segments.json with predicted intervals')
    args = parser.parse_args()

    # Load GT
    df_gt = pd.read_csv(args.labels)
    gt_dict = {}
    for _, row in df_gt.iterrows():
        ep = row['url']
        # extract episode id suffix
        epi = Path(ep).name if '/' in ep else ep
        # use entire URL string; for compatibility assume 'episode_id' field in JSON matches this
        if epi not in gt_dict:
            gt_dict[epi] = []
        gt_dict[epi].append([row['start'], row['end']])

    # Load predictions
    preds = json.loads(args.predictions.read_text())
    pred_dict = {}
    for entry in preds:
        epi = entry['episode_id']
        pred_dict.setdefault(epi, []).append([entry['start'], entry['end']])

    # Compute IoU per episode
    ious = []
    for epi, gt_ints in gt_dict.items():
        pr_ints = pred_dict.get(epi, [])
        iou = compute_iou(gt_ints, pr_ints)
        print(f'Episode {epi}: IoU = {iou:.3f}')
        ious.append(iou)

    if ious:
        mean_iou = np.mean(ious)
        print(f'\nMean IoU over {len(ious)} episodes: {mean_iou:.3f}')
    else:
        print('No matching episodes for evaluation.')
