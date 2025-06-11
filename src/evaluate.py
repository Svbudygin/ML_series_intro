import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def compute_iou(gt_intervals, pred_intervals):
    """
    Compute Intersection over Union (IoU) for two sets of intervals.
    Returns (iou, intersection_length, union_length).
    Supports intervals as [start,end] or {'start':..., 'end':...}.
    """
    def merge_intervals(intervals):
        if not intervals:
            return []
        # Normalize to list of [s,e]
        norm = []
        for iv in intervals:
            if isinstance(iv, dict):
                s, e = iv['start'], iv['end']
            else:
                s, e = iv[0], iv[1]
            norm.append([s, e])
        norm.sort(key=lambda x: x[0])
        merged = [norm[0]]
        for s, e in norm[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    gt = merge_intervals(gt_intervals)
    pr = merge_intervals(pred_intervals)

    # Compute intersection length
    inter = 0.0
    for gs, ge in gt:
        for ps, pe in pr:
            inter += max(0.0, min(ge, pe) - max(gs, ps))
    # Compute union = sum(gt)+sum(pr)-inter
    gt_len = sum(e - s for s, e in gt)
    pr_len = sum(e - s for s, e in pr)
    union = gt_len + pr_len - inter
    iou = inter / union if union > 0 else 0.0
    return iou, inter, union

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate intro detection using IoU')
    parser.add_argument('--labels', type=Path, required=True,
                        help='cleaned_labels.csv with ground truth intervals')
    parser.add_argument('--predictions', type=Path, required=True,
                        help='intro_segments.json with predicted intervals')
    parser.add_argument('--thr', type=float, default=0.0,
                        help='Threshold: count episodes with IoU >= thr')
    args = parser.parse_args()

    # Load ground truth and derive episode_id
    df_gt = pd.read_csv(args.labels)
    df_gt['episode_id'] = df_gt['url'].apply(lambda u: Path(u).stem)
    gt_dict = df_gt.groupby('episode_id')[['start', 'end']].apply(
        lambda grp: grp.to_records(index=False).tolist()
    ).to_dict()

    # Load predictions
    preds = json.loads(Path(args.predictions).read_text())
    pred_dict = {}
    for entry in preds:
        pred_dict.setdefault(entry['episode_id'], []).append([entry['start'], entry['end']])

    # Evaluate per episode
    total_inter, total_union = 0.0, 0.0
    ious = []
    count_thr = 0
    print("Episode IoUs:")
    for epi, gt_ints in gt_dict.items():
        pr_ints = pred_dict.get(epi, [])
        iou, inter, union = compute_iou(gt_ints, pr_ints)
        print(f"  {epi}: IoU={iou:.3f} (∩={inter:.2f}, ∪={union:.2f})")
        ious.append(iou)
        total_inter += inter
        total_union += union
        if iou >= args.thr:
            count_thr += 1

    if not ious:
        print("No episodes to evaluate.")
    else:
        mean_iou = np.mean(ious)
        global_iou = total_inter / total_union if total_union > 0 else 0.0
        print(f"\nMean IoU over {len(ious)} episodes: {mean_iou:.3f}")
        print(f"Global IoU (all intervals): {global_iou:.3f}")
        if args.thr > 0:
            pct = count_thr / len(ious) * 100
            print(f"Episodes with IoU ≥ {args.thr:.2f}: {count_thr}/{len(ious)} ({pct:.1f}%)")
