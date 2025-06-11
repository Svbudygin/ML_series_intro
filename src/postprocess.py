import argparse
import pandas as pd
from pathlib import Path


def group_windows(df: pd.DataFrame, thr: float = 0.6, min_len: int = 3, max_len: int = 20):
    """Return list of (episode_id, start, end) after grouping high-score windows."""
    results = []
    for ep_id, g in df.groupby('episode_id'):
        cand = g[g['intro_score'] >= thr].sort_values('start')
        if cand.empty:
            continue
        cur_s, cur_e = None, None
        for _, row in cand.iterrows():
            s, e = row['start'], row['end']
            if cur_s is None:
                cur_s, cur_e = s, e
                continue
            # если перекрывается/соседствует → расширяем
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                if min_len <= cur_e - cur_s <= max_len:
                    results.append((ep_id, cur_s, cur_e))
                cur_s, cur_e = s, e
        # push last
        if cur_s is not None and min_len <= cur_e - cur_s <= max_len:
            results.append((ep_id, cur_s, cur_e))
    return results


def main(pred_path: Path, out_json: Path, thr: float):
    df = pd.read_parquet(pred_path)
    segments = group_windows(df, thr)
    # write simple JSON list
    import json
    out = [{'episode_id': ep, 'start': float(s), 'end': float(e)} for ep, s, e in segments]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved final intro segments → {out_json} (episodes with intro: {len(out)})")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Post-process window scores into intro intervals')
    p.add_argument('--pred', required=True, type=Path, help='windows_pred.parquet')
    p.add_argument('--out',  required=True, type=Path, help='intro_segments.json')
    p.add_argument('--thr',  type=float, default=0.6, help='score threshold')
    args = p.parse_args()
    main(args.pred, args.out, args.thr)
