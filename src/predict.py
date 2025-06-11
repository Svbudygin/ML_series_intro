import argparse
import pandas as pd
import lightgbm as lgb
from pathlib import Path


def predict(model_path: Path, windows_path: Path, out_path: Path):
    booster = lgb.Booster(model_file=str(model_path))
    df = pd.read_parquet(windows_path)
    if df.empty:
        print(f"[predict] {windows_path} is empty – nothing to predict.")
        return
    X = df.drop(columns=['episode_id', 'is_intro'], errors='ignore')
    df['intro_score'] = booster.predict(X, num_iteration=booster.best_iteration)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved predictions → {out_path}  (rows={len(df)})")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Predict intro scores for windows')
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--windows', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    predict(args.model, args.windows, args.out)
