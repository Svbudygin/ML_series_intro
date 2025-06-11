import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time


def process_episode(ep_path, labels_df, min_sec, max_sec, step_sec):
    ep_id = ep_path.name.split('_', 1)[1]
    vid_file = ep_path / 'video_embeds.npy'
    aud_file = ep_path / 'audio_embeds.npy'
    if not vid_file.exists() or not aud_file.exists():
        return [], []

    video_emb = np.load(vid_file, mmap_mode='r')
    audio_emb = np.load(aud_file, mmap_mode='r')
    T = min(video_emb.shape[0], audio_emb.shape[0])

    mask = labels_df['url'].str.contains(ep_id)
    ep_labels = labels_df[mask][['start', 'end']].values.tolist()
    has_labels = bool(ep_labels)

    train_rows, test_rows = [], []
    for start in range(0, T - min_sec + 1, step_sec):
        end_max = min(start + max_sec, T)
        for end in range(start + min_sec, end_max + 1, step_sec):
            vid_vec = video_emb[start:end].mean(axis=0)
            aud_vec = audio_emb[start:end].mean(axis=0)
            center = start + (end - start) / 2
            if has_labels:
                is_intro = int(any(st <= center <= en for st, en in ep_labels))
            else:
                is_intro = np.nan

            row = {
                'episode_id': ep_id,
                'start': float(start),
                'end': float(end),
                'length': float(end - start),
                'rel_start': float(start / T),
                'is_intro': is_intro
            }
            for i, v in enumerate(vid_vec):
                row[f'v{i}'] = float(v)
            for i, v in enumerate(aud_vec):
                row[f'a{i}'] = float(v)

            if has_labels:
                train_rows.append(row)
            else:
                test_rows.append(row)
    return train_rows, test_rows


def build_windows(interim_dir: Path, labels_df: pd.DataFrame,
                  min_sec: int, max_sec: int, step_sec: int,
                  n_workers: int = None):
    ep_dirs = sorted(interim_dir.glob('episode_*'))
    print(f"Found {len(ep_dirs)} episodes", flush=True)

    train_list, test_list = [], []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_episode, ep, labels_df, min_sec, max_sec, step_sec): ep for ep in ep_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Episodes'):
            tr, te = fut.result()
            train_list.extend(tr)
            test_list.extend(te)
    print(f"Processed episodes in {time.time() - start_time:.1f}s", flush=True)

    train_df = pd.DataFrame(train_list)
    test_df = pd.DataFrame(test_list)
    print(f"Built windows: train={train_df.shape}, test={test_df.shape}", flush=True)
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Build sliding windows from embeddings')
    parser.add_argument('--interim_dir',    type=Path, required=True, help='Path to data/interim')
    parser.add_argument('--cleaned_labels', type=Path, required=True, help='Path to cleaned_labels.csv')
    parser.add_argument('--out_dir',        type=Path, required=True, help='Output directory for windows_*.parquet')
    parser.add_argument('--min_sec',        type=int, default=3)
    parser.add_argument('--max_sec',        type=int, default=12)
    parser.add_argument('--step_sec',       type=int, default=1)
    parser.add_argument('--workers',        type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    labels_df = pd.read_csv(args.cleaned_labels)
    print("Starting window build...", flush=True)
    train_df, test_df = build_windows(Path(args.interim_dir), labels_df,
                                      args.min_sec, args.max_sec, args.step_sec,
                                      args.workers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving train parquet...", flush=True)
    train_df.to_parquet(args.out_dir / 'windows_train.parquet', index=False)
    print("Saving test parquet...", flush=True)
    test_df.to_parquet(args.out_dir / 'windows_test.parquet', index=False)
    print(f"✔ Saved windows: train={len(train_df)}, test={len(test_df)}", flush=True)

if __name__ == '__main__':
    main()
