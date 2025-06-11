import argparse
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path


def setup_logger(log_path: Path):
    logger = logging.getLogger('clean_labels')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_time(ts: str) -> float:
    """Convert time string H:MM:SS or MM:SS to seconds."""
    parts = ts.strip().split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = parts
        return m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {ts}")


def load_json_labels(path: Path, tag: str) -> pd.DataFrame:
    # JSON is dict of {url: {start: "0:01:42", end: "0:01:47", ...}}
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    df = pd.DataFrame.from_dict(raw, orient='index').reset_index()
    df = df.rename(columns={'index': 'url'})
    df['dataset'] = tag
    return df


def main(train_json: Path, test_json: Path, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'clean_report.md'
    cleaned_path = out_dir / 'cleaned_labels.csv'
    suspects_path = out_dir / 'suspects.csv'
    log_path = out_dir / 'clean_labels.log'

    logger = setup_logger(log_path)

    # Load JSON into DataFrames
    df_train = load_json_labels(train_json, 'train')
    df_test = load_json_labels(test_json, 'test')
    df = pd.concat([df_train, df_test], ignore_index=True)
    logger.info(f'Loaded {len(df_train)} train + {len(df_test)} test → {len(df)} total rows')

    # Required columns
    required = {'url', 'start', 'end'}
    assert required.issubset(df.columns), f'Missing columns: {required - set(df.columns)}'

    # Drop missing
    before = len(df)
    df = df.dropna(subset=list(required))
    dropped_na = before - len(df)
    logger.info(f'Dropped {dropped_na} rows with missing values')

    # Parse time strings → seconds
    df['start_sec'] = df['start'].astype(str).apply(parse_time)
    df['end_sec'] = df['end'].astype(str).apply(parse_time)

    # Swap if needed, drop zero-length
    mask_swap = df['start_sec'] > df['end_sec']
    swapped = mask_swap.sum()
    df.loc[mask_swap, ['start_sec', 'end_sec']] = df.loc[mask_swap, ['end_sec', 'start_sec']].values
    logger.info(f'Swapped start/end for {swapped} rows')
    mask_zero = df['start_sec'] == df['end_sec']
    zero_len = mask_zero.sum()
    df = df[~mask_zero]
    logger.info(f'Removed {zero_len} zero-length intervals')

    # Compute length
    df['length'] = df['end_sec'] - df['start_sec']

    # Remove duplicates by url, keep longest
    before_dup = len(df)
    df = df.sort_values('length', ascending=False)
    df = df.drop_duplicates(subset=['dataset', 'url'], keep='first')
    dropped_dup = before_dup - len(df)
    logger.info(f'Removed {dropped_dup} duplicate intervals per URL')

    # Outliers: length > max(20, µ+3σ)
    m, s = df['length'].mean(), df['length'].std()
    thresh = max(20.0, m + 3*s)
    mask_out = df['length'] > thresh
    suspects = df[mask_out]
    df_clean = df[~mask_out]
    logger.info(f'Moved {len(suspects)} outliers (length > {thresh:.1f}s) to suspects.csv')

    # Keep only needed columns
    df_clean = df_clean[['dataset', 'url', 'start_sec', 'end_sec']]
    df_clean = df_clean.rename(columns={'start_sec': 'start', 'end_sec': 'end'})

    # Type conversion
    df_clean['start'] = df_clean['start'].astype('float32')
    df_clean['end'] = df_clean['end'].astype('float32')

    # Save
    df_clean.to_csv(cleaned_path, index=False)
    suspects.to_csv(suspects_path, index=False)
    logger.info(f'Saved cleaned labels to {cleaned_path}, suspects to {suspects_path}')

    # Report
    with report_path.open('w') as f:
        f.write('# Cleaning Report\n')
        f.write(f'- Initial rows: {len(df_train) + len(df_test)}\n')
        f.write(f'- Dropped NA: {dropped_na}\n')
        f.write(f'- Swapped: {swapped}\n')
        f.write(f'- Zero-length removed: {zero_len}\n')
        f.write(f'- Duplicates removed: {dropped_dup}\n')
        f.write(f'- Outliers moved: {len(suspects)} (thresh {thresh:.1f}s)\n')
    logger.info(f'Report written to {report_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean JSON labels for episode intros')
    parser.add_argument('--train', required=True, type=Path, help='Path to train_labels.json')
    parser.add_argument('--test',  required=True, type=Path, help='Path to test_labels.json')
    parser.add_argument('--out_dir', required=False, type=Path, default=Path('data/interim'), help='Output directory')
    args = parser.parse_args()
    main(args.train, args.test, args.out_dir)
