#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
import csv

def extract_frames(video_path: Path, out_dir: Path, fps: int = 1, scale_width: int = 256):
    out_frames = out_dir / 'frames'
    out_frames.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'frames_meta.csv'

    # ffmpeg: вырезаем кадры
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps},scale={scale_width}:-1',
        '-q:v', '2',
        str(out_frames / 'frame_%06d.jpg')
    ]
    subprocess.run(cmd, check=True)

    # Составляем CSV
    with csv_path.open('w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['frame_path', 'timestamp'])
        for img in sorted(out_frames.glob('frame_*.jpg')):
            idx = int(img.stem.split('_')[1])
            ts = (idx - 1) / fps
            writer.writerow([str(img), f'{ts:.3f}'])

    print(f'Extracted {len(list(out_frames.glob("*.jpg")))} frames → {out_frames}')
    print(f'Metadata saved to {csv_path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video',    type=Path, required=True, help='Input mp4')
    p.add_argument('--out_dir',  type=Path, required=True, help='Where to dump frames/')
    p.add_argument('--fps',      type=int, default=1)
    p.add_argument('--scale_width', type=int, default=256)
    args = p.parse_args()
    extract_frames(args.video, args.out_dir, args.fps, args.scale_width)
