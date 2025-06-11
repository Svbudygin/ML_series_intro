import argparse
import subprocess
from pathlib import Path
import csv


def extract_frames(video_path: Path, out_dir: Path, fps: int = 1, scale_width: int = 256):
    """
    Extract frames from video at given fps and save as jpg, plus a CSV of frame paths and timestamps.
    """
    out_frames = out_dir / 'frames'
    out_frames.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'frames_meta.csv'

    # Use ffmpeg to extract frames
    # -vf fps=fps,scale=scale_width:-1
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps},scale={scale_width}:-1',
        '-q:v', '2',
        str(out_frames / 'frame_%06d.jpg')
    ]
    subprocess.run(cmd, check=True)

    # Build CSV of frame paths and timestamps
    with csv_path.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_path', 'timestamp'])
        # Frame files named frame_000001.jpg → timestamp = idx/fps
        for img in sorted(out_frames.glob('frame_*.jpg')):
            idx = int(img.stem.split('_')[1])
            ts = (idx - 1) / fps
            writer.writerow([str(img), f'{ts:.3f}'])

    print(f'Extracted frames to {out_frames} and metadata to {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--video', type=Path, required=True, help='Path to input video file')
    parser.add_argument('--out_dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    parser.add_argument('--scale_width', type=int, default=256, help='Width to resize frames (maintain aspect)')
    args = parser.parse_args()
    extract_frames(args.video, args.out_dir, args.fps, args.scale_width)

