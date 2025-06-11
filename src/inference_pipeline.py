import argparse
import subprocess
from pathlib import Path
import tempfile


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end intro detection pipeline')
    parser.add_argument('--video', type=Path, required=True, help='Input episode MP4')
    parser.add_argument('--labels', type=Path, required=True, help='cleaned_labels.csv for building windows')
    parser.add_argument('--model', type=Path, required=True, help='LightGBM model file')
    parser.add_argument('--out', type=Path, required=True, help='Output JSON for intro segments')
    parser.add_argument('--min_sec', type=int, default=3)
    parser.add_argument('--max_sec', type=int, default=12)
    parser.add_argument('--step_sec', type=int, default=1)
    parser.add_argument('--thr', type=float, default=0.6)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ep_id = args.video.stem
        interim = tmp_path / f'episode_{ep_id}'
        processed = tmp_path / 'processed'
        interim.mkdir()
        processed.mkdir()

        # 1. extract frames
        run(['python', 'src/data/extract_frames.py',
             '--video', str(args.video),
             '--out_dir', str(interim),
             '--fps', '1'])
        # 2. extract audio
        run(['python', 'src/data/extract_audio.py',
             '--video', str(args.video),
             '--out_dir', str(interim)])
        # 3. video embeddings
        run(['python', 'src/features/video_embeddings.py',
             '--meta', str(interim / 'frames_meta.csv'),
             '--out',  str(interim / 'video_embeds.npy'),
             '--device', 'cpu'])
        # 4. audio embeddings
        run(['python', 'src/features/audio_embeddings.py',
             '--mels_dir', str(interim / 'mels'),
             '--out',      str(interim / 'audio_embeds.npy')])
        # 5. build windows
        run(['python', 'src/data/build_windows.py',
             '--interim_dir',    str(tmp_path),
             '--cleaned_labels', str(args.labels),
             '--out_dir',        str(processed),
             '--min_sec',        str(args.min_sec),
             '--max_sec',        str(args.max_sec),
             '--step_sec',       str(args.step_sec)])
        # 6. predict
        run(['python', 'src/predict.py',
             '--model',  str(args.model),
             '--windows', str(processed / 'windows_test.parquet'),
             '--out',     str(processed / 'windows_pred.parquet')])
        # 7. postprocess
        run(['python', 'src/postprocess.py',
             '--pred', str(processed / 'windows_pred.parquet'),
             '--out',  str(args.out),
             '--thr',  str(args.thr)])

        print(f"\nDone! Intro segments saved to {args.out}")
