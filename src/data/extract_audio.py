#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
import librosa
import numpy as np

def extract_audio(video_path: Path, out_dir: Path,
                  sr: int = 16000, n_mels: int = 128,
                  hop_length: int = 160, win_length: int = 400):
    out_wav = out_dir / 'audio.wav'
    out_specs = out_dir / 'mels'
    out_specs.mkdir(parents=True, exist_ok=True)

    # ffmpeg → WAV
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-ar', str(sr), '-ac', '1', '-f', 'wav',
        str(out_wav)
    ]
    subprocess.run(cmd, check=True)

    # mel-спектр
    y, _ = librosa.load(str(out_wav), sr=sr)
    secs = len(y) // sr
    for t in range(secs):
        seg = y[t*sr:(t+1)*sr]
        S = librosa.feature.melspectrogram(y=seg, sr=sr,
                                           n_mels=n_mels,
                                           hop_length=hop_length,
                                           win_length=win_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        np.save(out_specs / f'mel_{t:06d}.npy', S_db)

    print(f'Extracted audio to {out_wav}')
    print(f'Saved {secs} mel-spectrograms to {out_specs}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video',   type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--sr',      type=int, default=16000)
    p.add_argument('--n_mels',  type=int, default=128)
    p.add_argument('--hop_length', type=int, default=160)
    p.add_argument('--win_length', type=int, default=400)
    args = p.parse_args()
    extract_audio(args.video, args.out_dir,
                  args.sr, args.n_mels, args.hop_length, args.win_length)
