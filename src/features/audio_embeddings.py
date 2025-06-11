import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_audio_embeddings(mels_dir: Path, out_path: Path):
    """
    Extract simple audio embeddings by averaging mel-spectrograms per second.
    Each .npy file in mels_dir corresponds to one 1-second segment.
    Output is a Numpy array of shape (n_segments, n_mels).
    """
    # Find all mel files
    mel_files = sorted(mels_dir.glob('*.npy'))
    if not mel_files:
        raise FileNotFoundError(f"No .npy files found in {mels_dir}")

    embeddings = []
    for mel_path in tqdm(mel_files, desc='Audio Embeddings'):
        # Load mel-spectrogram: shape (n_mels, time_frames)
        S_db = np.load(mel_path)
        # Simple embedding: average over time axis → (n_mels,)
        emb = S_db.mean(axis=1)
        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)
    # Save embeddings: shape (n_segments, n_mels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Saved audio embeddings to {out_path} with shape {embeddings.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute audio embeddings from mel spectrograms')
    parser.add_argument('--mels_dir', type=Path, required=True,
                        help='Directory with mel_*.npy files')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output .npy file for embeddings')
    args = parser.parse_args()
    extract_audio_embeddings(args.mels_dir, args.out)
