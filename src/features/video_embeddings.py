import argparse
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def extract_video_embeddings(frames_meta: Path, out_path: Path,
                             model_name: str = 'ViT-B-32',
                             pretrained: str = 'laion2b_e16',
                             device: str = 'cuda',
                             batch_size: int = 32):
    """
    Load frames metadata, encode images with CLIP, and save embeddings to .npy
    """
    # Load metadata
    df = pd.read_csv(frames_meta)
    frame_paths = df['frame_path'].tolist()

    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=torch.device(device)
    )
    model.eval()

    # Allocate output
    n = len(frame_paths)
    with torch.no_grad():
        embeddings = []
        for i in tqdm(range(0, n, batch_size), desc='Video Embeddings'):
            batch_paths = frame_paths[i:i+batch_size]
            images = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            images = torch.stack(images, dim=0).to(device)
            emb = model.encode_image(images)
            emb = emb.cpu().numpy()
            embeddings.append(emb)
            print(i)
        print(n)
        embeddings = np.vstack(embeddings)

    # Save
    np.save(out_path, embeddings)
    print(f'Saved video embeddings to {out_path} (shape {embeddings.shape})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute CLIP embeddings for video frames')
    parser.add_argument('--meta',     type=Path, required=True,
                        help='Path to frames_meta.csv')
    parser.add_argument('--out',      type=Path, required=True,
                        help='Path to output .npy embeddings')
    parser.add_argument('--model',    type=str, default='ViT-B-32', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='laion2b_e16', help='Pretrained weights')
    parser.add_argument('--device',   type=str, default='cuda', help='Torch device')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extract_video_embeddings(args.meta, args.out, args.model, args.pretrained, args.device, args.batch_size)
