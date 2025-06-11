# ML Series Intro Finder

This repository provides an end-to-end pipeline to automatically detect and extract the intro segment in TV series episodes. The pipeline is modular, allowing you to run, debug, and optimize each stage independently.

## Project Structure

```
ML_series_intro/
├── data/
│   ├── raw/         # Original MP4 episodes
│   │   └── episodes/
│   ├── interim/     # Extracted frames, audio spectrograms, embeddings
│   └── processed/   # Sliding window feature tables for modeling
├── models/          # Trained LightGBM model & feature importances
└── src/             # Python scripts for each pipeline stage
```

---

## 0. Setup Environment

```bash
python -m venv venv           # Create virtual environment
source venv/bin/activate      # Activate it
pip install --upgrade pip

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch librosa tqdm pandas pyarrow pillow lightgbm
```

> **Notes:**
>
> * PyTorch & open\_clip for video embeddings (CLIP)
> * librosa for audio feature extraction
> * lightgbm for modeling

---

## 1. Clean Labels

```bash
python src/data/clean_labels.py \
  --train data/raw/labels/train_labels.json \
  --test  data/raw/labels/test_labels.json \
  --out_dir data/interim
```

* Fixes interval errors (start/end swaps, zero-length, out-of-bounds)
* Removes duplicates and outliers
* Outputs `data/interim/cleaned_labels.csv`

---

## 2. Extract Frames & Audio

### 2a. Extract Frames

```bash
python src/data/extract_frames.py \
  --video data/raw/episodes/.../episode.mp4 \
  --out_dir data/interim/episode_<id> \
  --fps 1
```

*Batch over all episodes:*

```bash
for vid in data/raw/episodes/data_test_short/*/*.mp4; do
  sid=$(basename "$(dirname "$vid")")
  python src/data/extract_frames.py \
    --video "$vid" \
    --out_dir data/interim/episode_"$sid" --fps 1
done
```

> Outputs `frames/*.jpg` + `frames_meta.csv` (timestamps)

### 2b. Extract Audio → Mel-Spectrograms

```bash
python src/data/extract_audio.py \
  --video data/raw/episodes/.../episode.mp4 \
  --out_dir data/interim/episode_<id>
```

*Batch similarly as frames.*

> Outputs `mels/mel_000000.npy ...` (one per second)

---

## 3. Compute Embeddings

### 3a. Video Embeddings (CLIP)

```bash
python src/features/video_embeddings.py \
  --meta data/interim/episode_<id>/frames_meta.csv \
  --out  data/interim/episode_<id>/video_embeds.npy \
  --device cpu
```

> Produces `video_embeds.npy` shape T×512

### 3b. Audio Embeddings (Averaged Mel)

```bash
python src/features/audio_embeddings.py \
  --mels_dir data/interim/episode_<id>/mels \
  --out      data/interim/episode_<id>/audio_embeds.npy
```

> Produces `audio_embeds.npy` shape T×128

---

## 4. Build Sliding Windows

```bash
python src/data/build_windows.py \
  --interim_dir    data/interim \
  --cleaned_labels data/interim/cleaned_labels.csv \
  --out_dir        data/processed \
  --min_sec 3 --max_sec 12 --step_sec 1
```

* Generates:

  * `data/processed/windows_train.parquet` (features + `is_intro` label)
  * `data/processed/windows_test.parquet` (features only)

---

## 5. Train Model

```bash
python src/train.py \
  --train_parquet data/processed/windows_train.parquet \
  --model_out     models/lightgbm_intro.txt \
  --imp_out       models/feature_importances.csv
```

* Fits a binary LightGBM classifier on sliding-window features
* Saves model and feature importances

---

## 6. Predict on Test Windows

```bash
python src/predict.py \
  --model   models/lightgbm_intro.txt \
  --windows data/processed/windows_test.parquet \
  --out     data/processed/windows_pred.parquet
```

---

## 7. Post-Process Window Scores

```bash
python src/postprocess.py \
  --pred data/processed/windows_pred.parquet \
  --out  results/intro_segments.json \
  --thr  0.6
```

* Groups consecutive windows with score ≥ threshold
* Filters out too short/long segments
* Outputs final JSON of `{episode_id, start, end}`

---

## 8. Evaluate (IoU)

```bash
python src/evaluate.py \
  --labels      data/interim/cleaned_labels.csv \
  --predictions results/intro_segments.json
```

* Computes IoU per episode and mean IoU overall

---

## 9. End-to-End Wrapper

```bash
python src/inference_pipeline.py \
  --video data/raw/episodes/.../episode1.mp4 \
  --labels data/interim/cleaned_labels.csv \
  --model  models/lightgbm_intro.txt \
  --out    results/intro_segments.json
```

Runs all steps 2️⃣→7️⃣ sequentially in a temporary workspace.

---

## Pipeline Overview

1. **Label Cleaning**: ensures reliable ground truth.
2. **Data Extraction**: converts videos to frames & spectrograms.
3. **Embeddings**: generates CLIP & Mel embeddings.
4. **Windowing**: forms labeled/unlabeled windows.
5. **Modeling**: trains a LightGBM classifier.
6. **Post-Processing**: merges windows into final intervals.
7. **Evaluation**: measures performance (IoU).

