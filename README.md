# Детектор заставок в сериалах

Полный, воспроизводимый пайплайн, который на вход получает серии в формате MP4, а на выходе — JSON‑файл с временем начала и конца короткой заставки (интро).


---
## 📂 Куда положить данные
```
ML_series_intro/
└── data/
    └── raw/
        ├── labels/                    # train_labels.json, test_labels.json
        └── episodes/                 # любые подпапки с mp4‑файлами
            ├── data_test_short/
            │   ├── -220020068_456241671/-220020068_456241671.mp4
            │   └── -220020068_456249204/...
            └── ...
```

После запуска конвейера дерево проекта постепенно наполняется так:
```
ML_series_intro/
├── data/
│   ├── raw/labels/…                       # исходная разметка
│   ├── raw/episodes/…                     # все mp4
│   ├── interim/                           # промежуточка
│   │   ├── cleaned_labels.csv             # готовая разметка
│   │   ├── suspects.csv                   # подозрительные интервалы
│   │   └── episode_<id>/                  # один каталог на эпизод
│   │       ├── frames/ *.jpg              # кадры +-1 fps
│   │       ├── frames_meta.csv            # «кадр→время»
│   │       ├── mels/ mel_*.npy            # аудио‑спектры
│   │       ├── video_embeds.npy           # T×512 CLIP‑векторы
│   │       └── audio_embeds.npy           # T×128 Mel‑векторы
│   └── processed/
│       ├── windows_train.parquet          # окна + метка
│       ├── windows_test.parquet           # окна без метки (если есть)
│       └── windows_pred.parquet           # score модели
├── models/
│   ├── lightgbm_intro.txt                 # бинарь LightGBM
│   ├── lightgbm_intro.json                # мета‑инфо (AUC, итерации)
│   └── feature_importances.csv            # топ фич
├── results/
│   └── intro_segments.json                # финальные интро‑интервалы
├── notebooks/analysis.ipynb               # графики + EDA
└── src/                                   # весь код конвейера
```
---
## 0 — Подготовка окружения
```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch librosa tqdm pandas pyarrow pillow lightgbm
```

---
## 1 — Очистка разметки
```bash
python src/data/clean_labels.py \
  --train data/raw/labels/train_labels.json \
  --test  data/raw/labels/test_labels.json  \
  --out_dir data/interim
```
*Исправляет неверные интервалы, удаляет дубликаты и выбросы.*  
> Выход: `data/interim/cleaned_labels.csv`

---
## 2 — Извлечение кадров и аудио
### 2.1 Кадры (ffmpeg, 1 fps)
```bash
for vid in data/raw/episodes/*/*/*.mp4; do
  id=$(basename "$(dirname "$vid")")          # «-220020068_456241671»
  python src/data/extract_frames.py \
    --video   "$vid" \
    --out_dir data/interim/episode_"$id" \
    --fps 1
done
```
Создаётся `frames/*.jpg` + `frames_meta.csv`.

### 2.2 Аудио → мел‑спектры
```bash
for vid in data/raw/episodes/*/*/*.mp4; do
  id=$(basename "$(dirname "$vid")")
  python src/data/extract_audio.py \
    --video   "$vid" \
    --out_dir data/interim/episode_"$id"
done
```
Создаётся `mels/mel_000000.npy` (по одному файлу на секунду).

---
## 3 — Векторизация эмбеддингов
### 3.1 Видео (CLIP)
```bash
for ep in data/interim/episode_*; do
  python src/features/video_embeddings.py \
    --meta "$ep/frames_meta.csv" \
    --out  "$ep/video_embeds.npy" \
    --device cpu
done
```
### 3.2 Аудио (средний Mel)
```bash
for ep in data/interim/episode_*; do
  python src/features/audio_embeddings.py \
    --mels_dir "$ep/mels" \
    --out      "$ep/audio_embeds.npy"
done
```

---
## 4 — Формирование скользящих окон
```bash
python src/data/build_windows.py \
  --interim_dir    data/interim \
  --cleaned_labels data/interim/cleaned_labels.csv \
  --out_dir        data/processed \
  --min_sec 3 --max_sec 12 --step_sec 1 \
  --workers 4       # число потоков = ядер CPU
```
*Окно ⪅12 с сдвигается каждую секунду; усредняются эмбеддинги → таблица фич.*  
> `windows_train.parquet` (с меткой `is_intro`), `windows_test.parquet` (если есть эпизоды без лейблов).

---
## 5 — Обучение LightGBM
```bash
python src/train.py \
  --train_parquet data/processed/windows_train.parquet \
  --model_out     models/lightgbm_intro.txt \
  --imp_out       models/feature_importances.csv
```
Сохраняется модель и важность фич.

---
## 6 — Предсказания окон
```bash
python src/predict.py \
  --model   models/lightgbm_intro.txt \
  --windows data/processed/windows_train.parquet \
  --out     data/processed/windows_pred.parquet
```
*(если `windows_test.parquet` не пустой — меняйте путь)*

---
## 7 — Склейка окон в единую заставку
```bash
python src/postprocess.py \
  --pred data/processed/windows_pred.parquet \
  --out  results/intro_segments.json \
  --thr  0.6
```
Алгоритм группирует соседние окна со score ≥ 0.6 и отрезает слишком короткие/длинные.

---
## 8 — Проверка качества (IoU)
```bash
python src/evaluate.py \
  --labels      data/interim/cleaned_labels.csv \
  --predictions results/intro_segments.json
```
Выводит IoU для каждого эпизода и средний.

---
Внутри: извлекает кадры+аудио → эмбедды → окна → предсказание → пост‑обработка.

---
## 🔍 Аналитика результатов
В папке `notebooks/` лежит `analysis.ipynb`, где строятся:
- лог‑гистограмма `intro_score`
- распределение положения окон с высоким score
- графики score по времени для выбранных эпизодов

> **Совет:** после запуска ноутбука сделайте скриншот графиков и прикрепите в отчёт.

---
## Шаги конвейера в двух словах
| Шаг | Назначение |
|-----|-------------|
| **Очистка лейблов** | Убираем ошибки разметки, чтобы модель училась на чистых данных |
| **Извлечение данных** | Превращаем видео в кадры и звуковую «картинку» |
| **Эмбеддинги** | CLIP + средний Mel → компактные векторы вместо «сырых» пикселей |
| **Окна** | Смотрим на фрагменты 3–12 с и помечаем, попадают ли они в заставку |
| **LightGBM** | Учим бинарный классификатор «окно — интро / не интро» |
| **Post‑processing** | Склеиваем окна, получаем один чистый интервал интро |
| **Evaluation** | Проверяем IoU, подбираем порог |

