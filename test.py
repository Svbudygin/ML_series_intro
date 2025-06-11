import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Замените на ваш реальный импорт compute_iou
from src.evaluate import compute_iou


def load_windows(path: Path) -> pd.DataFrame:
    """
    Загрузить DataFrame с окнами-предсказаниями из parquet.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл с окнами не найден: {path}")
    df = pd.read_parquet(path)
    # Проверка наличия нужных столбцов
    required_cols = {'episode_id', 'start', 'end', 'intro_score'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В DataFrame отсутствуют столбцы: {missing}")
    return df


def load_gt(path: Path) -> pd.DataFrame:
    """
    Загрузить CSV с ground truth, добавить столбец episode_id.
    Ожидается столбец 'url' и столбцы 'start', 'end' с временными метками.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV с GT не найден: {path}")
    gt = pd.read_csv(path)
    if 'url' not in gt.columns:
        raise ValueError("В GT нет столбца 'url'")
    # Выделяем episode_id из имени файла (stem)
    gt['episode_id'] = gt['url'].apply(lambda u: Path(u).stem)
    # Проверка: должны быть столбцы 'start' и 'end'
    for col in ['start', 'end']:
        if col not in gt.columns:
            raise ValueError(f"В GT нет столбца '{col}'")
    return gt


def load_json(path: Path) -> Optional[Dict]:
    """
    Загрузить JSON из файла, вернуть dict или None, если файл не найден.
    """
    if not path.exists():
        print(f"[INFO] Файл JSON не найден: {path}, пропускаем загрузку.")
        return None
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка при разборе JSON из {path}: {e}")


def post_with_thr(
    df_windows: pd.DataFrame,
    thr: float
) -> List[Dict]:
    """
    На основе окон df_windows и порога thr формирует по одному сегменту на эпизод:
    берёт первое окно (минимальный start) среди тех, где intro_score >= thr.
    Возвращает список dict: {"episode_id": ..., "start": float, "end": float}.
    """
    out = []
    # Группировка по episode_id
    for ep, grp in df_windows.groupby("episode_id"):
        # Фильтруем по порогу
        cand = grp[grp.intro_score >= thr]
        if cand.empty:
            # если нет ни одного окна выше порога — пропускаем эпизод
            continue
        # Сортируем по start, берём первое
        first = cand.nsmallest(1, 'start').iloc[0]
        out.append({
            "episode_id": ep,
            "start": float(first.start),
            "end": float(first.end)
        })
    return out


def compute_mean_iou_for_thresholds(
    df_windows: pd.DataFrame,
    gt: pd.DataFrame,
    thresholds: List[float]
) -> List[Tuple[float, float]]:
    """
    Для каждого thr в thresholds считает средний IoU:
      1) строит предсказанные сегменты пост-обработкой post_with_thr
      2) для каждого эпизода из GT вычисляет IoU между списком GT-интервалов
         (start/end из gt) и предсказанным сегментом (список длины 0 или 1)
      3) если для эпизода нет предсказанного сегмента, compute_iou должен вернуть 0 или
         вы можете явно обрабатывать этот случай
    Возвращает список кортежей (thr, mean_iou).
    """
    results: List[Tuple[float, float]] = []
    # Группировка GT по эпизодам заранее:
    gt_groups = {ep: group[['start', 'end']].values.tolist()
                 for ep, group in gt.groupby("episode_id")}
    all_episodes = list(gt_groups.keys())

    for thr in thresholds:
        segs = post_with_thr(df_windows, thr)
        # Построим словарь episode_id -> предсказанный список интервалов
        pred_dict = {item['episode_id']: [(item['start'], item['end'])] for item in segs}
        ious = []
        for epi in all_episodes:
            gt_intervals = gt_groups[epi]
            pr_intervals = pred_dict.get(epi, [])
            # compute_iou ожидает формат: list of [start, end] или list of tuples?
            # Предположим, что compute_iou принимает list из [start, end] списков:
            # преобразуем pr_intervals из [(s,e)] в [[s,e]] например.
            pr_list = [[float(s), float(e)] for (s, e) in pr_intervals]
            try:
                iou = compute_iou(gt_intervals, pr_list)
            except Exception as e:
                # Если compute_iou падает, можно залогировать
                print(f"[WARN] Ошибка compute_iou для эпизода {epi}, thr={thr}: {e}")
                iou = 0.0
            # На случай, если compute_iou возвращает None
            if iou is None:
                iou = 0.0
            ious.append(iou)
        mean_iou = float(np.mean(ious)) if ious else 0.0
        results.append((thr, mean_iou))
    return results


def find_best_threshold(results: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Из списка (thr, mean_iou) находит пару с максимальным mean_iou.
    """
    if not results:
        return 0.0, 0.0
    best = max(results, key=lambda x: x[1])
    return best


def main():
    # Пути к файлам; скорректируйте по необходимости
    windows_path = Path("data/processed/windows_pred.parquet")
    gt_path = Path("data/interim/cleaned_labels.csv")
    json_path = Path("results/intro_segments.json")  # опциональный файл

    # 1) Загрузка данных
    print("Загружаем предсказанные окна...")
    dfw = load_windows(windows_path)
    print(f"Всего окон: {len(dfw)}")

    print("Загружаем GT...")
    gt = load_gt(gt_path)
    n_rows_gt = len(gt)
    n_eps = gt['episode_id'].nunique()
    print(f"GT-строк: {n_rows_gt}, эпизодов: {n_eps}")

    # 2) Опционально: загрузка JSON финальных сегментов
    preds_ref = load_json(json_path)
    if preds_ref is not None:
        print("Загружены предикты финальных сегментов (intro_segments).")
        # здесь можно использовать preds_ref по задаче, если нужно
    else:
        print("Файл с финальными сегментами не загружен.")

    # 3) Генерируем сетку порогов
    thresholds = list(np.linspace(0.1, 0.9, 17))
    print(f"Перебираем пороги: {thresholds}")

    # 4) Считаем средний IoU для каждого thr
    print("Вычисляем средний IoU для каждого порога...")
    results = compute_mean_iou_for_thresholds(dfw, gt, thresholds)

    # 5) Находим лучший
    best_thr, best_iou = find_best_threshold(results)
    print(f"🎯 Лучший порог: {best_thr:.2f}, средний IoU = {best_iou:.3f}")

    # 6) (Опционально) можно вывести таблицу результатов или сохранить её
    # Например, сохранить в CSV:
    results_df = pd.DataFrame(results, columns=['threshold', 'mean_iou'])
    results_csv = Path("results/threshold_iou_results.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print(f"Результаты по порогам сохранены в {results_csv}")

    # 7) (Опционально) визуализация зависимости
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(results_df['threshold'], results_df['mean_iou'], marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Mean IoU')
        plt.title('Зависимость среднего IoU от порога')
        plt.grid(True)
        # Сохраним картинку
        fig_path = Path("results/threshold_vs_iou.png")
        plt.savefig(fig_path)
        print(f"График зависимости сохранён в {fig_path}")
        # Если вы запускаете в Jupyter, можно также plt.show()
    except ImportError:
        print("matplotlib не установлен, пропускаем визуализацию.")

if __name__ == "__main__":
    main()
