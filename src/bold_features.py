"""
Эксперимент 2: простой BOLD-сигнал как вектор признаков по вокселю, наряду
со структурным патчем.

На вход подаётся <sub>_task-all_space-T2_filtered_func.nii.gz` - полностью
предобработанные функциональные данные FSL FEAT (коррекция движения,
пространственное сглаживание и временная фильтрация верхних частот уже
применены; это тот же файл, на котором работает GLM первого уровня самого
исходного конвейера). Намеренно НЕ используется минимально предобработанная
альтернатива desc-preproc_bold, которая тоже существует, так как
руководитель специально просил отфильтрованные/детрендированные данные, а
не "что попроще".

Что этот модуль добавляет сверх этого: повоксельное линейное детрендирование
(лёгкая, стандартная подстраховка - фильтр верхних частот FEAT убирает
медленный дрейф в частотной области, а это убирает любой остаточный
линейный тренд во временной области) и z-нормализация (собственный
временной ряд каждого вокселя приводится к нулевому среднему/единичной
дисперсии), чтобы воксели с разной исходной интенсивностью
сигнала/усилением сканера были сопоставимы перед подачей в сеть - та же
логика, что и при нормализации структурных патчей по статистике всего мозга
T1.

Чего этот модуль НЕ делает: симметризацию неоднородности поля/артефактов
выпадения сигнала (потеря сигнала около границ воздух-ткань, например в
орбитофронтальных/височных областях), упомянутую руководителем как реальную
проблему для данных qBOLD/EPI. Для этого нужны индивидуальные карты поля
пациента и настоящий шаг коррекции искажений, а не общая операция над
временным рядом вокселя - отмечено как открытый пробел, а не молча
пропущено.
"""

import numpy as np
from scipy.signal import detrend


def extract_bold_vectors(bold_4d, coords):
    """
    bold_4d: массив (X, Y, Z, T).
    coords: целочисленный массив (N, 3) координат вокселей.
    Возвращает float32-массив (N, T), один сырой временной ряд на координату.
    """
    return bold_4d[coords[:, 0], coords[:, 1], coords[:, 2], :].astype(np.float32)


def normalize_bold_vectors(vectors, eps=1e-6):
    """Повоксельное линейное детрендирование + z-нормализация. vectors: (N, T) -> (N, T) float32."""
    detrended = detrend(vectors, axis=1, type="linear")
    mean = detrended.mean(axis=1, keepdims=True)
    std = detrended.std(axis=1, keepdims=True)
    return ((detrended - mean) / (std + eps)).astype(np.float32)


def parse_events_tsv(path):
    """Читает BIDS events.tsv (колонки onset, duration, trial_type, в секундах)."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((float(row["onset"]), float(row["duration"]), row["trial_type"]))
    return rows


def condition_block_indices(events, trial_type, tr, n_timepoints, skip_seconds=4.8):
    """
    Индексы временных точек (TR), попадающих внутрь блоков `trial_type`, с
    пропуском первых `skip_seconds` каждого блока. Пропуск учитывает
    гемодинамическую задержку: BOLD-сигналу нужно ~4-6с, чтобы начать
    нарастать после начала блока, поэтому включение этих TR подмешало бы
    сигнал из хвоста отклика *предыдущего* блока - осознанный выбор, а не
    "как обычно делают" (согласно акценту руководителя на понимании каждого
    шага).
    """
    indices = []
    for onset, duration, cond in events:
        if cond != trial_type:
            continue
        start = int(np.ceil((onset + skip_seconds) / tr))
        end = int(np.floor((onset + duration) / tr))
        indices.extend(range(max(start, 0), min(end, n_timepoints)))
    return sorted(set(indices))


def compute_condition_features(bold_4d, coords, events, tr, conditions=("calc", "mem", "rest"), skip_seconds=4.8):
    """
    Процентное изменение сигнала по вокселю и по условию относительно
    временного среднего этого вокселя по всему run: для каждого условия -
    среднее значение BOLD во время блоков этого условия (с поправкой на
    задержку, см. condition_block_indices) минус общее среднее по run,
    делённое на общее среднее.

    "rest" используется здесь как метка условия базового уровня без задачи
    (в других местах именования файлов этого конвейера называется
    "control") - дизайн блоков в events.tsv содержит только типы испытаний
    calc/mem/rest, отдельной метки "control" нет, а rest - единственное
    условие без задачи, поэтому это соответствие выведено, а не задано явно
    - отмечено здесь, а не принято молча как предположение.

    Возвращает float32-массив (N, len(conditions)).
    """
    series = bold_4d[coords[:, 0], coords[:, 1], coords[:, 2], :].astype(np.float64)  # (N, T)
    grand_mean = series.mean(axis=1)

    feats = np.zeros((series.shape[0], len(conditions)), dtype=np.float64)
    for i, cond in enumerate(conditions):
        idx = condition_block_indices(events, cond, tr, series.shape[1], skip_seconds)
        cond_mean = series[:, idx].mean(axis=1)
        feats[:, i] = (cond_mean - grand_mean) / grand_mean * 100

    return feats.astype(np.float32)
