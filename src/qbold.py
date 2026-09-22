"""
Настоящий шаг T2-картирования из авторского MATLAB-конвейера mq-BOLD
(qBOLD_BIDS_Hct_April21.zip, run3_mqBOLD_rOEF.m / calc_T2_map.m,
GitLab: https://gitlab.lrz.de/nmrm_lab/public_projects/mq-bold), перенесён
на Python и запускается на реальных 8-эховых данных MESE, полученных с
OpenNeuro (см. scripts/download_mese.py).

ЧТО ЭТО ДЕЛАЕТ: повоксельная моноэкспоненциальная подгонка T2 по 8-эховой
серии MESE: S(TE) = a * exp(-TE / T2). Исходный MATLAB-код (calc_T2_map.m /
fit2param.m) решает это собственным методом бисекции по T2 в сочетании с
замкнутым линейным решением для амплитуды. Здесь вместо этого используется
стандартная, эквивалентная линеаризованная подгонка: log(S) = log(a) -
TE/T2, решаемая по каждому вокселю методом обычных наименьших квадратов
(векторизованно, без Python-цикла по вокселям). Это хорошо
зарекомендовавший себя, учебниковый подход для моноэкспоненциальной
релаксометрии, который должен близко совпадать с результатом бисекции из
источника на хорошо обусловленном (положительном, монотонно убывающем)
сигнале, но это НЕ побайтовый перенос - явно отмечено, а не выдаётся за
идентичный результат.

ЧЕГО ЭТО НЕ ДЕЛАЕТ: это лишь T2-половина (спин-эхо) от R2'. Чтобы получить
настоящие R2', OEF и CMRO2, дополнительно нужны, для каждого условия
(задача vs базовый уровень): карта T2*/T2S (из отдельного
многоэхового градиент-эхо снимка - "MEGRE" - во время каждого условия
задачи), CBF (из pCASL), CBV (из DSC-перфузии с контрастным веществом) и
Hct каждого пациента. Ни одного из этих сырых входных данных нет в наличии
(см. README.md "Current data status") - этот модуль намеренно
останавливается на T2, единственной части, по которой у нас есть полные
реальные данные, вместо того чтобы заполнять остальное числами-заглушками.
"""

import numpy as np


def fit_t2_map(echo_volumes, echo_times_ms, mask=None, t2_clip_ms=150.0):
    """
    echo_volumes: массив (n_echoes, X, Y, Z) изображений магнитуды.
    echo_times_ms: массив (n_echoes,) времён эха в миллисекундах.
    mask: опциональный булев массив (X, Y, Z), ограничивающий подгонку
        (для вокселей вне маски возвращается T2=0). Если не задан,
        используются все воксели с положительным первым эхо (соответствует
        маске интенсивности источника вида
        `maske = vol(:,:,:,1) > mean(vol(:,:,:,1))`, здесь реализовано
        просто как "сигнал присутствует").

    Возвращает (t2_map_ms, amplitude_map, fit_error_percent), каждый формы
    (X,Y,Z). fit_error_percent - среднее абсолютное процентное отклонение
    подгонки от данных, соответствует `error`/`devmap` источника (то есть
    выходу T2.error).
    """
    echo_volumes = np.asarray(echo_volumes, dtype=np.float64)
    echo_times_ms = np.asarray(echo_times_ms, dtype=np.float64)
    n_echoes = echo_volumes.shape[0]
    if n_echoes != len(echo_times_ms):
        raise ValueError("несовпадение длин echo_volumes и echo_times_ms")

    shape = echo_volumes.shape[1:]
    if mask is None:
        mask = echo_volumes[0] > 0
    mask = mask.astype(bool)

    signal = echo_volumes.reshape(n_echoes, -1).T          # (n_вокселей, n_echoes)
    voxel_mask = mask.reshape(-1) & np.all(signal > 0, axis=1)

    t2_map = np.zeros(signal.shape[0], dtype=np.float64)
    amp_map = np.zeros(signal.shape[0], dtype=np.float64)
    error_map = np.zeros(signal.shape[0], dtype=np.float64)

    log_signal = np.log(signal[voxel_mask])                # (n_валидных, n_echoes)
    A = np.stack([np.ones(n_echoes), -echo_times_ms], axis=1)  # log(S) = log(a) - TE/T2
    coeffs, *_ = np.linalg.lstsq(A, log_signal.T, rcond=None)  # (2, n_валидных)
    log_a, inv_t2 = coeffs[0], coeffs[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(inv_t2 > 0, 1.0 / inv_t2, 0.0)
    amp = np.exp(log_a)

    predicted = amp[:, None] * np.exp(-echo_times_ms[None, :] / np.where(t2[:, None] == 0, np.inf, t2[:, None]))
    observed = signal[voxel_mask]
    pct_error = 100 * np.abs(observed - predicted).sum(axis=1) / np.abs(observed).sum(axis=1)

    t2_map[voxel_mask] = np.clip(t2, 0, t2_clip_ms)
    amp_map[voxel_mask] = amp
    error_map[voxel_mask] = pct_error

    return t2_map.reshape(shape), amp_map.reshape(shape), error_map.reshape(shape)
