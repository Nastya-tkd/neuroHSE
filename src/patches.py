"""
Извлечение 3D-патчей вокруг вокселей и защищённое от утечки разбиение
train/test.

Согласно указаниям руководителя: разделить мозг одного пациента на две
независимые половины (левое/правое полушарие или передняя/задняя часть),
чтобы соседние воксели никогда не оказывались по обе стороны разбиения
train/test - обычное случайное разбиение вокселей привело бы к утечке
информации, поскольку соседние воксели делят почти весь свой патч.
"""

import numpy as np


def extract_patch(volume, center, patch_size):
    """
    Кубический патч со стороной `patch_size` (должна быть нечётной) с
    центром в вокселе `center` = (i, j, k). Края, выходящие за границы
    объёма, дополняются нулями.
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size должен быть нечётным, чтобы у патча был чётко определённый центр")
    r = patch_size // 2
    i, j, k = center
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)

    lo = np.array([i - r, j - r, k - r])
    hi = np.array([i + r, j + r, k + r])
    src_lo = np.clip(lo, 0, None)
    src_hi = np.minimum(hi, np.array(volume.shape) - 1)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)

    patch[
        dst_lo[0]:dst_hi[0] + 1, dst_lo[1]:dst_hi[1] + 1, dst_lo[2]:dst_hi[2] + 1
    ] = volume[
        src_lo[0]:src_hi[0] + 1, src_lo[1]:src_hi[1] + 1, src_lo[2]:src_hi[2] + 1
    ]
    return patch


def extract_patches(volume, centers, patch_size):
    """centers: (N, 3) целочисленный массив координат вокселей. Возвращает (N, p, p, p)."""
    return np.stack([extract_patch(volume, c, patch_size) for c in centers], axis=0)


def labeled_voxel_coords(label_mask, brain_mask=None):
    """
    Координаты вокселей, в которых label_mask конечен и не равен нулю
    (то есть имеет определённое значение concordant(+1)/discordant(-1)),
    опционально ограниченные маской brain_mask. Возвращает целочисленный
    массив (N, 3).
    """
    valid = np.isfinite(label_mask) & (label_mask != 0)
    if brain_mask is not None:
        valid &= brain_mask.astype(bool)
    return np.argwhere(valid)


def split_by_axis(coords, axis_index, midpoint, margin):
    """
    Делит координаты вокселей на две защищённые от утечки группы вдоль
    одной оси (0=x/лево-право, 1=y/перед-зад, 2=z/низ-верх).

    Воксели в пределах `margin` вокселей от `midpoint` с любой стороны
    полностью отбрасываются, так что ни один патч с одной стороны не может
    пересекаться с патчем с другой стороны (безопасно, пока
    margin >= patch_size // 2).

    Возвращает булевы массивы (mask_side_a, mask_side_b) по первой оси
    coords, где side_a - это coord[axis_index] < midpoint - margin, а
    side_b - coord[axis_index] > midpoint + margin.
    """
    vals = coords[:, axis_index]
    side_a = vals < (midpoint - margin)
    side_b = vals > (midpoint + margin)
    return side_a, side_b


def hemisphere_midpoint(volume_shape, axis_index=0):
    """Индекс вокселя средней линии вдоль заданной оси, в предположении, что
    объём примерно центрирован на мозге вдоль этой оси (верно для этих
    изображений в пространстве пациента, которые не приведены аффинно к
    симметричному шаблону)."""
    return volume_shape[axis_index] / 2.0
