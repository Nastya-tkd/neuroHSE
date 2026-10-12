"""
Улучшенная регистрация атласа Glasser/HCP-MMP1.0, построенная на вновь
обнаруженных данных: полный, непорезанный T1 (`desc-fmriprep_T1w.nii.gz`,
~170x256x256, ~1мм) существует в derivatives каждого пациента рядом с
обрезанной до 30 срезов (99мм) версией `space-T2_desc-brain_T1w.nii.gz`,
которую весь проект использовал как единственный T1 до сих пор - потому
что именно в её сетке определены метки CMRO2/BOLD. Полный T1 никогда не
применялся, поскольку метки в нём не определены - но он даёт куда лучшую
промежуточную цель для регистрации атласа, чем исходная обрезанная сетка.

Старый путь (glasser_atlas.py): MNI (шаблон, низкое разрешение) напрямую
на обрезанную до 30 срезов T2-сетку - плохо обусловленная регистрация:
мало анатомических ориентиров, анизотропные воксели 2x2x3.3мм, частичное
поле обзора.

Новый путь (этот модуль): MNI -> полный T1 того же пациента (SyN,
полноценная трёхмерная анатомия, ~1мм) -> обрезанная T2-сетка того же
пациента (Affine, тот же мозг, просто другое разрешение/поле обзора - уже
хорошо обусловленная задача, поскольку анатомия идентична). Два
последовательных, лучше обусловленных шага вместо одного плохо
обусловленного.

Итоговая карта вокселей (та же сетка/метки, что у glasser_atlas.py) по
построению сравнима с ней - используется, чтобы проверить, была ли
неточность границ старой регистрации (явно раскрытая ограничением в
glasser_atlas.py) достаточно большой, чтобы объяснить часть нулевого
результата, а не просто числом ради числа.
"""

import os
import numpy as np

from src.glasser_atlas import ATLAS_LABELS_PATH, MNI_TEMPLATE_PATH, _ensure_mni_template

ATLAS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "atlas_cache")


def get_subject_glasser_atlas_v2(subject, t2_t1_path, native_t1_path, cache_dir=ATLAS_CACHE_DIR):
    """
    Та же возвращаемая форма, что у glasser_atlas.get_subject_glasser_atlas
    (целочисленный массив на сетке space-T2), но через двухшаговую цепочку
    MNI -> полный T1 -> T2-сетка вместо одного прямого шага MNI -> T2-сетка.
    """
    import ants

    out_path = os.path.join(cache_dir, f"{subject}_glasser_T2space_v2.nii.gz")
    if os.path.exists(out_path):
        img = ants.image_read(out_path)
        return img.numpy().astype(np.int32)

    if not os.path.exists(ATLAS_LABELS_PATH):
        raise FileNotFoundError(f"{ATLAS_LABELS_PATH} отсутствует.")
    _ensure_mni_template()

    t2_t1_img = ants.image_read(t2_t1_path)
    native_t1_img = ants.image_read(native_t1_path)
    mni_img = ants.image_read(MNI_TEMPLATE_PATH)
    atlas_img = ants.image_read(ATLAS_LABELS_PATH)

    # Шаг 1: MNI -> полный T1 пациента (SyN, хорошо обусловлено - полный мозг,
    # ~1мм, уже скальп-стрипнутый).
    reg1 = ants.registration(fixed=native_t1_img, moving=mni_img, type_of_transform="SyN")
    atlas_in_native = ants.apply_transforms(
        fixed=native_t1_img, moving=atlas_img, transformlist=reg1["fwdtransforms"], interpolator="genericLabel"
    )

    # Шаг 2: полный T1 -> обрезанная T2-сетка того же пациента (Affine - тот
    # же мозг, просто другое разрешение/поле обзора, не нужна нелинейная
    # деформация).
    reg2 = ants.registration(fixed=t2_t1_img, moving=native_t1_img, type_of_transform="Affine")
    atlas_in_t2 = ants.apply_transforms(
        fixed=t2_t1_img, moving=atlas_in_native, transformlist=reg2["fwdtransforms"], interpolator="genericLabel"
    )

    os.makedirs(cache_dir, exist_ok=True)
    ants.image_write(atlas_in_t2, out_path)
    return atlas_in_t2.numpy().astype(np.int32)
