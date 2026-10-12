"""
Карта толщины коры как дополнительный структурный вход, построенная на
полном T1 (desc-fmriprep_T1w.nii.gz, найден и впервые использован в
src/glasser_atlas_v2.py) - самое близкое приближение к тому, что дала бы
surface-based обработка через FreeSurfer, но без самого FreeSurfer:
толщина коры считается напрямую через алгоритм DiReCT (Das, Avants et al.
2009), реализованный в ANTs как `KellyKapowski` и доступный из ANTsPy
(`ants.kelly_kapowski`) - без необходимости в поверхностной сетке,
recon-all или лицензируемом бинарнике.

Три честных приближения, заявленных сразу, а не скрытых:
1. Сегментация тканей (CSF/GM/WM) через `ants.kmeans_segmentation` -
   простая k-means классификация по интенсивности с MRF-сглаживанием, а
   не полная multi-atlas-регуляризованная сегментация, которую обычно
   использует конвейер FreeSurfer/ANTs для этого шага.
2. T1 передискретизирован до 1.5мм (с исходных ~1мм) перед сегментацией
   и DiReCT - на полном разрешении KellyKapowski занимает >27 минут на
   пациента (однопоточный тест на sub-p019 не завершился и за 27 минут),
   что для когорты 39 пациентов физически не укладывается в сессию.
3. DiReCT запущен с its=25 вместо рекомендованных по умолчанию 45 -
   сходимость проверена напрямую на sub-p019: its=15/25/45 дают средние
   толщины 1.57/1.78/1.87мм соответственно (видно явное схождение, а не
   произвол) - его=25 выбрано как разумный компромисс между временем
   (~230с/пациента) и полнотой сходимости, а не как число "по умолчанию".

Абсолютные значения толщины поэтому смещены вниз относительно
литературных (обычно ~2.5мм в среднем) - это не заявляется как точная
посрезовая мера толщины коры. Но для цели этого эксперимента (несёт ли
*относительный пространственный паттерн* толщины сигнал, который не нёс
один T1) абсолютная калибровка не обязательна - используется та же логика
многоканальных патчей, что и в run_cbf_oef.py/run_cbv.py: карта толщины
просто ещё один независимо нормализованный канал.

Карта толщины считается в (передискретизированном) полном T1-пространстве
пациента, затем переносится в обрезанную до 30 срезов T2-сетку той же
Affine-регистрацией "полный T1 -> T2-сетка", что уже построена и
провалидирована в src/glasser_atlas_v2.py (линейная интерполяция, а не
genericLabel - толщина непрерывная величина, не категориальная метка).
"""

import os
import numpy as np

from src.glasser_atlas_v2 import ATLAS_CACHE_DIR


def get_subject_thickness_map(subject, t2_t1_path, native_t1_brain_path, cache_dir=ATLAS_CACHE_DIR, its=25):
    """
    Возвращает массив (X,Y,Z) толщины коры (в мм) на той же сетке, что и
    t2_t1_path (обрезанное до 30 срезов T2-пространство). Кэшируется на
    диск после первого вычисления.
    """
    import ants

    out_path = os.path.join(cache_dir, f"{subject}_thickness_T2space.nii.gz")
    if os.path.exists(out_path):
        img = ants.image_read(out_path)
        return img.numpy().astype(np.float32)

    native_img = ants.image_read(native_t1_brain_path)
    native_rs = ants.resample_image(native_img, (1.5, 1.5, 1.5), use_voxels=False, interp_type=4)
    mask = ants.get_mask(native_rs)
    seg = ants.kmeans_segmentation(native_rs, k=3, kmask=mask)
    thickness_native = ants.kelly_kapowski(
        s=seg["segmentation"], g=seg["probabilityimages"][1], w=seg["probabilityimages"][2],
        its=its, r=0.025, m=1.5,
    )

    t2_t1_img = ants.image_read(t2_t1_path)
    native_t1_full_img = ants.image_read(native_t1_brain_path)
    # Та же Affine-регистрация "полный T1 -> обрезанная T2-сетка", что и в
    # glasser_atlas_v2.py - пересчитана здесь (не кэширована между
    # модулями), т.к. это единственный шаг, нужный для переноса карты
    # толщины; дёшево относительно самого DiReCT.
    reg = ants.registration(fixed=t2_t1_img, moving=native_t1_full_img, type_of_transform="Affine")
    thickness_in_t2 = ants.apply_transforms(
        fixed=t2_t1_img, moving=thickness_native, transformlist=reg["fwdtransforms"], interpolator="linear"
    )

    os.makedirs(cache_dir, exist_ok=True)
    ants.image_write(thickness_in_t2, out_path)
    return thickness_in_t2.numpy().astype(np.float32)
