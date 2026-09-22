"""
Регистрирует настоящую корковую парцелляцию Glasser/HCP-MMP1.0 (360
областей, по 180 на полушарие) в собственную сетку пространства T2 каждого
пациента через ANTsPy SyN-регистрацию к встроенному шаблону MNI152 T1 - не
нужен ни внешний хост атласа, ни программа регистрации, кроме двух пакетов
PyPI (antspyx, nilearn), оба доступны в этой сессии.

Замечание для честности, напрямую от сопровождающего самого атласа (см.
atlas_cache/ORIGIN.md и
https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL): HCP-MMP1.0 был
создан и валидирован для *поверхностной* регистрации (FreeSurfer +
Connectome Workbench); использование его через *объёмную* MNI-регистрацию,
как делает этот модуль, явно отмечено создателями атласа (со ссылкой на
Coalson, Van Essen & Glasser 2018, PNAS) как вносящее реальную неточность
границ - это грубое объёмное приближение Glasser, а не методологически
предпочтительная поверхностная версия. Также отметим, что атлас был
привязан к шаблону типа ICBM2009c, тогда как целью регистрации здесь
служит встроенный в nilearn шаблон MNI152 (другой, хотя и близкий вариант
MNI) - второй источник приближения. Указано как есть, а не преподнесено как
"тот самый" атлас.

Регистрация занимает ~5-10с на пациента (SyN, низкое разрешение шаблона
2мм) - достаточно дёшево, чтобы просто вычислять по пациенту и кэшировать
на диск, не стоит запекать заранее в репозиторий.
"""

import os
import numpy as np

ATLAS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "atlas_cache")
ATLAS_LABELS_PATH = os.path.join(ATLAS_CACHE_DIR, "MNI_Glasser_HCP_v1.0.nii.gz")
MNI_TEMPLATE_PATH = os.path.join(ATLAS_CACHE_DIR, "mni152_t1_2mm.nii.gz")

_mni_template = None
_atlas_image = None


def _ensure_mni_template():
    """Поставляется вместе с nilearn (без сетевого запроса) - кэшируется на диск один раз как обычный nifti."""
    if os.path.exists(MNI_TEMPLATE_PATH):
        return
    import nibabel as nib
    from nilearn import datasets
    os.makedirs(ATLAS_CACHE_DIR, exist_ok=True)
    mni_nib = datasets.load_mni152_template(resolution=2)
    nib.save(mni_nib, MNI_TEMPLATE_PATH)


def get_subject_glasser_atlas(subject, t1_path, cache_dir=ATLAS_CACHE_DIR):
    """
    Возвращает целочисленный массив (X,Y,Z) той же формы/сетки, что и T1
    пациента (space-T2), с метками parcel'ов Glasser (0=фон/не кора,
    1-180=области левого полушария, 1000-1180=области правого полушария).
    Кэшируется на диск для каждого пациента после первого вызова.
    """
    import ants

    out_path = os.path.join(cache_dir, f"{subject}_glasser_T2space.nii.gz")
    if os.path.exists(out_path):
        img = ants.image_read(out_path)
        return img.numpy().astype(np.int32)

    if not os.path.exists(ATLAS_LABELS_PATH):
        raise FileNotFoundError(
            f"{ATLAS_LABELS_PATH} отсутствует - сначала клонируйте github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL "
            "и скопируйте туда MNI_Glasser_HCP_v1.0.nii.gz."
        )
    _ensure_mni_template()

    subj_img = ants.image_read(t1_path)
    mni_img = ants.image_read(MNI_TEMPLATE_PATH)
    atlas_img = ants.image_read(ATLAS_LABELS_PATH)

    reg = ants.registration(fixed=subj_img, moving=mni_img, type_of_transform="SyN")
    warped = ants.apply_transforms(
        fixed=subj_img, moving=atlas_img, transformlist=reg["fwdtransforms"], interpolator="genericLabel"
    )
    os.makedirs(cache_dir, exist_ok=True)
    ants.image_write(warped, out_path)
    return warped.numpy().astype(np.int32)
