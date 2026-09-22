"""
Устойчивый загрузчик по пациентам, покрывающий обе схемы именования
производных данных, встречающиеся в ds004873: исходную когорту из 25
пациентов (sub-p019...sub-p055) и более позднюю когорту
sub-p058...sub-p068, использующую другие соглашения:

  - T1w: у sub-pXXX...58-68 нет заранее очищенного от черепа
    `desc-brain_T1w` в пространстве T2, только файл всей головы
    `space-T2_T1w.nii` - здесь он очищается от черепа с помощью
    собственной маски мозга пациента.
  - Маска мозга: `BrMsk_CSF_30slices.nii.gz` -> при отсутствии используется
    `BrMsk_CSF.nii`.
  - CMRO2 для условий задачи (calc/mem): `desc-orig_cmro2` -> при
    отсутствии используется `desc-CBV_cmro2` (вариант с CBV-коррекцией).
    Это смешение (orig для условия control/базового уровня,
    CBV-скорректированный для условия задачи) не придумано нами - это
    собственное соглашение исходного конвейера (combined_pipeline.py:
    CMRO2_mode=='corrected' использует CBV-скорректированную карту для
    задачи, но всегда desc-orig для базового уровня; у desc-CBV нигде в
    датасете нет пары для baseline/control).

Отсутствующий BOLD_percchange для одного конкретного контраста (замечено у
sub-p066: вообще нет memcontrol, только calccontrol) пропускает только этот
контраст, а не всего пациента - обрабатывается вызывающим кодом, который
проверяет, какие контрасты вернулись.
"""

import os
import numpy as np

from src.dataio import load_nifti
from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR

T1W_CANDIDATES = ["_space-T2_desc-brain_T1w.nii.gz", "_space-T2_T1w.nii"]
MASK_CANDIDATES = ["_BrMsk_CSF_30slices.nii.gz", "_BrMsk_CSF.nii"]
CMRO2_TASK_CANDIDATES = ["_space-T2_desc-orig_cmro2.nii", "_space-T2_desc-CBV_cmro2.nii"]
CMRO2_CONTROL_CANDIDATES = ["_task-control_space-T2_desc-orig_cmro2.nii"]  # в этом датасете никогда не имеет CBV-аналога


def _find_key(version_map, subject, suffix):
    matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
    return matches[0] if matches else None


def _download_first_match(version_map, subject, candidates, out_dir):
    for suffix in candidates:
        key = _find_key(version_map, subject, suffix)
        if key is None:
            continue
        version_id, _ = version_map[key]
        fname = key.split("/")[-1]
        out_path = os.path.join(out_dir, fname)
        download_versioned(key, version_id, out_path)
        return out_path, suffix
    return None, None


def load_subject_robust(subject, data_dir=DATA_DIR, versions_cache_dir=VERSIONS_CACHE_DIR):
    """
    Возвращает (t1, affine, mask, cmro2_dict, bold_pct_dict, notes) или None,
    если у пациента отсутствует что-то невосполнимое (T1w, или маска, или
    control CMRO2 - без них вообще ничего нельзя использовать).
    cmro2_dict / bold_pct_dict содержат только те условия/контрасты,
    которые действительно были найдены - вызывающий код проверяет, какие
    контрасты полные.
    notes: список человекочитаемых строк, описывающих любые использованные
    запасные варианты, для лога запуска.
    """
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)
    notes = []

    t1_path, t1_suffix = _download_first_match(version_map, subject, T1W_CANDIDATES, out_dir)
    mask_path, mask_suffix = _download_first_match(version_map, subject, MASK_CANDIDATES, out_dir)
    if t1_path is None or mask_path is None:
        return None, [f"полностью отсутствует T1w и/или маска мозга"]

    t1, affine, _ = load_nifti(t1_path)
    mask_raw, _, _ = load_nifti(mask_path)
    mask = (mask_raw > 0.5).astype(np.uint8)

    if t1_suffix == "_space-T2_T1w.nii":
        t1 = t1 * mask.astype(t1.dtype)  # файл всей головы: сами удаляем череп
        notes.append("T1w: файл всей головы, череп удалён с помощью собственной маски мозга пациента")
    if mask_suffix == "_BrMsk_CSF.nii":
        notes.append("маска мозга: BrMsk_CSF.nii (у этого пациента нет варианта _30slices)")

    control_path, control_suffix = _download_first_match(version_map, subject, CMRO2_CONTROL_CANDIDATES, out_dir)
    if control_path is None:
        return None, notes + ["полностью отсутствует CMRO2 для условия control"]
    cmro2 = {"control": load_nifti(control_path)[0].squeeze()}

    bold_pct = {}
    for cond in ["calc", "mem"]:
        cand = [f"_task-{cond}{s}" for s in CMRO2_TASK_CANDIDATES]
        task_path, task_suffix = _download_first_match(version_map, subject, cand, out_dir)
        if task_path is not None:
            cmro2[cond] = load_nifti(task_path)[0].squeeze()
            if task_suffix and "CBV" in task_suffix:
                notes.append(f"CMRO2 task-{cond}: запасной вариант desc-CBV (CBV-скорректированный), desc-orig недоступен")

        bold_key = f"_task-{cond}control_space-T2_BOLD_percchange.nii.gz"
        bold_path, _ = _download_first_match(version_map, subject, [bold_key], out_dir)
        if bold_path is not None:
            bold_pct[cond] = load_nifti(bold_path)[0]
        else:
            notes.append(f"BOLD_percchange task-{cond}control: не найден, контраст '{cond}' недоступен")

    return (t1, affine, mask, cmro2, bold_pct), notes
