"""
Robust per-subject loader covering both derivatives naming schemes found
in ds004873: the original 25-subject cohort (sub-p019...sub-p055) and the
later sub-p058...sub-p068 cohort, which uses different conventions:

  - T1w: sub-pXXX...58-68 have no pre-skull-stripped `desc-brain_T1w` in
    T2 space, only the whole-head `space-T2_T1w.nii` - skull-stripped here
    using the subject's own brain mask instead.
  - Brain mask: `BrMsk_CSF_30slices.nii.gz` -> falls back to `BrMsk_CSF.nii`.
  - CMRO2 for task conditions (calc/mem): `desc-orig_cmro2` -> falls back
    to `desc-CBV_cmro2` (a CBV-corrected variant). This mixing (orig for
    the control/baseline condition, CBV-corrected for the task condition)
    is not improvised - it is the source pipeline's own convention
    (combined_pipeline.py: CMRO2_mode=='corrected' uses the CBV-corrected
    map for task but always desc-orig for baseline; desc-CBV has no
    baseline/control counterpart anywhere in the dataset).

A missing BOLD_percchange for one specific contrast (seen for sub-p066:
no memcontrol at all, only calccontrol) skips that contrast only, not the
whole subject - handled by the caller checking which contrasts came back.
"""

import os
import numpy as np

from src.dataio import load_nifti
from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR

T1W_CANDIDATES = ["_space-T2_desc-brain_T1w.nii.gz", "_space-T2_T1w.nii"]
MASK_CANDIDATES = ["_BrMsk_CSF_30slices.nii.gz", "_BrMsk_CSF.nii"]
CMRO2_TASK_CANDIDATES = ["_space-T2_desc-orig_cmro2.nii", "_space-T2_desc-CBV_cmro2.nii"]
CMRO2_CONTROL_CANDIDATES = ["_task-control_space-T2_desc-orig_cmro2.nii"]  # never has a CBV counterpart in this dataset


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
    Returns (t1, affine, mask, cmro2_dict, bold_pct_dict, notes) or None if
    the subject is missing something unrecoverable (T1w or mask or a
    control CMRO2 - without those nothing at all is usable).
    cmro2_dict / bold_pct_dict only contain the conditions/contrasts that
    were actually found - caller checks which contrasts are complete.
    notes: list of human-readable strings describing any fallback used,
    for the run log.
    """
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)
    notes = []

    t1_path, t1_suffix = _download_first_match(version_map, subject, T1W_CANDIDATES, out_dir)
    mask_path, mask_suffix = _download_first_match(version_map, subject, MASK_CANDIDATES, out_dir)
    if t1_path is None or mask_path is None:
        return None, [f"missing T1w and/or brain mask entirely"]

    t1, affine, _ = load_nifti(t1_path)
    mask_raw, _, _ = load_nifti(mask_path)
    mask = (mask_raw > 0.5).astype(np.uint8)

    if t1_suffix == "_space-T2_T1w.nii":
        t1 = t1 * mask.astype(t1.dtype)  # whole-head file: skull-strip ourselves
        notes.append("T1w: whole-head file, skull-stripped with the subject's own brain mask")
    if mask_suffix == "_BrMsk_CSF.nii":
        notes.append("brain mask: BrMsk_CSF.nii (no _30slices variant for this subject)")

    control_path, control_suffix = _download_first_match(version_map, subject, CMRO2_CONTROL_CANDIDATES, out_dir)
    if control_path is None:
        return None, notes + ["missing control-condition CMRO2 entirely"]
    cmro2 = {"control": load_nifti(control_path)[0].squeeze()}

    bold_pct = {}
    for cond in ["calc", "mem"]:
        cand = [f"_task-{cond}{s}" for s in CMRO2_TASK_CANDIDATES]
        task_path, task_suffix = _download_first_match(version_map, subject, cand, out_dir)
        if task_path is not None:
            cmro2[cond] = load_nifti(task_path)[0].squeeze()
            if task_suffix and "CBV" in task_suffix:
                notes.append(f"CMRO2 task-{cond}: desc-CBV (CBV-corrected) fallback, no desc-orig available")

        bold_key = f"_task-{cond}control_space-T2_BOLD_percchange.nii.gz"
        bold_path, _ = _download_first_match(version_map, subject, [bold_key], out_dir)
        if bold_path is not None:
            bold_pct[cond] = load_nifti(bold_path)[0]
        else:
            notes.append(f"BOLD_percchange task-{cond}control: not found, contrast '{cond}' unavailable")

    return (t1, affine, mask, cmro2, bold_pct), notes
