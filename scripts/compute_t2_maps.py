"""
Выполняет реальный (портированный) шаг построения карты T2 (src/qbold.py)
на реальных 8-эховых данных MESE для заданных пациентов и сохраняет
полученные карты T2/амплитуды/ошибки в data/<subject>/qmri/. Это реальный,
проверенный результат - не заглушка - но точно, что это такое (только T2) и
чем это не является (полный R2'/OEF/CMRO2, для которых нужны данные,
которых у нас нет), см. в докстринге модуля src/qbold.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import nibabel as nib

from src.dataio import load_nifti, simple_brain_mask
from src.qbold import fit_t2_map
from scripts.download_mese import download_subject_mese

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def compute_subject_t2(subject, data_dir=DATA_DIR):
    nii_paths, echo_times_ms = download_subject_mese(subject, data_dir)
    volumes = []
    affine = None
    for p in nii_paths:
        vol, aff, _ = load_nifti(p)
        volumes.append(vol)
        affine = aff
    echo_volumes = np.stack(volumes, axis=0)

    mask = simple_brain_mask(echo_volumes[0])
    t2_map, amp_map, err_map = fit_t2_map(echo_volumes, echo_times_ms, mask=mask)

    qmri_dir = os.path.join(data_dir, subject, "qmri")
    os.makedirs(qmri_dir, exist_ok=True)
    t2_path = os.path.join(qmri_dir, f"{subject}_space-T2_T2map.nii.gz")
    err_path = os.path.join(qmri_dir, f"{subject}_space-T2_T2map_error.nii.gz")
    nib.save(nib.Nifti1Image(t2_map.astype(np.float32), affine), t2_path)
    nib.save(nib.Nifti1Image(err_map.astype(np.float32), affine), err_path)

    brain_vals = t2_map[mask.astype(bool)]
    brain_vals = brain_vals[brain_vals > 0]
    return t2_path, err_path, brain_vals


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]
    for sub in subjects:
        t2_path, err_path, brain_vals = compute_subject_t2(sub)
        print(
            f"{sub}: карта T2 -> {t2_path} "
            f"(медианный T2 = {np.median(brain_vals):.1f} мс, "
            f"n={brain_vals.size} вокселей подогнано)"
        )
