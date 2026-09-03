"""
Runs the real (ported) T2-mapping step (src/qbold.py) on the real 8-echo
MESE data for the given subjects and saves the resulting T2/amplitude/error
maps under data/<subject>/qmri/. This is real, verified output - not a
placeholder - but see src/qbold.py's module docstring for exactly what it
is (T2 only) and is not (full R2'/OEF/CMRO2, which need data we don't have).
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
            f"{sub}: T2 map -> {t2_path} "
            f"(median T2 = {np.median(brain_vals):.1f} ms, "
            f"n={brain_vals.size} voxels fit)"
        )
