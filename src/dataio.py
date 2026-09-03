"""
Minimal I/O helpers: load NIfTI volumes as numpy arrays, and build the
derivative file paths used by the two_modes_of_hemodynamics pipeline
(BIDS-like naming: sub-pXXX, task-<cond>, space-<space>, desc-<desc>).
"""

import os
import numpy as np
import nibabel as nib


def load_nifti(path):
    """Returns (data as float32 ndarray, affine, nibabel image)."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine, img


def sub_id(numeric_id):
    """021 -> 'sub-p021', already-prefixed strings are passed through."""
    if isinstance(numeric_id, str) and numeric_id.startswith("sub-"):
        return numeric_id
    return "sub-p{:03d}".format(int(numeric_id))


class SubjectPaths:
    """
    Builds derivative file paths for one subject, matching the naming
    convention in https://github.com/NeuroenergeticsLab/two_modes_of_hemodynamics
    (see combined_pipeline.py, e.g. lines 10092-10109).

    derivatives_dir/
      sub-pXXX/anat/sub-pXXX_desc-fmriprep_T1w{_brain}.nii.gz
      sub-pXXX/func/sub-pXXX_task-<cond><baseline>_space-<space>_desc-fmriprep_BOLD_percchange.nii.gz
      sub-pXXX/qmri/sub-pXXX_task-<cond>_space-<space>_desc-orig_cmro2.nii.gz
    """

    def __init__(self, derivatives_dir, subject, space="T2"):
        self.sub = sub_id(subject)
        self.space = space
        self.anat_dir = os.path.join(derivatives_dir, self.sub, "anat")
        self.func_dir = os.path.join(derivatives_dir, self.sub, "func")
        self.qmri_dir = os.path.join(derivatives_dir, self.sub, "qmri")

    def t1w(self, brain_extracted=True):
        suffix = "_brain" if brain_extracted else ""
        return os.path.join(self.anat_dir, f"{self.sub}_desc-fmriprep_T1w{suffix}.nii.gz")

    def brain_mask(self):
        return os.path.join(self.anat_dir, f"{self.sub}_desc-fmriprep_brain_mask.nii.gz")

    def bold_percchange(self, task, baseline="control"):
        return os.path.join(
            self.func_dir,
            f"{self.sub}_task-{task}{baseline}_space-{self.space}_desc-fmriprep_BOLD_percchange.nii.gz",
        )

    def cmro2(self, condition, desc="orig"):
        return os.path.join(
            self.qmri_dir,
            f"{self.sub}_task-{condition}_space-{self.space}_desc-{desc}_cmro2.nii.gz",
        )
