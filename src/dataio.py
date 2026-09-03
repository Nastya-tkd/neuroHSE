"""
Minimal I/O helpers: load NIfTI volumes as numpy arrays, and build the
derivative file paths used by the two_modes_of_hemodynamics pipeline
(BIDS-like naming: sub-pXXX, task-<cond>, space-<space>, desc-<desc>).
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage


def simple_brain_mask(t1_volume):
    """
    Approximate brain mask via Otsu thresholding + largest connected
    component + hole filling. Used only for subjects where we have a raw
    (not already skull-stripped) T1w and no FSL/nilearn BET available.
    Good enough to restrict patch centers to brain tissue; not a substitute
    for a real skull-stripping tool if precise face/skull removal matters.
    """
    nz = t1_volume[t1_volume > 0]
    hist, bin_edges = np.histogram(nz, bins=256)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.clip(weight1, 1e-12, None)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.clip(weight2[::-1], 1e-12, None))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    threshold = bin_centers[np.argmax(variance12)]

    mask = t1_volume > threshold
    labeled, n = ndimage.label(mask)
    if n > 0:
        largest = 1 + np.argmax(ndimage.sum(mask, labeled, range(1, n + 1)))
        mask = labeled == largest
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(np.uint8)


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
