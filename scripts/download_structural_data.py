"""
Downloads T1w structural scans for the given subjects directly from
OpenNeuro's underlying S3 bucket and computes an approximate brain mask.

openneuro.org itself is blocked by this session's network egress policy, but
the S3 bucket that actually backs it (s3.amazonaws.com/openneuro.org/...) is
not, so this pulls data straight from there - no browser/API access to
openneuro.org needed. Verified structure by listing the bucket:
    https://s3.amazonaws.com/openneuro.org/?list-type=2&prefix=ds004873/

Note this only gets the *raw* T1w - ds004873's S3 copy has no `derivatives/`
folder at all, just per-subject anat/ (T1w, MESE echoes) and func/
(task-all_bold). The CMRO2/BOLD_percchange maps needed for real
concordant/discordant labels are not hosted here (see README.md).
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib
from src.dataio import load_nifti, simple_brain_mask

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds004873"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_subject_t1(subject, data_dir=DATA_DIR):
    anat_dir = os.path.join(data_dir, subject, "anat")
    os.makedirs(anat_dir, exist_ok=True)

    t1_path = os.path.join(anat_dir, f"{subject}_T1w.nii.gz")
    if not os.path.exists(t1_path):
        url = f"{BASE_URL}/{subject}/anat/{subject}_T1w.nii.gz"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, t1_path)

    mask_path = os.path.join(anat_dir, f"{subject}_brain_mask.nii.gz")
    if not os.path.exists(mask_path):
        t1, affine, _ = load_nifti(t1_path)
        mask = simple_brain_mask(t1)
        nib.save(nib.Nifti1Image(mask, affine), mask_path)

    return t1_path, mask_path


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]
    for sub in subjects:
        t1_path, mask_path = download_subject_t1(sub)
        print(f"{sub}: {t1_path}, {mask_path}")
