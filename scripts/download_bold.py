"""
Downloads <sub>_task-all_space-T2_filtered_func.nii.gz (FSL FEAT fully
preprocessed BOLD - motion correction, smoothing, temporal high-pass filter
already applied) for Experiment 2, via the same S3 version-history recovery
as scripts/download_real_labels.py. ~270-430MB per subject.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR

SUFFIX = "_task-all_space-T2_filtered_func.nii.gz"


def download_subject_bold(subject, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)

    key = f"ds004873/derivatives/{subject}/func/{subject}{SUFFIX}"
    if key not in version_map:
        raise FileNotFoundError(f"{key} not found in version history for {subject}")
    version_id, last_modified = version_map[key]

    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject}{SUFFIX}")
    download_versioned(key, version_id, out_path)
    print(f"{subject}: {out_path} ({last_modified})")
    return out_path


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]
    for sub in subjects:
        download_subject_bold(sub)
