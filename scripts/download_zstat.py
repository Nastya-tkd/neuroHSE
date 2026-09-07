"""
Downloads the real per-voxel first-level GLM Z-statistic map for the
BOLD activation contrast (calc>control / mem>control), recovered the
same way as everything else via S3 version history.

Found by reading the source pipeline's own analysis notebooks
(github.com/NeuroenergeticsLab/two_modes_of_hemodynamics,
D_Fig2C_native_space_analysis.ipynb / Replication_data_analyses.ipynb):
`{sub}_1stlevel_{contrast}control_space-T2.nii.gz` is the first-level
FSL z-statistic for that exact contrast, thresholded at z=2.5 in the
source pipeline's own ROI-definition step (`z_thr=2.5`). This is a
genuine per-voxel statistical-confidence map, not the |CMRO2_percchange|
magnitude proxy used in scripts/run_reliability_filtered.py - it
directly answers "how much can this specific voxel's activation
estimate be trusted", the actual quantity Buchel et al. (2026) argue
drives most of the apparent concordant/discordant "noise".

Caveat, disclosed rather than glossed over: this Z-statistic is for the
BOLD side of the contrast only (fMRI activation), not a joint BOLD+CMRO2
uncertainty estimate - the source pipeline itself reuses this same map
as the shared significance gate for CBF/OEF/CMRO2 ROI analyses too (see
the notebook: qBOLD-derived quantities are masked with this same
z-thresholded BOLD map, not their own independent statistic), so this is
the closest genuine, non-proxy reliability signal available in this
dataset for either variable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR


def download_subject_zstat(subject, contrast, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    """contrast: 'calc' or 'mem'. Returns the local path, or None if this
    subject/contrast doesn't have this file (matches the same coverage
    gaps as everything else - not every subject has every contrast)."""
    suffix = f"_1stlevel_{contrast}control_space-T2.nii.gz"
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)

    matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
    if not matches:
        return None
    key = matches[0]
    version_id, last_modified = version_map[key]
    fname = key.split("/")[-1]
    out_path = os.path.join(out_dir, fname)
    download_versioned(key, version_id, out_path)
    return out_path


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019"]
    for sub in subjects:
        for contrast in ["calc", "mem"]:
            p = download_subject_zstat(sub, contrast)
            print(f"{sub} {contrast}: {p}")
