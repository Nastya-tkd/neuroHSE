"""
Downloads the baseline (control-condition) CBF and OEF maps found
alongside CMRO2 in the same derivatives tree, recovered via the same S3
version-history mechanism as everything else (see download_real_labels.py).

These were not part of the original label-defining file set - CMRO2 was
already precomputed and recoverable directly, so CBF/OEF were never
needed before. They matter now as a genuinely different kind of
"structural" input: physiological baseline maps (how much blood/oxygen a
voxel receives at rest) rather than T1 anatomical intensity. Using only
the *control*-condition (baseline) values, not the task-condition ones,
keeps this non-circular - the concordant/discordant label is defined by
the *change* between task and control, not by either condition's raw
value alone.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR

CBF_OEF_SUFFIXES = [
    "_task-control_space-T2_cbf.nii",
    "_task-control_space-T2_oef.nii",
]


def download_subject_cbf_oef(subject, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)

    downloaded = {}
    for suffix in CBF_OEF_SUFFIXES:
        matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
        if not matches:
            print(f"  [!] no match for {subject}{suffix}")
            continue
        key = matches[0]
        version_id, last_modified = version_map[key]
        fname = key.split("/")[-1]
        out_path = os.path.join(out_dir, fname)
        download_versioned(key, version_id, out_path)
        downloaded[suffix] = out_path
        print(f"  {fname} ({last_modified})")

    return downloaded


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019"]
    for sub in subjects:
        print(f"=== {sub} ===")
        download_subject_cbf_oef(sub)
