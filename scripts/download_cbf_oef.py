"""
Downloads CBF and OEF maps found alongside CMRO2 in the same derivatives
tree, recovered via the same S3 version-history mechanism as everything
else (see download_real_labels.py).

These were not part of the original label-defining file set - CMRO2 was
already precomputed and recoverable directly, so CBF/OEF were never
needed before. They matter now as a genuinely different kind of
"structural" input: physiological maps (how much blood/oxygen a voxel
receives) rather than T1 anatomical intensity.

Subfolder note (found when adding task-condition support): CBF exists in
*two* derivatives subfolders per subject/condition - perf/ and qmri/ -
with different S3 version IDs and timestamps (perf/ from the original
2023 processing, qmri/ from a later 2026 re-run), while OEF only ever
exists in qmri/. Since CMRO2 = CBF x OEF x CaO2 was computed by the
pipeline from its own internal CBF, and OEF has no perf/ counterpart to
be consistent with, this now explicitly prefers qmri/ over perf/ so CBF
and OEF are always drawn from the same processing lineage - not left to
whichever happened to sort first in the version map (which is what an
earlier version of this function implicitly did, picking perf/ for CBF
while OEF necessarily came from qmri/, an unintentional pipeline
mismatch quietly fixed here).
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


def download_subject_cbf_oef(subject, condition="control", versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    """condition: 'control' (baseline, default) or 'calc'/'mem' (task-condition,
    used by run_task_cbf_oef.py for the dynamic/task-period variant)."""
    suffixes = [f"_task-{condition}_space-T2_cbf.nii", f"_task-{condition}_space-T2_oef.nii"]
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)

    downloaded = {}
    for suffix in suffixes:
        matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
        if not matches:
            print(f"  [!] no match for {subject}{suffix}")
            continue
        qmri_matches = [k for k in matches if "/qmri/" in k]
        key = qmri_matches[0] if qmri_matches else matches[0]
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
