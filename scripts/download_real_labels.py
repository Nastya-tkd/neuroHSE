"""
Downloads the real files needed for the real concordant/discordant label,
recovered from OpenNeuro's S3 object-version history for ds004873
(CC0-licensed). The dataset's current top-level listing only shows raw
T1w/MESE/BOLD (see README.md "Current data status"), but earlier snapshots
of the same public, CC0 dataset included a full `derivatives/` tree with
exactly the qmri/func outputs the source pipeline (combined_pipeline.py)
expects. S3 keeps old object versions even after a key is "deleted"
(a delete marker, not an erasure) - `list_versions.py` walks that history
and this script fetches, for each needed file, the most recent version
that still has real content, via `?versionId=...`.

For each subject, downloads:
  - <sub>_space-T2_desc-brain_T1w.nii.gz   (structural, same space as labels)
  - <sub>_task-calccontrol_space-T2_BOLD_percchange.nii.gz
  - <sub>_task-memcontrol_space-T2_BOLD_percchange.nii.gz
  - <sub>_task-{calc,control,mem}_space-T2_desc-orig_cmro2.nii
  - <sub>_BrMsk_CSF_30slices.nii.gz         (brain mask, same space)
"""

import os
import sys
import json
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map

S3_BASE = "https://s3.amazonaws.com/openneuro.org/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VERSIONS_CACHE_DIR = os.path.join(DATA_DIR, ".versions_cache")

NEEDED_SUFFIXES = [
    "_space-T2_desc-brain_T1w.nii.gz",
    "_task-calccontrol_space-T2_BOLD_percchange.nii.gz",
    "_task-memcontrol_space-T2_BOLD_percchange.nii.gz",
    "_task-calc_space-T2_desc-orig_cmro2.nii",
    "_task-control_space-T2_desc-orig_cmro2.nii",
    "_task-mem_space-T2_desc-orig_cmro2.nii",
    "_BrMsk_CSF_30slices.nii.gz",
]


def download_versioned(key, version_id, out_path):
    if os.path.exists(out_path):
        return out_path
    url = S3_BASE + key.replace(" ", "%20") + f"?versionId={version_id}"
    urllib.request.urlretrieve(url, out_path)
    return out_path


def download_subject_labels(subject, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)

    downloaded = {}
    for suffix in NEEDED_SUFFIXES:
        matches = [k for k in version_map if k.endswith(subject + suffix) or k.endswith(suffix) and subject in k]
        matches = [k for k in matches if k.split("/")[-1].startswith(subject)]
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
    subjects = sys.argv[1:] or ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]
    for sub in subjects:
        print(f"=== {sub} ===")
        download_subject_labels(sub)
