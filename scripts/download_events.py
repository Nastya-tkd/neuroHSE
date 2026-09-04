"""
Downloads <sub>_task-all_events.tsv (task block timing: onset, duration,
trial_type) via S3 version-history recovery, same approach as the other
download_*.py scripts - but from the RAW dataset prefix (ds004873/sub-pXXX/),
not derivatives/, since events.tsv lives with the raw BOLD data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR

SUFFIX = "_task-all_events.tsv"


def download_subject_events(subject, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    cache_path = os.path.join(versions_cache_dir, f"{subject}_raw.json")
    version_map = get_or_build_version_map(f"ds004873/{subject}/", cache_path)

    key = f"ds004873/{subject}/func/{subject}{SUFFIX}"
    if key not in version_map:
        raise FileNotFoundError(f"{key} not found in version history for {subject}")
    version_id, last_modified = version_map[key]

    out_dir = os.path.join(data_dir, subject, "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject}{SUFFIX}")
    download_versioned(key, version_id, out_path)
    return out_path


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019"]
    for sub in subjects:
        path = download_subject_events(sub)
        print(f"{sub}: {path}")
