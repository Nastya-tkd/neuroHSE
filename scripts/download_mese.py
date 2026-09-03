"""
Downloads the full 8-echo MESE series (+ JSON sidecars, for real EchoTime
values) for the given subjects from OpenNeuro's S3 bucket, same approach as
scripts/download_structural_data.py.
"""

import os
import sys
import json
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds004873"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_ECHOES = 8


def download_subject_mese(subject, data_dir=DATA_DIR):
    anat_dir = os.path.join(data_dir, subject, "anat")
    os.makedirs(anat_dir, exist_ok=True)

    echo_times_ms = []
    nii_paths = []
    for echo in range(1, N_ECHOES + 1):
        stem = f"{subject}_echo-{echo:02d}_MESE"
        nii_path = os.path.join(anat_dir, f"{stem}.nii.gz")
        json_path = os.path.join(anat_dir, f"{stem}.json")

        if not os.path.exists(nii_path):
            urllib.request.urlretrieve(f"{BASE_URL}/{subject}/anat/{stem}.nii.gz", nii_path)
        if not os.path.exists(json_path):
            urllib.request.urlretrieve(f"{BASE_URL}/{subject}/anat/{stem}.json", json_path)

        with open(json_path) as f:
            echo_times_ms.append(json.load(f)["EchoTime"] * 1000.0)
        nii_paths.append(nii_path)

    return nii_paths, echo_times_ms


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]
    for sub in subjects:
        nii_paths, tes = download_subject_mese(sub)
        print(f"{sub}: {len(nii_paths)} echoes, TEs(ms)={tes}")
