"""
Скачивает полный, непорезанный T1 (fMRIPrep anat/, нативное пространство,
~170x256x256, ~1мм) - найден в derivatives каждого пациента, но никогда не
использовавшийся в проекте, поскольку метки CMRO2/BOLD определены только в
обрезанной до 30 срезов (99мм) сетке space-T2. Используется как
промежуточная цель для улучшенной регистрации атласа Glasser
(src/glasser_atlas_v2.py) - полный T1 даёт куда лучше обусловленную
регистрацию к шаблону MNI, чем прямая регистрация MNI на обрезанную
30-срезовую сетку.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR


def download_subject_native_t1(subject, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    suffix = f"_desc-fmriprep_T1w.nii.gz"
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
        p = download_subject_native_t1(sub)
        print(f"{sub}: {p}")
