"""
Скачивает реальные файлы, необходимые для реальной метки
concordant/discordant, восстановленные из истории версий объектов S3
OpenNeuro для ds004873 (лицензия CC0). Текущий листинг верхнего уровня
датасета показывает только исходные T1w/MESE/BOLD (см. README.md "Current
data status"), но более ранние снимки того же публичного датасета с
лицензией CC0 включали полное дерево `derivatives/` ровно с теми выходами
qmri/func, которые ожидает исходный конвейер (combined_pipeline.py). S3
хранит старые версии объекта даже после "удаления" ключа (маркер удаления,
а не стирание) - `list_versions.py` проходит по этой истории, и этот
скрипт для каждого нужного файла загружает самую свежую версию, в которой
ещё есть реальное содержимое, через `?versionId=...`.

Для каждого пациента скачивается:
  - <sub>_space-T2_desc-brain_T1w.nii.gz   (структурный снимок, то же пространство, что и метки)
  - <sub>_task-calccontrol_space-T2_BOLD_percchange.nii.gz
  - <sub>_task-memcontrol_space-T2_BOLD_percchange.nii.gz
  - <sub>_task-{calc,control,mem}_space-T2_desc-orig_cmro2.nii
  - <sub>_BrMsk_CSF_30slices.nii.gz         (маска мозга, то же пространство)
"""

import os
import sys
import json
import time
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


def download_versioned(key, version_id, out_path, retries=4):
    """Сначала скачивает во временный файл .part и переименовывает его в
    out_path только после полной, проверенной передачи - чтобы неудачная
    или прерванная попытка (большие загрузки filtered_func иногда обрываются
    посередине) никогда не оставляла повреждённый файл там, где проверка
    "уже скачано, пропуск" выше доверилась бы ему при следующем запуске.
    При временных сбоях повторяет попытку с задержкой."""
    if os.path.exists(out_path):
        return out_path
    url = S3_BASE + key.replace(" ", "%20") + f"?versionId={version_id}"
    tmp_path = out_path + ".part"

    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url) as resp:
                expected = resp.headers.get("Content-Length")
                expected = int(expected) if expected else None
                with open(tmp_path, "wb") as f:
                    written = 0
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
            if expected is not None and written != expected:
                raise IOError(f"неполная загрузка: получено {written} из {expected} байт")
            os.replace(tmp_path, out_path)
            return out_path
        except Exception as e:
            last_err = e
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_err


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
            print(f"  [!] совпадение для {subject}{suffix} не найдено")
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
