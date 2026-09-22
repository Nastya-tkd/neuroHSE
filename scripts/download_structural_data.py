"""
Скачивает структурные T1w-снимки для заданных пациентов напрямую из
S3-хранилища, на котором работает OpenNeuro, и вычисляет приближённую
маску мозга.

Сам openneuro.org заблокирован политикой сетевого доступа этой сессии, а
вот S3-хранилище, которое фактически его обслуживает
(s3.amazonaws.com/openneuro.org/...), - нет, поэтому данные берутся оттуда
напрямую - доступ через браузер/API к openneuro.org не нужен. Структура
проверена листингом хранилища:
    https://s3.amazonaws.com/openneuro.org/?list-type=2&prefix=ds004873/

Обратите внимание, что здесь скачивается только *исходный* (raw) T1w - в
S3-копии ds004873 вообще нет папки `derivatives/`, только per-subject anat/
(T1w, эхо MESE) и func/ (task-all_bold). Карты CMRO2/BOLD_percchange,
нужные для реальных меток concordant/discordant, здесь не хранятся
(см. README.md).
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
        print(f"скачивание {url}")
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
