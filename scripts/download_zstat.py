"""
Скачивает реальную повоксельную карту Z-статистики GLM первого уровня для
контраста активации BOLD (calc>control / mem>control), восстановленную тем
же способом, что и всё остальное, через историю версий S3.

Обнаружено при чтении собственных аналитических ноутбуков исходного
конвейера (github.com/NeuroenergeticsLab/two_modes_of_hemodynamics,
D_Fig2C_native_space_analysis.ipynb / Replication_data_analyses.ipynb):
`{sub}_1stlevel_{contrast}control_space-T2.nii.gz` - это z-статистика FSL
первого уровня для данного конкретного контраста, пороговая на z=2.5 на
собственном шаге определения ROI исходного конвейера (`z_thr=2.5`). Это
настоящая повоксельная карта статистической достоверности, а не proxy-
величина |CMRO2_percchange|, используемая в scripts/run_reliability_filtered.py
- она напрямую отвечает на вопрос "насколько можно доверять оценке
активации именно этого вокселя", ту самую величину, которая, по мнению
Buchel et al. (2026), определяет большую часть кажущегося "шума"
concordant/discordant.

Оговорка, а не умолчание: эта Z-статистика относится только к стороне BOLD
контраста (активация фМРТ), а не к совместной оценке неопределённости
BOLD+CMRO2 - сам исходный конвейер повторно использует эту же карту как
общий порог значимости и для ROI-анализов CBF/OEF/CMRO2 (см. ноутбук:
величины, производные от qBOLD, маскируются той же пороговой по z картой
BOLD, а не собственной независимой статистикой), так что это ближайший
настоящий, не-proxy сигнал надёжности, доступный в этом датасете для
любой из переменных.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR


def download_subject_zstat(subject, contrast, versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    """contrast: 'calc' или 'mem'. Возвращает локальный путь, либо None,
    если для этого пациента/контраста такого файла нет (те же пробелы в
    покрытии, что и везде - не у каждого пациента есть каждый контраст)."""
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
