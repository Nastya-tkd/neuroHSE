"""
Скачивает карты R2' для базового (control-условие) уровня, восстановленные
тем же механизмом истории версий S3, что и всё остальное.

R2' - последняя из физиологических карт исходного qBOLD-конвейера
(R2'/CBV/CBF/OEF), ещё не испытанная в этом проекте как вход (CBF, OEF и
CBV уже проверены по отдельности, см. scripts/run_cbf_oef.py,
scripts/run_cbv.py). Физиологически R2' отражает содержание
дезоксигемоглобина/венозную оксигенацию - величина, из которой вместе с
CBV сам qBOLD-конвейер вычисляет OEF, формально отличная от уже
проверенных CBF (поток), OEF (экстракция) и CBV (объём).

Та же логика отсутствия цикличности, что и в download_cbf_oef.py/
download_cbv.py: используется только R2' *control*-условия (базового
уровня), никогда условия задачи - метка concordant/discordant определяется
изменением между задачей и control, поэтому одна лишь базовая карта не
участвует в этом вычислении.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR


def download_subject_r2prime(subject, condition="control", versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    suffix = f"_task-{condition}_space-T2_R2prime.nii"
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
        p = download_subject_r2prime(sub, "control")
        print(f"{sub}: {p}")
