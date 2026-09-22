"""
Скачивает карты CBV (объём мозгового кровотока) для базового
(control-условие) уровня, восстановленные тем же механизмом истории версий
S3, что и всё остальное.

Мотивировано напрямую гипотезой Алексея Осадчего (переданной через
пользователя): морфология/плотность капилляров определяет локальные
свойства доставки кислорода в ткани, и это может оставлять реальный,
физически обоснованный след во временах релаксации МРТ за счёт
внутривоксельного усреднения по компартментам (T1/T2 крови и ткани
различаются, поэтому доля объёма крови в вокселе измеримо сдвигает
наблюдаемую релаксацию - тот же физический механизм, на который опирается
сама визуализация BOLD/DSC), даже несмотря на то, что отдельные капилляры
намного меньше разрешения вокселя. Из карт, использованных в этом проекте,
CBV (объём крови) - более прямой макроскопический proxy для плотности
капилляров/сосудистой морфологии, чем CBF (поток) или один T1 - до сих пор
не проверенный.

Та же логика избежания цикличности, что и в scripts/download_cbf_oef.py:
используется только CBV *control*-условия (базового уровня), никогда
значения условия задачи - метка concordant/discordant определяется
изменением между задачей и базовым уровнем, поэтому одна лишь базовая
карта не участвует в этом вычислении.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR


def download_subject_cbv(subject, condition="control", versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    suffix = f"_task-{condition}_space-T2_cbv.nii"
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
        p = download_subject_cbv(sub, "control")
        print(f"{sub}: {p}")
