"""
Скачивает карты CBF и OEF, найденные рядом с CMRO2 в том же дереве
derivatives, восстановленные тем же механизмом истории версий S3, что и
всё остальное (см. download_real_labels.py).

Они не входили в исходный набор файлов, определяющих метки - CMRO2 уже был
предвычислен и восстанавливался напрямую, поэтому CBF/OEF раньше не
требовались. Теперь они важны как принципиально иной вид "структурного"
входа: физиологические карты (сколько крови/кислорода получает воксель), а
не анатомическая интенсивность T1.

Замечание про подпапки (обнаружено при добавлении поддержки условий
задачи): CBF существует в *двух* подпапках derivatives на пациента/условие
- perf/ и qmri/ - с разными ID версий S3 и временными метками (perf/ из
исходной обработки 2023 года, qmri/ из более позднего перерасчёта 2026
года), тогда как OEF всегда существует только в qmri/. Поскольку CMRO2 =
CBF x OEF x CaO2 вычислялся конвейером из его собственного внутреннего
CBF, а у OEF нет пары в perf/, с которой можно было бы соотноситься,
теперь здесь явно предпочитается qmri/ вместо perf/, чтобы CBF и OEF всегда
брались из одной и той же ветки обработки - а не из того, что случайно
оказалось раньше в карте версий (именно так неявно вела себя более ранняя
версия этой функции, выбирая perf/ для CBF, тогда как OEF обязательно
приходил из qmri/, - непреднамеренное рассогласование конвейера, тихо
исправленное здесь).
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
    """condition: 'control' (базовый уровень, по умолчанию) или 'calc'/'mem'
    (условие задачи, используется run_task_cbf_oef.py для динамического
    варианта в период выполнения задачи)."""
    suffixes = [f"_task-{condition}_space-T2_cbf.nii", f"_task-{condition}_space-T2_oef.nii"]
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)

    downloaded = {}
    for suffix in suffixes:
        matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
        if not matches:
            print(f"  [!] совпадение для {subject}{suffix} не найдено")
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
