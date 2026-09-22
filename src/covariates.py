"""
Пункт 3 из списка последующих задач: возраст/Hct/пол как ковариаты наряду
со структурным патчем. participants.tsv восстановлен через историю версий
S3 (см. docstring scripts/download_real_labels.py о том, почему в текущем
листинге датасета его нет напрямую).
"""

import csv


def load_participants(path):
    """Возвращает {subject: {"age": float, "hct": float, "sex": 0/1}} для
    строк с полными данными (автоматически исключает строки EXCLUDED /
    без значения Hct, те же критерии, что и в ALL_SUBJECTS из
    src/cohort.py)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pid = row["participant_id"]
            if not pid.startswith("sub-p"):
                continue
            age, hct, sex = row.get("age", "").strip(), row.get("Hct", "").strip(), row.get("sex", "").strip()
            if not age or not hct or not sex:
                continue
            out[pid] = {"age": float(age), "hct": float(hct), "sex": 1.0 if sex == "m" else 0.0}
    return out


def covariate_vector(subject, participants, age_mean, age_std, hct_mean, hct_std):
    """[age_z, hct_z, sex] на пациента - z-нормализация с использованием
    переданной статистики по когорте (вычисленной один раз по обучающей
    выборке, не повоксельно)."""
    row = participants[subject]
    return [(row["age"] - age_mean) / age_std, (row["hct"] - hct_mean) / hct_std, row["sex"]]
