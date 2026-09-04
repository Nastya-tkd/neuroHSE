"""
Item 3 of the follow-up list: age/Hct/sex as covariates alongside the
structural patch. participants.tsv recovered via S3 version history (see
scripts/download_real_labels.py's docstring for why the current dataset
listing doesn't have it directly).
"""

import csv


def load_participants(path):
    """Returns {subject: {"age": float, "hct": float, "sex": 0/1}} for rows
    with complete data (excludes EXCLUDED / missing-Hct rows automatically,
    same criteria as src/cohort.py's ALL_SUBJECTS)."""
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
    """Per-subject [age_z, hct_z, sex] - z-scored using cohort-level stats
    passed in (computed once over the training set, not per-voxel)."""
    row = participants[subject]
    return [(row["age"] - age_mean) / age_std, (row["hct"] - hct_mean) / hct_std, row["sex"]]
