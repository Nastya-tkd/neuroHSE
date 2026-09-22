"""
SMOKE-ТЕСТ ПО ВСЕМ 5 ПАЦИЕНТАМ - НЕ НАУЧНЫЙ РЕЗУЛЬТАТ.

Та же оговорка, что и в scripts/smoke_test.py, распространённая на всех 5
пациентов, для которых у нас есть реальный T1w. Метки по-прежнему остаются
синтетическим плейсхолдером по порогу интенсивности T1 (без биологического
смысла), потому что реальные метки concordant/discordant всё ещё
заблокированы отсутствием данных CBF/CBV/Hct (см. README.md).

Этот скрипт существует, чтобы показать, что конвейер - не случайная удача
на одном пациенте: он корректно проходит от начала до конца на каждом
реальном пациенте - и чтобы построить сводный график по всем пациентам.
Он НЕ показывает, что модель способна обнаруживать что-либо о вокселях
concordant/discordant, по причинам, объяснённым в README.md и в переписке:
без реальной метки здесь нет ничего биологического, что можно было бы
обнаружить.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.dataio import load_nifti
from src.train import run_hemisphere_experiment
from scripts.smoke_test import make_synthetic_label

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "smoke_test_all")

SUBJECTS = ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for sub in SUBJECTS:
        t1_path = os.path.join(DATA_DIR, sub, "anat", f"{sub}_T1w.nii.gz")
        mask_path = os.path.join(DATA_DIR, sub, "anat", f"{sub}_brain_mask.nii.gz")

        t1, affine, _ = load_nifti(t1_path)
        mask, _, _ = load_nifti(mask_path)
        label = make_synthetic_label(t1, mask, seed=hash(sub) % (2**31))

        print(f"=== {sub}: SMOKE-ТЕСТ (синтетическая метка) ===")
        results = run_hemisphere_experiment(
            t1_volume=t1,
            label_volume=label,
            brain_mask=mask,
            affine=affine,
            patch_size=9,
            axis_index=0,
            max_voxels_per_side=600,
            epochs=8,
            out_dir=os.path.join(OUT_DIR, sub),
            subject_name=f"{sub}_SMOKETEST",
        )
        all_results[sub] = results
        print(sub, results)

    # сводка по всем пациентам
    fig, ax = plt.subplots(figsize=(8, 4))
    subs = list(all_results.keys())
    a_vals = [all_results[s]["A_train_B_test"] for s in subs]
    b_vals = [all_results[s]["B_train_A_test"] for s in subs]
    x = np.arange(len(subs))
    width = 0.35
    ax.bar(x - width / 2, a_vals, width, label="обучение A / тест B", color="#8e44ad")
    ax.bar(x + width / 2, b_vals, width, label="обучение B / тест A", color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайный уровень")
    ax.set_xticks(x)
    ax.set_xticklabels(subs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("SMOKE-ТЕСТ (синтетические метки, не биологические) - все 5 пациентов")
    ax.legend(fontsize=8)
    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, "cross_subject_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nСводка по всем пациентам сохранена в {summary_path}")


if __name__ == "__main__":
    main()
