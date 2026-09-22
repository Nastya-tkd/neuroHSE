"""
Эксперимент 2 (по плану научного руководителя): структурный патч +
обычный сигнал BOLD (filtered_func, без контрастного агента / без DSC) в
качестве входного временного ряда на воксель, наряду со структурными
патчами. Те же реальные метки CMRO2/BOLD_percchange, что и в
scripts/run_real_experiment.py.

Запускается только для пациентов/контрастов, для которых уже был выполнен
Эксперимент 1 (только структура), чтобы результаты были напрямую
сопоставимы.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

from src.dataio import load_nifti
from src.train import run_hemisphere_experiment_multimodal
from scripts.run_real_experiment import load_subject_arrays, build_label, SUBJECTS, CONTRASTS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_bold_4d(sub):
    path = os.path.join(DATA_DIR, sub, "derivatives", f"{sub}_task-all_space-T2_filtered_func.nii.gz")
    bold, affine, _ = load_nifti(path)
    return bold


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for sub in SUBJECTS:
        arrays = load_subject_arrays(sub)
        bold_4d = load_bold_4d(sub)
        print(f"{sub}: T1 {arrays['t1'].shape}, BOLD {bold_4d.shape}")

        for contrast in CONTRASTS:
            label = build_label(arrays, contrast)
            n_conc = int((label > 0).sum())
            n_disc = int((label < 0).sum())
            if min(n_conc, n_disc) < 50:
                continue

            out_dir = os.path.join(OUT_DIR, f"{sub}_{contrast}")
            results = run_hemisphere_experiment_multimodal(
                t1_volume=arrays["t1"],
                bold_4d=bold_4d,
                label_volume=label,
                brain_mask=arrays["mask"],
                affine=arrays["affine"],
                patch_size=9,
                axis_index=0,
                max_voxels_per_side=min(1500, min(n_conc, n_disc)),
                epochs=20,
                out_dir=out_dir,
                subject_name=f"{sub}_{contrast}_EXP2",
            )
            all_results[(sub, contrast)] = results
            print(f"  {sub} {contrast} PatchBOLDNet -> {results}")

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{s}|{c}": r for (s, c), r in all_results.items()}, f, indent=1)

    labels_x = [f"{s}\n{c}" for (s, c) in all_results.keys()]
    a_vals = [all_results[k]["A_train_B_test"] for k in all_results]
    b_vals = [all_results[k]["B_train_A_test"] for k in all_results]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels_x))
    width = 0.35
    ax.bar(x - width / 2, a_vals, width, label="обучение A / тест B", color="#8e44ad")
    ax.bar(x + width / 2, b_vals, width, label="обучение B / тест A", color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайный уровень")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон научного руководителя")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("Эксперимент 2: структурный патч + обычный сигнал BOLD (PatchBOLDNet)\nреальные метки CMRO2 + BOLD_percchange")
    ax.legend(fontsize=8)
    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, "experiment2_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено {summary_path}")

    all_accs = np.array(a_vals + b_vals)
    print(f"\nОбщий итог: mean={all_accs.mean():.3f} std={all_accs.std():.3f} n={len(all_accs)} прогонов")


if __name__ == "__main__":
    main()
