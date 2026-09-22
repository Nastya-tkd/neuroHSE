"""
РЕАЛЬНЫЙ Эксперимент 1: классификация concordant/discordant только по
структуре, на РЕАЛЬНЫХ метках (не синтетических), для всех 5 пациентов.

Реальная метка = sign(CMRO2_percchange) * sign(BOLD_percchange), вычислена
по реальным CMRO2 (результат принципа Фика) и реальным картам
BOLD_percchange, восстановленным через scripts/download_real_labels.py (см.
docstring этого скрипта о происхождении данных: производные OpenNeuro под
лицензией CC0, восстановлены из истории версий S3). CMRO2_percchange =
(CMRO2_task - CMRO2_control) / CMRO2_control * 100, что в точности
совпадает с собственным вычислением в combined_pipeline.py
(~строка 10111: `percchange_CMRO2 = (task_CMRO2 - base_CMRO2) / base_CMRO2 * 100`).

Структурный вход - это <sub>_space-T2_desc-brain_T1w.nii.gz - тот же T1w с
удалённым черепом в пространстве T2, что используется в остальных местах
исходного конвейера, и, что критически важно, в ТОМ ЖЕ пространстве, что и
карты меток CMRO2/BOLD_percchange (все в space-T2), поэтому центры патчей
и воксели меток совпадают без дополнительной регистрации.

Запускает один эксперимент на каждую пару (пациент, контраст) - calc-vs-control
и mem-vs-control - в качестве встроенной проверки воспроизводимости: если
структурные данные действительно несут информацию о concordance, эффект
должен проявиться в обоих контрастах задачи, а не только в одном.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.dataio import load_nifti
from src.labeling import concordance_label
from src.train import run_hemisphere_experiment

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "real_experiment")

SUBJECTS = ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]
CONTRASTS = ["calc", "mem"]


def load_subject_arrays(sub):
    d = os.path.join(DATA_DIR, sub, "derivatives")
    t1, affine, _ = load_nifti(os.path.join(d, f"{sub}_space-T2_desc-brain_T1w.nii.gz"))
    mask_raw, _, _ = load_nifti(os.path.join(d, f"{sub}_BrMsk_CSF_30slices.nii.gz"))
    mask = (mask_raw > 0.5).astype(np.uint8)

    cmro2_control, _, _ = load_nifti(os.path.join(d, f"{sub}_task-control_space-T2_desc-orig_cmro2.nii"))
    cmro2_calc, _, _ = load_nifti(os.path.join(d, f"{sub}_task-calc_space-T2_desc-orig_cmro2.nii"))
    cmro2_mem, _, _ = load_nifti(os.path.join(d, f"{sub}_task-mem_space-T2_desc-orig_cmro2.nii"))
    # некоторые qmri-выходы содержат лишнее одномерное 4-е измерение
    cmro2_control = cmro2_control.squeeze()
    cmro2_calc = cmro2_calc.squeeze()
    cmro2_mem = cmro2_mem.squeeze()
    bold_calccontrol, _, _ = load_nifti(os.path.join(d, f"{sub}_task-calccontrol_space-T2_BOLD_percchange.nii.gz"))
    bold_memcontrol, _, _ = load_nifti(os.path.join(d, f"{sub}_task-memcontrol_space-T2_BOLD_percchange.nii.gz"))

    return {
        "t1": t1, "affine": affine, "mask": mask,
        "cmro2_control": cmro2_control, "cmro2_calc": cmro2_calc, "cmro2_mem": cmro2_mem,
        "bold_calccontrol": bold_calccontrol, "bold_memcontrol": bold_memcontrol,
    }


def cmro2_percchange(task, control):
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (task - control) / control * 100
    pct[~np.isfinite(pct)] = 0
    return pct


def build_label(arrays, contrast):
    cmro2_task = arrays[f"cmro2_{contrast}"]
    cmro2_control = arrays["cmro2_control"]
    bold = arrays[f"bold_{contrast}control"]

    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & arrays["mask"].astype(bool)
    cmro2_pct = cmro2_percchange(cmro2_task, cmro2_control)
    label = np.zeros(arrays["t1"].shape, dtype=np.float32)
    conc = concordance_label(bold, cmro2_pct)
    label[valid] = conc[valid]
    return label


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}   # (sub, contrast) -> fold_results dict
    class_balance = {}

    for sub in SUBJECTS:
        arrays = load_subject_arrays(sub)
        for contrast in CONTRASTS:
            label = build_label(arrays, contrast)
            n_conc = int((label > 0).sum())
            n_disc = int((label < 0).sum())
            class_balance[(sub, contrast)] = (n_conc, n_disc)
            print(f"{sub} {contrast}: concordant={n_conc}, discordant={n_disc}")

            if min(n_conc, n_disc) < 50:
                print(f"  [!] слишком мало размеченных вокселей одного класса, пропуск")
                continue

            out_dir = os.path.join(OUT_DIR, f"{sub}_{contrast}")
            results = run_hemisphere_experiment(
                t1_volume=arrays["t1"],
                label_volume=label,
                brain_mask=arrays["mask"],
                affine=arrays["affine"],
                patch_size=9,
                axis_index=0,
                max_voxels_per_side=min(1500, min(n_conc, n_disc)),
                epochs=20,
                out_dir=out_dir,
                subject_name=f"{sub}_{contrast}_REAL",
            )
            all_results[(sub, contrast)] = results
            print(f"  -> {results}")

    # сводный график: точность по пациенту x контрасту x направлению разбиения
    labels_x = [f"{s}\n{c}" for (s, c) in all_results.keys()]
    a_vals = [all_results[k]["A_train_B_test"] for k in all_results]
    b_vals = [all_results[k]["B_train_A_test"] for k in all_results]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels_x))
    width = 0.35
    ax.bar(x - width / 2, a_vals, width, label="обучение A / тест B (левое полушарие -> правое)", color="#8e44ad")
    ax.bar(x + width / 2, b_vals, width, label="обучение B / тест A (правое полушарие -> левое)", color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайный уровень")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон научного руководителя")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("РЕАЛЬНЫЙ Эксперимент 1: классификация concordant/discordant только по структуре\n(реальные метки CMRO2 + BOLD_percchange, 5 пациентов x 2 контраста задачи)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, "real_experiment_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nСводка сохранена в {summary_path}")

    print("\n=== баланс классов ===")
    for k, (nc, nd) in class_balance.items():
        print(k, "concordant:", nc, "discordant:", nd)


if __name__ == "__main__":
    main()
