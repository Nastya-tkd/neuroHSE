"""
REAL Experiment 1: structural-only concordant/discordant classification,
on REAL labels (not synthetic), for all 5 subjects.

Real label = sign(CMRO2_percchange) * sign(BOLD_percchange), computed from
real CMRO2 (Fick's principle output) and real BOLD_percchange maps recovered
via scripts/download_real_labels.py (see that script's docstring for
provenance: CC0-licensed OpenNeuro derivatives, recovered from S3 version
history). CMRO2_percchange = (CMRO2_task - CMRO2_control) / CMRO2_control * 100,
exactly matching combined_pipeline.py's own computation
(~line 10111: `percchange_CMRO2 = (task_CMRO2 - base_CMRO2) / base_CMRO2 * 100`).

Structural input is <sub>_space-T2_desc-brain_T1w.nii.gz - the same T2-space
brain-extracted T1w used elsewhere in the source pipeline, and critically in
the SAME space as the CMRO2/BOLD_percchange label maps (all space-T2), so
patch centers and label voxels line up without extra registration.

Runs one experiment per (subject, contrast) pair - calc-vs-control and
mem-vs-control - as a built-in replication check: if structural data really
carries concordance information, the effect should show up in both task
contrasts, not just one.
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
    # some qmri outputs carry a trailing singleton 4th dim
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
                print(f"  [!] too few labeled voxels of one class, skipping")
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

    # summary plot: accuracy per subject x contrast x fold direction
    labels_x = [f"{s}\n{c}" for (s, c) in all_results.keys()]
    a_vals = [all_results[k]["A_train_B_test"] for k in all_results]
    b_vals = [all_results[k]["B_train_A_test"] for k in all_results]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels_x))
    width = 0.35
    ax.bar(x - width / 2, a_vals, width, label="A train / B test (left hemi -> right)", color="#8e44ad")
    ax.bar(x + width / 2, b_vals, width, label="B train / A test (right hemi -> left)", color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="supervisor's target range")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("REAL Experiment 1: structural-only concordant/discordant classification\n(real CMRO2 + BOLD_percchange labels, 5 subjects x 2 task contrasts)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, "real_experiment_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nSummary saved to {summary_path}")

    print("\n=== class balance ===")
    for k, (nc, nd) in class_balance.items():
        print(k, "concordant:", nc, "discordant:", nd)


if __name__ == "__main__":
    main()
