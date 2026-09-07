"""
The dynamic counterpart to run_cbf_oef.py: instead of baseline (resting)
CBF/OEF, uses the *change* during the task itself -
dCBF = CBF_task - CBF_control, dOEF = OEF_task - OEF_control - the
neurovascular response to the stimulus, not the physiological "resting
backdrop" tested before.

IMPORTANT circularity disclosure, read before interpreting any result
here. CMRO2 (Fick's principle) = CBF x OEF x CaO2, so
CMRO2_percchange = (CBF_task*OEF_task - CBF_control*OEF_control) /
(CBF_control*OEF_control) * 100 (CaO2 cancels, assumed ~constant across
conditions). dCBF and dOEF (as used here) are NOT algebraically the same
quantity as CMRO2_percchange - the multiplicative/baseline-normalized
combination is not recoverable from the two differences alone without
also knowing the absolute CBF_control/OEF_control values - but they are
the two physiological quantities CMRO2_percchange is built from, and are
correlated with it in a way baseline-only CBF/OEF is not. This sits
strictly between run_cbf_oef.py's baseline experiment (a legitimate,
clearly non-circular covariate) and the oracle positive control in
run_extras.py (label-defining values fed in directly, not a scientific
result). Framed and reported as its own honest category, not as either
of those two: if this scores near chance, that's an even stronger null
than the baseline case (task-period signal doesn't help either, despite
being physiologically closer to the label). If it scores well above
chance, that must be reported as reflecting its partial algebraic
closeness to the label - like the oracle control - not as a discovered
structural biomarker.

Same protocol otherwise as run_cbf_oef.py: T1-only (1ch) vs.
T1+dCBF+dOEF (3ch) SimplePatchCNN, same voxels, same leakage-safe
hemisphere split, full available cohort.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.cohort import ALL_SUBJECTS
from src.dataio import load_nifti
from src.labeling import concordance_label
from src.patches import split_by_axis, hemisphere_midpoint
from src.model import SimplePatchCNN
from src.train import (
    select_labeled_coords,
    extract_and_normalize_patches,
    extract_and_normalize_multichannel_patches,
    train_one_fold,
)
from src import viz
from src.subject_loader import load_subject_robust
from scripts.download_cbf_oef import download_subject_cbf_oef

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "task_cbf_oef")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0


def build_label(cmro2, bold_pct, mask, contrast):
    if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
        return None
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    label[valid] = concordance_label(bold, pct)[valid]
    return label


def load_cbf_oef_condition(sub, condition, mask, expected_shape):
    files = download_subject_cbf_oef(sub, condition=condition)
    if len(files) < 2:
        return None
    d = os.path.join("data", sub, "derivatives")
    cbf, _, _ = load_nifti(os.path.join(d, f"{sub}_task-{condition}_space-T2_cbf.nii"))
    oef, _, _ = load_nifti(os.path.join(d, f"{sub}_task-{condition}_space-T2_oef.nii"))
    cbf, oef = cbf.squeeze(), oef.squeeze()
    if cbf.shape != expected_shape or oef.shape != expected_shape:
        return None
    m = mask.astype(bool)
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0) * m
    oef = np.nan_to_num(oef, nan=0.0, posinf=0.0, neginf=0.0) * m
    return cbf, oef


def process_subject(sub):
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [error] {sub}: {e}")
        return None
    if result is None:
        print(f"  [skip] {sub}: {notes}")
        return None
    t1, affine, mask, cmro2, bold_pct = result

    control_cbf_oef = load_cbf_oef_condition(sub, "control", mask, t1.shape)
    if control_cbf_oef is None:
        print(f"  [skip] {sub}: no usable control CBF/OEF")
        return None
    cbf_control, oef_control = control_cbf_oef

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    rng = np.random.default_rng(hash(sub) % (2**31))

    out = {}
    for contrast in CONTRASTS:
        label = build_label(cmro2, bold_pct, mask, contrast)
        if label is None:
            continue
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        task_cbf_oef = load_cbf_oef_condition(sub, contrast, mask, t1.shape)
        if task_cbf_oef is None:
            print(f"  [skip contrast] {sub} {contrast}: no usable task-condition CBF/OEF")
            continue
        cbf_task, oef_task = task_cbf_oef
        d_cbf = cbf_task - cbf_control
        d_oef = oef_task - oef_control

        coords, labels = select_labeled_coords(label, mask, max_voxels=None, seed=SEED)
        side_a, side_b = split_by_axis(coords, 0, midpoint, margin)

        def subsample(m):
            idx = np.where(m)[0]
            if len(idx) > MAX_VOXELS_PER_SIDE_PER_SUBJECT:
                idx = rng.choice(idx, size=MAX_VOXELS_PER_SIDE_PER_SUBJECT, replace=False)
            return idx

        idx_a, idx_b = subsample(side_a), subsample(side_b)
        used_idx = np.concatenate([idx_a, idx_b])
        used_coords = coords[used_idx]
        side_tags = np.array(["A"] * len(idx_a) + ["B"] * len(idx_b))

        patches_1ch = extract_and_normalize_patches(t1, used_coords, PATCH_SIZE, mask)
        patches_3ch = extract_and_normalize_multichannel_patches(
            [t1, d_cbf, d_oef], used_coords, PATCH_SIZE, mask
        )

        out[contrast] = {
            "patches_1ch": patches_1ch,
            "patches_3ch": patches_3ch,
            "labels": labels[used_idx],
            "side": side_tags,
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches_1ch": [], "patches_3ch": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub)
        except Exception:
            print(f"  [error] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append({"subject": sub, "status": "skipped"})
            continue
        log.append({"subject": sub, "status": "used", "contrasts": sorted(result.keys())})
        for contrast, d in result.items():
            pooled[contrast]["patches_1ch"].append(d["patches_1ch"])
            pooled[contrast]["patches_3ch"].append(d["patches_3ch"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used (have control + task CBF/OEF for >=1 contrast)")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches_1ch"]:
            print(f"{contrast}: no usable subjects, skipping")
            continue
        patches_1ch = np.concatenate(pooled[contrast]["patches_1ch"], axis=0)
        patches_3ch = np.concatenate(pooled[contrast]["patches_3ch"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)} side-A voxels, {len(pos_b)} side-B voxels ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            _, hist_struct, pred_struct, probs_struct = train_one_fold(
                patches_1ch[train_idx], labels[train_idx], patches_1ch[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
            )
            acc_struct = hist_struct["val_acc"][-1]

            _, hist_phys, pred_phys, probs_phys = train_one_fold(
                patches_3ch[train_idx], labels[train_idx], patches_3ch[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: SimplePatchCNN(in_channels=3),
            )
            acc_phys = hist_phys["val_acc"][-1]

            print(f"  {fold_name}: structural-only(T1)={acc_struct:.3f}  +dynamic dCBF/dOEF={acc_phys:.3f}")
            all_results[(contrast, fold_name)] = {
                "structural_only": acc_struct, "structural_plus_dcbf_doef": acc_phys, "n_subjects": n_subjects,
            }
            viz.plot_confusion_and_roc(
                labels[test_idx], pred_phys, probs_phys, f"dCBF/dOEF {contrast} {fold_name}: T1+task-dynamic dCBF/dOEF",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_task_cbf_oef_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    labels_x = [f"{c}\n{f}" for (c, f) in all_results.keys()]
    struct_vals = [all_results[k]["structural_only"] for k in all_results]
    phys_vals = [all_results[k]["structural_plus_dcbf_doef"] for k in all_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels_x))
    width = 0.35
    ax.bar(x - width / 2, struct_vals, width, label="T1 only", color="#7f8c8d")
    ax.bar(x + width / 2, phys_vals, width, label="T1 + task dCBF/dOEF (dynamic)", color="#e67e22")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("T1-only vs. T1+dynamic task-period dCBF/dOEF\n(partial algebraic closeness to the label - see script docstring)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "task_cbf_oef_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
