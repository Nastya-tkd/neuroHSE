"""
Baseline (control-condition) CBV (cerebral blood volume) as an extra input
channel alongside T1 - the direct test of Alexei Ossadtchi's hypothesis
that structural MRI contrasts might encode capillary/vascular morphology
via sub-voxel partial-volume averaging (blood vs. tissue T1/T2 differ, so a
voxel's blood-volume fraction measurably shifts its observed relaxation -
the same physical mechanism BOLD/DSC imaging itself relies on). Of the
physiological maps used in this project, CBV (blood volume) is the more
direct macroscopic proxy for capillary density/vascular morphology than
CBF (flow) or T1 intensity alone - untested until now.

Non-circularity: only the *control*-condition (resting-state) CBV is used,
never the task-condition value. The concordant/discordant label is defined
by the *change* between task and control
(sign(BOLD_percchange) x sign(CMRO2_percchange)); a baseline-only map is
not part of that computation, so this is a genuine physiological-covariate
experiment, not label leakage - the same logic as scripts/run_cbf_oef.py.

Runs two models per contrast per fold direction, apples-to-apples on the
exact same voxels/patches:
  - structural-only (T1, 1 channel).
  - structural + baseline CBV (T1+CBV, 2 channels, co-registered and
    independently normalized per channel).

Majority-class baseline is computed and reported alongside both models'
accuracy from the start (same discipline as every experiment since the
ROI-averaged correction), even though this is not a filtering experiment -
so class balance is identical between the two arms and the check is mainly
a transparency/consistency habit rather than an expected source of
surprise here.
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
from scripts.download_cbv import download_subject_cbv

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cbv")
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


def load_cbv(sub, mask, expected_shape):
    """Baseline CBV, brain-masked and cleaned of non-finite noise outside
    the mask (raw qmri maps aren't pre-skull-stripped)."""
    path = download_subject_cbv(sub)
    if path is None:
        return None
    cbv, _, _ = load_nifti(path)
    cbv = cbv.squeeze()
    if cbv.shape != expected_shape:
        return None
    m = mask.astype(bool)
    cbv = np.nan_to_num(cbv, nan=0.0, posinf=0.0, neginf=0.0) * m
    return cbv


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

    cbv = load_cbv(sub, mask, t1.shape)
    if cbv is None:
        print(f"  [skip] {sub}: no usable baseline CBV")
        return None

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
        patches_2ch = extract_and_normalize_multichannel_patches(
            [t1, cbv], used_coords, PATCH_SIZE, mask
        )

        out[contrast] = {
            "patches_1ch": patches_1ch,
            "patches_2ch": patches_2ch,
            "labels": labels[used_idx],
            "side": side_tags,
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches_1ch": [], "patches_2ch": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["patches_2ch"].append(d["patches_2ch"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used (have baseline CBV + >=1 contrast)")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches_1ch"]:
            print(f"{contrast}: no usable subjects, skipping")
            continue
        patches_1ch = np.concatenate(pooled[contrast]["patches_1ch"], axis=0)
        patches_2ch = np.concatenate(pooled[contrast]["patches_2ch"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)} side-A voxels, {len(pos_b)} side-B voxels ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            test_labels = labels[test_idx]
            majority_baseline = float(max(test_labels.mean(), 1 - test_labels.mean()))

            _, hist_struct, pred_struct, probs_struct = train_one_fold(
                patches_1ch[train_idx], labels[train_idx], patches_1ch[test_idx], test_labels,
                epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
            )
            acc_struct = hist_struct["val_acc"][-1]

            _, hist_phys, pred_phys, probs_phys = train_one_fold(
                patches_2ch[train_idx], labels[train_idx], patches_2ch[test_idx], test_labels,
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: SimplePatchCNN(in_channels=2),
            )
            acc_phys = hist_phys["val_acc"][-1]

            beats = "YES" if acc_phys > majority_baseline else "no"
            print(f"  {fold_name}: structural-only(T1)={acc_struct:.3f}  +baseline CBV={acc_phys:.3f}  "
                  f"majority_baseline={majority_baseline:.3f}  beats_baseline={beats}")
            all_results[(contrast, fold_name)] = {
                "structural_only": acc_struct, "structural_plus_cbv": acc_phys,
                "majority_baseline": majority_baseline, "n_subjects": n_subjects,
                "n_train": len(train_idx), "n_test": len(test_idx),
            }
            viz.plot_confusion_and_roc(
                test_labels, pred_phys, probs_phys, f"CBV {contrast} {fold_name}: T1+baseline CBV",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_cbv_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    labels_x = [f"{c}\n{f}" for (c, f) in all_results.keys()]
    struct_vals = [all_results[k]["structural_only"] for k in all_results]
    phys_vals = [all_results[k]["structural_plus_cbv"] for k in all_results]
    base_vals = [all_results[k]["majority_baseline"] for k in all_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels_x))
    width = 0.25
    ax.bar(x - width, struct_vals, width, label="T1 only", color="#7f8c8d")
    ax.bar(x, phys_vals, width, label="T1 + baseline CBV", color="#c0392b")
    ax.bar(x + width, base_vals, width, label="majority-class baseline", color="#c0392b", alpha=0.35, hatch="//")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("T1-only vs. T1+baseline CBV (vascular-morphology hypothesis)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "cbv_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
