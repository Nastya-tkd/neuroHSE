"""
Completes the cohort: run_pooled_cohort.py and run_finetune.py only ever
loaded subjects with the "original" derivatives naming scheme (fixed
filenames), so all 10 subjects that use the alternate scheme found in
sub-p058...sub-p068 (whole-head T1w instead of pre-skull-stripped,
BrMsk_CSF.nii instead of _30slices, desc-CBV_cmro2 fallback for the task
condition, calc-only BOLD_percchange) were silently skipped every time,
even though 9 of the 10 are actually usable data.

This script uses src.subject_loader.load_subject_robust (which tries both
naming schemes) for the WHOLE cohort, so the "new" subjects are included
alongside the original 25 usable ones - approaching the dataset's full
N=40. Only sub-p058 is unrecoverable (genuinely has no control-condition
CMRO2 in T2 space anywhere in the S3 history, confirmed by hand). Only
"calc" is available for the 9 new subjects (no "mem" derivatives exist for
them at all), so the "mem" contrast keeps its original-cohort-only N.

Structural-only classification (SimplePatchCNN, patch=9), matching
run_pooled_cohort.py's methodology minus the condition-BOLD arm - the new
subjects don't have downloadable raw 4D BOLD/events under either naming
scheme investigated so far, so a like-for-like BOLD-features comparison
across the FULL expanded cohort isn't attempted here. This isolates the
one real question this script answers: does adding real held-out subjects
change the structural-only result at all.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.cohort import ALL_SUBJECTS
from src.labeling import concordance_label
from src.patches import split_by_axis, hemisphere_midpoint
from src.model import SimplePatchCNN
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold
from src import viz
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "full_cohort")
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


def process_subject(sub):
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [error] {sub}: {e}")
        return None, [f"error: {e}"]
    if result is None:
        print(f"  [skip] {sub}: {notes}")
        return None, notes
    t1, affine, mask, cmro2, bold_pct = result
    if notes:
        print(f"  [notes] {sub}: {notes}")

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
            print(f"  [skip contrast] {sub} {contrast}: degenerate label (conc={n_conc}, disc={n_disc})")
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

        patches = extract_and_normalize_patches(t1, used_coords, PATCH_SIZE, mask)
        out[contrast] = {"patches": patches, "labels": labels[used_idx], "side": side_tags}

    return (out if out else None), notes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result, notes = process_subject(sub)
        except Exception:
            print(f"  [error] {sub}:\n{traceback.format_exc()}")
            result, notes = None, ["unhandled error"]
        if result is None:
            log.append({"subject": sub, "status": "skipped", "notes": notes})
            continue
        log.append({"subject": sub, "status": "used", "contrasts": sorted(result.keys()), "notes": notes})
        for contrast, d in result.items():
            pooled[contrast]["patches"].append(d["patches"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used (>=1 contrast)")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            print(f"{contrast}: no usable subjects, skipping")
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)} side-A voxels, {len(pos_b)} side-B voxels ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            _, hist, pred, probs = train_one_fold(
                patches[train_idx], labels[train_idx], patches[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
            )
            acc = hist["val_acc"][-1]
            print(f"  {fold_name}: n_subjects={n_subjects} acc={acc:.3f}")
            all_results[(contrast, fold_name)] = {"acc": acc, "n_subjects": n_subjects, "n_train": len(train_idx), "n_test": len(test_idx)}
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"full cohort {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = list(all_results.keys())
    vals = [all_results[k]["acc"] for k in keys]
    ax.bar([f"{c}\n{f}" for c, f in keys], vals, color="#2980b9")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_ylim(0, 1)
    ax.set_title(f"Full-cohort structural-only classification ({n_used}/{len(ALL_SUBJECTS)} subjects used)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "full_cohort_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
