"""
Recasts the task as 3-class (concordant / discordant / unreliable)
instead of forcing every voxel into a binary concordant/discordant
call - a more honest reflection of the Buchel et al. (2026) finding
(see README/report 06j-06k) that a large share of voxels' true sign
isn't statistically determinable at all. Rather than silently keeping
those voxels in a 2-class problem (implicitly asserting they have a
knowable true sign) or silently dropping them (as the reliability-
filtered experiments did), this gives the model an explicit third
option and scores it on whether it can tell reliable from unreliable
voxels at all, not just get the sign right on the ones that are.

Class 2 ("unreliable") = the bottom half of each fold's *training*
voxels by |CMRO2_percchange| (same magnitude proxy as
run_reliability_filtered.py, same threshold-fit-on-train-only
discipline so nothing about test labels leaks into the cutoff); classes
0/1 = discordant/concordant for voxels above that threshold. Same
architecture (SimplePatchCNN, patch=9, now with n_classes=3 and
CrossEntropyLoss via train_one_fold_multiclass) and voxel-selection
protocol as every other structural-only run in this project, for direct
comparability.

Reports both standard 3-class accuracy (chance = 1/3, not 1/2 - the
callout box in the summary plot marks this explicitly) and a secondary,
more interpretable number: binary sign-accuracy restricted to voxels
that were *actually* reliable in the test set (a strict "predicted the
right side of a coin AND correctly recognized it as flippable" score).
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
from src.patches import split_by_axis, hemisphere_midpoint, labeled_voxel_coords
from src.model import SimplePatchCNN
from src.train import extract_and_normalize_patches, train_one_fold_multiclass
from src import viz
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "three_class")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0
CLASS_NAMES = ["discordant", "concordant", "unreliable"]


def build_label_and_reliability(cmro2, bold_pct, mask, contrast):
    if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
        return None, None
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    label[valid] = concordance_label(bold, pct)[valid]
    reliability = np.abs(pct)
    return label, reliability


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

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    rng = np.random.default_rng(hash(sub) % (2**31))

    out = {}
    for contrast in CONTRASTS:
        label, reliability = build_label_and_reliability(cmro2, bold_pct, mask, contrast)
        if label is None:
            continue
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        coords = labeled_voxel_coords(label, mask)
        raw_labels = label[coords[:, 0], coords[:, 1], coords[:, 2]]
        binary_labels = (raw_labels > 0).astype(np.int64)  # 1=concordant, 0=discordant (pre-unreliable-split)
        rel_vals = reliability[coords[:, 0], coords[:, 1], coords[:, 2]]

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

        out[contrast] = {
            "patches": patches,
            "binary_labels": binary_labels[used_idx],
            "side": side_tags,
            "reliability": rel_vals[used_idx],
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "binary_labels": [], "side": [], "reliability": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["patches"].append(d["patches"])
            pooled[contrast]["binary_labels"].append(d["binary_labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["reliability"].append(d["reliability"])
            pooled[contrast]["subject"].append(np.full(len(d["binary_labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        binary_labels = np.concatenate(pooled[contrast]["binary_labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        reliability = np.concatenate(pooled[contrast]["reliability"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)}/{len(pos_b)} voxels A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            rel_threshold = np.quantile(reliability[train_idx], 0.5)
            three_class = np.where(reliability >= rel_threshold, binary_labels, 2).astype(np.int64)

            _, hist, pred, probs = train_one_fold_multiclass(
                patches[train_idx], three_class[train_idx], patches[test_idx], three_class[test_idx],
                n_classes=3, epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: SimplePatchCNN(n_classes=3),
            )
            acc = hist["val_acc"][-1]

            true_test, pred_test = three_class[test_idx], pred
            reliable_mask = true_test != 2
            if reliable_mask.sum() > 0:
                strict_acc = float((pred_test[reliable_mask] == true_test[reliable_mask]).mean())
            else:
                strict_acc = float("nan")

            print(f"  {fold_name}: 3-class acc={acc:.3f} (chance=0.333)  strict binary-on-reliable acc={strict_acc:.3f}")
            all_results[(contrast, fold_name)] = {
                "three_class_acc": acc, "strict_reliable_acc": strict_acc,
                "n_train": len(train_idx), "n_test": len(test_idx), "n_subjects": n_subjects,
            }
            viz.plot_multiclass_confusion(
                true_test, pred_test, CLASS_NAMES, f"3-class {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_confusion.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(all_results.keys())
    x = np.arange(len(keys))
    width = 0.35
    three_vals = [all_results[k]["three_class_acc"] for k in keys]
    strict_vals = [all_results[k]["strict_reliable_acc"] for k in keys]
    ax.bar(x - width / 2, three_vals, width, label="3-class accuracy", color="#8e44ad")
    ax.bar(x + width / 2, strict_vals, width, label="binary acc. on true-reliable subset", color="#16a085")
    ax.axhline(1 / 3, color="gray", linestyle=":", label="3-class chance (0.333)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="binary chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("3-class (concordant/discordant/unreliable) framing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "three_class_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
