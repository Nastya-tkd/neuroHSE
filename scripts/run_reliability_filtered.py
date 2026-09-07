"""
Directly tests the explanation raised by an independent reanalysis of
this exact dataset's methodology: Buchel et al. (2026, eLife reviewed
preprint / bioRxiv, "Opposing BOLD signals and oxygen metabolism largely
arise from statistical uncertainty in metabolic estimates") re-examined
Epp et al.'s (2025, Nat Neurosci - the source of ds004873 and this
project's label) concordant/discordant classification and found that
77.2% of voxels could not be robustly classified once the statistical
uncertainty of the underlying CMRO2 estimates was accounted for - i.e.
most individual-voxel concordant/discordant labels may not be a stable
physiological property at all, but largely noise in the CMRO2 estimate
flipping its sign relative to true zero.

If that is a major contributor to this project's null result (152 real
training runs, chance-level across every architecture/feature tried),
then restricting to the voxels *least* likely to be noise-dominated
should raise structural-only accuracy relative to the full, unfiltered
set - even with no new features or architecture. This dataset has no
per-voxel uncertainty/SEM map to replicate Buchel et al.'s exact
statistical test, so a literature-standard proxy is used instead:
|CMRO2_percchange| magnitude. A fixed-scale measurement noise floor is
far more likely to flip the sign of a small percent change than a large
one - the same statistical logic (not the same numbers) behind their
filtering. This is disclosed as a proxy, not represented as a
reproduction of their exact method.

Runs SimplePatchCNN (patch=9, T1-only, matching the pooled-cohort
baseline methodology) on three inclusion sets built from the *same*
voxels/subjects for a fair comparison, most permissive to strictest:
  - all: every labeled voxel (this project's baseline all along)
  - top 50%: |CMRO2_percchange| above the per-contrast median
  - top 25%: |CMRO2_percchange| above the 75th percentile
Percentile thresholds are computed on each contrast's whole labeled-voxel
pool per fold's *training* side only, then applied to test as a fixed
cutoff - test voxels are filtered by that fixed number, not by a
percentile recomputed on the test side, so nothing about the test labels
leaks into the threshold.
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
from src.train import extract_and_normalize_patches, train_one_fold
from src import viz
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reliability_filtered")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0


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
    reliability = np.abs(pct)  # |CMRO2_percchange| - proxy for how noise-robust the sign is
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
        labels = (raw_labels > 0).astype(np.float32)
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
            "labels": labels[used_idx],
            "side": side_tags,
            "reliability": rel_vals[used_idx],
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "reliability": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["reliability"].append(d["reliability"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        reliability = np.concatenate(pooled[contrast]["reliability"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)}/{len(pos_b)} voxels A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            train_reliability = reliability[train_idx]
            for tier_name, keep_frac in [("all", 1.0), ("top50pct", 0.5), ("top25pct", 0.25)]:
                threshold = 0.0 if keep_frac == 1.0 else np.quantile(train_reliability, 1 - keep_frac)
                tr_keep = train_idx[reliability[train_idx] >= threshold]
                te_keep = test_idx[reliability[test_idx] >= threshold]
                if len(tr_keep) < 50 or len(te_keep) < 50 or len(np.unique(labels[tr_keep])) < 2 or len(np.unique(labels[te_keep])) < 2:
                    print(f"  {fold_name} {tier_name}: skipped (too few voxels/classes after filtering)")
                    continue

                _, hist, pred, probs = train_one_fold(
                    patches[tr_keep], labels[tr_keep], patches[te_keep], labels[te_keep],
                    epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
                )
                acc = hist["val_acc"][-1]
                print(f"  {fold_name} {tier_name}: n_train={len(tr_keep)} n_test={len(te_keep)} acc={acc:.3f}")
                all_results[(contrast, fold_name, tier_name)] = {
                    "acc": acc, "n_train": len(tr_keep), "n_test": len(te_keep), "n_subjects": n_subjects,
                }
                if tier_name == "top25pct":
                    viz.plot_confusion_and_roc(
                        labels[te_keep], pred, probs, f"reliability-filtered {contrast} {fold_name} {tier_name}",
                        os.path.join(OUT_DIR, f"{contrast}_{fold_name}_{tier_name}_eval.png"),
                    )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}|{t}": r for (c, f, t), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    keys = sorted(set((c, f) for (c, f, t) in all_results.keys()))
    x = np.arange(len(keys))
    width = 0.25
    colors = {"all": "#7f8c8d", "top50pct": "#e67e22", "top25pct": "#c0392b"}
    for i, tier in enumerate(["all", "top50pct", "top25pct"]):
        vals = [all_results.get((c, f, tier), {}).get("acc", np.nan) for c, f in keys]
        ax.bar(x + (i - 1) * width, vals, width, label=tier, color=colors[tier])
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Structural-only accuracy vs. |CMRO2_percchange| reliability filter\n(tests the Buchel et al. 2026 statistical-uncertainty explanation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "reliability_filtered_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
