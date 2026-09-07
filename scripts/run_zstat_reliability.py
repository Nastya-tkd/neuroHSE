"""
Real (not proxy) statistical-reliability filtering: uses the source
pipeline's own first-level BOLD activation Z-statistic
(scripts/download_zstat.py) instead of |CMRO2_percchange| magnitude
(scripts/run_reliability_filtered.py's proxy) to decide which voxels'
concordant/discordant label to trust. This is the direct answer to "get
a genuine statistical-uncertainty estimate instead of a proxy" - the
z-statistic is literally an effect-size/standard-error ratio from the
source GLM, not a magnitude heuristic invented for this project.

Same structural-only architecture (SimplePatchCNN, patch=9) and
leakage-safe protocol as every other filtering experiment here, on the
same voxels/subjects for direct comparability. Tiers: all / top 50% /
top 25% by |z|, threshold fit on each fold's training side only.

Learned from this project's own mistake (see README "methodological
note" after the Buchel-motivated experiments): every tier's own trivial
majority-class baseline is computed and reported *from the start* this
time, not added after a promising-looking number, since filtering by
any criterion correlated with the label can shift its class balance.
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
from src.patches import split_by_axis, hemisphere_midpoint, labeled_voxel_coords
from src.model import SimplePatchCNN
from src.train import extract_and_normalize_patches, train_one_fold
from src import viz
from src.subject_loader import load_subject_robust
from scripts.download_zstat import download_subject_zstat

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "zstat_reliability")
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


def load_zstat(sub, contrast, expected_shape):
    path = download_subject_zstat(sub, contrast)
    if path is None:
        return None
    z, _, _ = load_nifti(path)
    z = z.squeeze()
    if z.shape != expected_shape:
        return None
    return np.nan_to_num(np.abs(z), nan=0.0, posinf=0.0, neginf=0.0)


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
        label = build_label(cmro2, bold_pct, mask, contrast)
        if label is None:
            continue
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        zstat = load_zstat(sub, contrast, t1.shape)
        if zstat is None:
            print(f"  [skip contrast] {sub} {contrast}: no usable z-stat map")
            continue

        coords = labeled_voxel_coords(label, mask)
        raw_labels = label[coords[:, 0], coords[:, 1], coords[:, 2]]
        labels = (raw_labels > 0).astype(np.float32)
        z_vals = zstat[coords[:, 0], coords[:, 1], coords[:, 2]]

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
            "zstat": z_vals[used_idx],
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "zstat": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["zstat"].append(d["zstat"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used (have real z-stat map + >=1 contrast)")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        zstat = np.concatenate(pooled[contrast]["zstat"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)}/{len(pos_b)} voxels A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            train_z = zstat[train_idx]
            for tier_name, keep_frac in [("all", 1.0), ("top50pct", 0.5), ("top25pct", 0.25)]:
                threshold = 0.0 if keep_frac == 1.0 else np.quantile(train_z, 1 - keep_frac)
                tr_keep = train_idx[zstat[train_idx] >= threshold]
                te_keep = test_idx[zstat[test_idx] >= threshold]
                if len(tr_keep) < 50 or len(te_keep) < 50 or len(np.unique(labels[tr_keep])) < 2 or len(np.unique(labels[te_keep])) < 2:
                    print(f"  {fold_name} {tier_name}: skipped (too few voxels/classes after filtering)")
                    continue

                test_conc_frac = labels[te_keep].mean()
                majority_baseline = max(test_conc_frac, 1 - test_conc_frac)

                _, hist, pred, probs = train_one_fold(
                    patches[tr_keep], labels[tr_keep], patches[te_keep], labels[te_keep],
                    epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
                )
                acc = hist["val_acc"][-1]
                beats = "YES" if acc > majority_baseline else "no"
                print(f"  {fold_name} {tier_name}: n_train={len(tr_keep)} n_test={len(te_keep)} "
                      f"acc={acc:.3f} majority_baseline={majority_baseline:.3f} beats_baseline={beats}")
                all_results[(contrast, fold_name, tier_name)] = {
                    "acc": acc, "majority_baseline": majority_baseline,
                    "n_train": len(tr_keep), "n_test": len(te_keep), "n_subjects": n_subjects,
                }
                if tier_name == "top25pct":
                    viz.plot_confusion_and_roc(
                        labels[te_keep], pred, probs, f"z-stat filtered {contrast} {fold_name} {tier_name}",
                        os.path.join(OUT_DIR, f"{contrast}_{fold_name}_{tier_name}_eval.png"),
                    )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}|{t}": r for (c, f, t), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    keys = sorted(set((c, f) for (c, f, t) in all_results.keys()))
    x = np.arange(len(keys))
    width = 0.13
    tier_names = ["all", "top50pct", "top25pct"]
    colors = {"all": "#7f8c8d", "top50pct": "#e67e22", "top25pct": "#c0392b"}
    for i, tier in enumerate(tier_names):
        vals = [all_results.get((c, f, tier), {}).get("acc", np.nan) for c, f in keys]
        base = [all_results.get((c, f, tier), {}).get("majority_baseline", np.nan) for c, f in keys]
        ax.bar(x + (i - 1) * width * 2, vals, width, label=f"{tier} (model)", color=colors[tier])
        ax.bar(x + (i - 1) * width * 2 + width, base, width, label=f"{tier} (majority baseline)",
               color=colors[tier], alpha=0.4, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Real first-level Z-statistic reliability filter\n(model accuracy vs. its own majority-class baseline, side by side)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "zstat_reliability_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
