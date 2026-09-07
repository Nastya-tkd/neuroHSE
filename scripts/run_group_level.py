"""
A genuinely different scale of analysis, not just another filter on the
same voxel-level prediction task: instead of "does this patient's voxel
predict its own concordance", asks "does a real anatomical region's
population-level tendency toward concordance or discordance correlate
with that region's population-level structural profile" - group
statistics across subjects, per the "different outcome variable/scale of
analysis" option this report has flagged since the Buchel et al.
section. One row per (Glasser parcel, hemisphere side), not per voxel or
per subject.

Per real Glasser parcel (reusing scripts/run_roi_averaged.py's exact
per-subject ROI-averaging: raw CMRO2_task/control averaged within the
parcel first, then one ROI percent-change; BOLD_percchange averaged
separately), collects each subject's own ROI-level concordant/discordant
label (only from subjects with >=20 valid voxels in that parcel), and
requires >=MIN_SUBJECTS_PER_PARCEL contributing subjects for the parcel
to be included at all. The parcel's group-level label is the majority
vote across those subjects (which side does this region lean toward,
across the population); its feature is the population-average of each
contributing subject's own z-scored regional T1 profile (mean T1, T1
std, log-size).

Classifies with RegionMLP (same architecture as run_roi_averaged.py -
a plain feature vector, no CNN needed at the region level), split by
hemisphere side (train on one side's parcels' group patterns, test on
the other's - the same leakage-safe convention as everywhere else, now
applied to regions instead of voxels).

Majority-class baseline is computed and reported for every fold from the
start (the lesson from run_roi_averaged.py's own earlier, corrected
result - see README "methodological note" - applied here before, not
after, a number is reported).
"""

import os
import sys
import json
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.cohort import ALL_SUBJECTS
from src.labeling import concordance_label
from src.patches import hemisphere_midpoint
from src.model import RegionMLP
from src.train import train_one_fold
from src import viz
from src.subject_loader import load_subject_robust
from src.glasser_atlas import get_subject_glasser_atlas, ATLAS_CACHE_DIR

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "group_level")
CONTRASTS = ["calc", "mem"]
MIN_VOXELS_PER_ROI = 20
MIN_SUBJECTS_PER_PARCEL = 15
SEED = 0


def subject_roi_rows(sub):
    """Per-subject per-contrast per-parcel ROI label + structural feature,
    same computation as scripts/run_roi_averaged.py (kept independent
    rather than imported, since here per-parcel *subject-level* rows are
    kept for later cross-subject aggregation, not pooled/trained directly)."""
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [error] {sub}: {e}")
        return None
    if result is None:
        print(f"  [skip] {sub}: {notes}")
        return None
    t1, affine, mask, cmro2, bold_pct = result

    t1_cache_path = os.path.join(ATLAS_CACHE_DIR, f"{sub}_t1_for_reg.nii.gz")
    if not os.path.exists(t1_cache_path):
        import nibabel as nib
        os.makedirs(ATLAS_CACHE_DIR, exist_ok=True)
        nib.save(nib.Nifti1Image(t1.astype(np.float32), affine), t1_cache_path)
    try:
        atlas = get_subject_glasser_atlas(sub, t1_cache_path)
    except Exception as e:
        print(f"  [skip] {sub}: atlas registration failed ({e})")
        return None
    if atlas.shape != t1.shape:
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    m = mask.astype(bool)
    global_mean, global_std = t1[t1 != 0].mean(), t1[t1 != 0].std() + 1e-6
    x_idx = np.arange(t1.shape[0])

    rows = {c: [] for c in CONTRASTS}  # each: (parcel_id, side, label, feat)
    for contrast in CONTRASTS:
        if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
            continue
        cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
        valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & m

        for side_name, side_cond in [("A", x_idx < midpoint), ("B", x_idx >= midpoint)]:
            side_mask = np.zeros_like(m)
            side_mask[side_cond, :, :] = True
            region_valid = valid & side_mask
            coords = np.argwhere(region_valid)
            if len(coords) == 0:
                continue
            pids = atlas[coords[:, 0], coords[:, 1], coords[:, 2]]
            for pid in np.unique(pids):
                if pid == 0:
                    continue
                sel = pids == pid
                if sel.sum() < MIN_VOXELS_PER_ROI:
                    continue
                vx = coords[sel]
                roi_bold = bold[vx[:, 0], vx[:, 1], vx[:, 2]].mean()
                roi_cmro2_task = cmro2_task[vx[:, 0], vx[:, 1], vx[:, 2]].mean()
                roi_cmro2_control = cmro2_control[vx[:, 0], vx[:, 1], vx[:, 2]].mean()
                if roi_cmro2_control == 0:
                    continue
                roi_pct = (roi_cmro2_task - roi_cmro2_control) / roi_cmro2_control * 100
                if not np.isfinite(roi_pct) or roi_bold == 0:
                    continue
                roi_label = 1 if concordance_label(np.array([roi_bold]), np.array([roi_pct]))[0] > 0 else 0

                t1_vals = t1[vx[:, 0], vx[:, 1], vx[:, 2]]
                feat = np.array([
                    (t1_vals.mean() - global_mean) / global_std,
                    t1_vals.std() / global_std,
                    float(np.log1p(len(vx))),
                ], dtype=np.float32)
                rows[contrast].append((int(pid), side_name, roi_label, feat))

    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # per_parcel[contrast][(pid, side)] = list of (label, feat) across subjects
    per_parcel = {c: {} for c in CONTRASTS}
    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            rows = subject_roi_rows(sub)
        except Exception:
            print(f"  [error] {sub}:\n{traceback.format_exc()}")
            rows = None
        if rows is None:
            log.append({"subject": sub, "status": "skipped"})
            continue
        n_rows = {c: len(v) for c, v in rows.items()}
        log.append({"subject": sub, "status": "used", "n_rows": n_rows})
        for contrast, entries in rows.items():
            for pid, side, label, feat in entries:
                key = (pid, side)
                per_parcel[contrast].setdefault(key, []).append((label, feat))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects contributed >=1 ROI row")

    all_results = {}
    for contrast in CONTRASTS:
        parcels = per_parcel[contrast]
        group_rows = []  # (pid, side, group_label, group_feat, n_subjects)
        for (pid, side), entries in parcels.items():
            if len(entries) < MIN_SUBJECTS_PER_PARCEL:
                continue
            labels = [e[0] for e in entries]
            feats = np.stack([e[1] for e in entries], axis=0)
            group_label = 1 if Counter(labels).most_common(1)[0][0] == 1 else 0
            group_feat = feats.mean(axis=0)
            group_rows.append((pid, side, group_label, group_feat, len(entries)))

        if len(group_rows) < 20:
            print(f"\n=== {contrast}: only {len(group_rows)} parcels with >={MIN_SUBJECTS_PER_PARCEL} subjects, skipping ===")
            continue

        feats = np.stack([r[3] for r in group_rows], axis=0)
        labels = np.array([r[2] for r in group_rows], dtype=np.float32)
        sides = np.array([r[1] for r in group_rows])
        n_subj_per_parcel = [r[4] for r in group_rows]
        pos_a, pos_b = np.where(sides == "A")[0], np.where(sides == "B")[0]
        n_conc, n_disc = int((labels > 0).sum()), int((labels == 0).sum())
        print(f"\n=== {contrast}: {len(group_rows)} parcels ({len(pos_a)} side A / {len(pos_b)} side B), "
              f"{n_conc} group-concordant / {n_disc} group-discordant, "
              f"median subjects/parcel={int(np.median(n_subj_per_parcel))} ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            if len(train_idx) < 10 or len(test_idx) < 10 or len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[test_idx])) < 2:
                print(f"  {fold_name}: skipped (too few parcels or a single class)")
                continue
            test_conc_frac = labels[test_idx].mean()
            majority_baseline = max(test_conc_frac, 1 - test_conc_frac)

            _, hist, pred, probs = train_one_fold(
                feats[train_idx], labels[train_idx], feats[test_idx], labels[test_idx],
                epochs=30, batch_size=8, device="cpu", seed=SEED,
                model_factory=lambda: RegionMLP(in_dim=3, hidden=16),
            )
            acc = hist["val_acc"][-1]
            beats = "YES" if acc > majority_baseline else "no"
            print(f"  {fold_name}: n_train={len(train_idx)} n_test={len(test_idx)} "
                  f"acc={acc:.3f} majority_baseline={majority_baseline:.3f} beats_baseline={beats}")
            all_results[(contrast, fold_name)] = {
                "acc": acc, "majority_baseline": majority_baseline,
                "n_train": len(train_idx), "n_test": len(test_idx), "n_parcels": len(group_rows),
            }
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"group-level {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    if all_results:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        keys = list(all_results.keys())
        x = np.arange(len(keys))
        width = 0.35
        vals = [all_results[k]["acc"] for k in keys]
        base = [all_results[k]["majority_baseline"] for k in keys]
        ax.bar(x - width / 2, vals, width, label="model accuracy", color="#2980b9")
        ax.bar(x + width / 2, base, width, label="majority-class baseline", color="#2980b9", alpha=0.4, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Group-level (cross-subject, region-level) classification\nmodel vs. its own majority-class baseline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, "group_level_summary.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
