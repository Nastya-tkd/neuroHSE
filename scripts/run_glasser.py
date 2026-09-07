"""
Real anatomical parcellation (Glasser/HCP-MMP1.0), the one thing every
earlier report in this project said was genuinely blocked - both
externally (Zenodo/HuggingFace/OSF/NITRC) and, unlike CBF/OEF or the
MedicalNet weights, confirmed absent from this dataset's own derivatives
too. It turned out obtainable a different way: the atlas itself (not
weights, not the dataset) is a static, group-level label volume with no
protected/licensed distribution requirement, and a community mirror
exists as a plain file in a GitHub repo
(github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL) - a host already
reachable in this session. Getting it into each subject's own space
needed real image registration, done here with ANTsPy (PyPI) against a
nilearn-bundled MNI152 template (see src/glasser_atlas.py for the full
methodology and its honesty caveats - this is a coarse volumetric
approximation of Glasser, explicitly flagged as such by the atlas's own
maintainer, not the surface-based version the atlas was built/validated
for).

Otherwise identical in spirit to run_parcellation.py's k-means version:
each labeled voxel gets a 3-feature descriptor (mean T1, T1 std,
log-size) of the *real* Glasser parcel it falls in (not a data-driven
cluster), computed separately per hemisphere-split side so no feature
leaks across the train/test boundary, fed through PatchBOLDConditionNet
the same way. This directly tests whether real, group-consistent
anatomical region identity (as opposed to a data-driven local cluster)
carries the signal the local T1 patch alone did not.
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
from src.model import PatchBOLDConditionNet
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold_multimodal
from src import viz
from src.subject_loader import load_subject_robust
from src.glasser_atlas import get_subject_glasser_atlas, ATLAS_CACHE_DIR

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "glasser")
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


def region_stats_for_side(t1, mask, atlas, side_mask_x):
    """{parcel_id: (mean_t1, std_t1, log_size)} using only this side's own
    voxels - mirrors run_parcellation.py's per-side stats, so a real
    Glasser parcel (which in practice never straddles the midline, since
    IDs 1-180/1000-1180 are already hemisphere-specific) still can't leak
    features across the train/test hemisphere boundary even if the
    volumetric registration bleeds slightly across it."""
    xs = np.where(side_mask_x)[0]
    m = mask.astype(bool)
    coords = np.argwhere(m)
    coords = coords[np.isin(coords[:, 0], xs)]
    if len(coords) == 0:
        return {}
    vals = t1[coords[:, 0], coords[:, 1], coords[:, 2]]
    ids = atlas[coords[:, 0], coords[:, 1], coords[:, 2]]
    stats = {}
    for pid in np.unique(ids):
        sel = ids == pid
        cvals = vals[sel]
        stats[int(pid)] = (float(cvals.mean()), float(cvals.std() + 1e-6), float(np.log1p(sel.sum())))
    return stats


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
        print(f"  [skip] {sub}: atlas shape {atlas.shape} != t1 shape {t1.shape}")
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    x_idx = np.arange(t1.shape[0])
    side_a_x, side_b_x = x_idx < (midpoint - margin), x_idx > (midpoint + margin)

    stats_a = region_stats_for_side(t1, mask, atlas, side_a_x)
    stats_b = region_stats_for_side(t1, mask, atlas, side_b_x)
    global_mean, global_std = t1[t1 != 0].mean(), t1[t1 != 0].std() + 1e-6
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
        used_labels = labels[used_idx]

        patches = extract_and_normalize_patches(t1, used_coords, PATCH_SIZE, mask)

        region_feat = np.zeros((len(used_idx), 3), dtype=np.float32)
        for i, (x, y, z) in enumerate(used_coords):
            pid = int(atlas[x, y, z])
            stats = stats_a if side_tags[i] == "A" else stats_b
            mean, std, logsize = stats.get(pid, (0.0, 1.0, 0.0))
            region_feat[i] = [(mean - global_mean) / global_std, std / global_std, logsize]

        out[contrast] = {"patches": patches, "labels": used_labels, "side": side_tags, "region_feat": region_feat}

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "region_feat": []} for c in CONTRASTS}
    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub)
        except Exception:
            print(f"  [error] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append((sub, "skipped"))
            continue
        log.append((sub, sorted(result.keys())))
        for contrast, d in result.items():
            for key in ["patches", "labels", "side", "region_feat"]:
                pooled[contrast][key].append(d[key])

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for _, v in log if v != "skipped")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} subjects used")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        feats = np.concatenate(pooled[contrast]["region_feat"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {len(pos_a)}/{len(pos_b)} voxels A/B ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            _, hist, pred, probs = train_one_fold_multimodal(
                patches[train_idx], feats[train_idx], labels[train_idx],
                patches[test_idx], feats[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: PatchBOLDConditionNet(n_bold_features=3),
            )
            acc = hist["val_acc"][-1]
            print(f"  {fold_name}: acc={acc:.3f}")
            all_results[(contrast, fold_name)] = acc
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"Glasser {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": acc for (c, f), acc in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(all_results.keys())
    vals = [all_results[k] for k in keys]
    ax.bar([f"{c}\n{f}" for c, f in keys], vals, color="#8e44ad")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_ylim(0, 1)
    ax.set_title("Real Glasser/HCP-MMP1.0 parcel region features (volumetric approximation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "glasser_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
