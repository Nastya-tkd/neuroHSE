"""
Data-driven parcellation: a real substitute for the unobtainable Glasser
atlas, one level up from run_extras.py's fixed geometric grid.

Instead of drawing arbitrary straight-line grid boundaries, this clusters
each subject's own brain voxels (per hemisphere-split side, so no region
straddles train/test) with k-means over (y, z, local T1 intensity) - the
cluster boundaries follow that subject's actual tissue-intensity structure
instead of a fixed grid, which is the standard approach for a data-driven
parcellation when no group-level anatomical atlas is available. Still not
a real anatomical atlas (no correspondence to consistent named brain
regions across subjects, and no atlas prior at all) - reported as that,
not oversold.

Each labeled voxel then gets a 3-feature descriptor: its cluster's mean
T1, T1 std, and log-size (in voxels) - the structural "neighborhood
profile" of the region it was clustered into, fed through
PatchBOLDConditionNet the same way covariates/coarse-region features were.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.cohort import ALL_SUBJECTS
from src.dataio import load_nifti
from src.labeling import concordance_label
from src.patches import split_by_axis, hemisphere_midpoint
from src.model import PatchBOLDConditionNet
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold_multimodal
from src import viz
from scripts.download_real_labels import download_subject_labels, DATA_DIR

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "parcellation")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
N_CLUSTERS = 20
SEED = 0


def load_subject_core(sub):
    d = os.path.join(DATA_DIR, sub, "derivatives")
    t1, affine, _ = load_nifti(os.path.join(d, f"{sub}_space-T2_desc-brain_T1w.nii.gz"))
    mask_raw, _, _ = load_nifti(os.path.join(d, f"{sub}_BrMsk_CSF_30slices.nii.gz"))
    mask = (mask_raw > 0.5).astype(np.uint8)
    cmro2 = {}
    for cond in ["control", "calc", "mem"]:
        v, _, _ = load_nifti(os.path.join(d, f"{sub}_task-{cond}_space-T2_desc-orig_cmro2.nii"))
        cmro2[cond] = v.squeeze()
    bold_pct = {}
    for c in ["calc", "mem"]:
        v, _, _ = load_nifti(os.path.join(d, f"{sub}_task-{c}control_space-T2_BOLD_percchange.nii.gz"))
        bold_pct[c] = v
    return t1, affine, mask, cmro2, bold_pct


def build_targets(cmro2, bold_pct, mask, contrast):
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    label[valid] = concordance_label(bold, pct)[valid]
    return label


def cluster_side(t1, mask, side_mask_x, n_clusters, seed):
    """K-means over (y, z, T1 intensity) for one hemisphere-split side.
    Returns a full-volume int array of cluster ids (-1 outside this side's mask)
    and {cluster_id: (mean_t1, std_t1, log_size)}."""
    xs = np.where(side_mask_x)[0]
    m = mask.astype(bool)

    coords = np.argwhere(m)
    coords = coords[np.isin(coords[:, 0], xs)]
    if len(coords) < n_clusters * 5:
        return None, None

    vals = t1[coords[:, 0], coords[:, 1], coords[:, 2]]
    y_norm = coords[:, 1] / t1.shape[1]
    z_norm = coords[:, 2] / t1.shape[2]
    t1_norm = (vals - vals.mean()) / (vals.std() + 1e-6)
    features = np.stack([y_norm, z_norm, t1_norm * 0.5], axis=1)  # intensity weighted down vs spatial

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=3).fit(features)
    labels = km.labels_

    cluster_id_map = np.full(t1.shape, -1, dtype=np.int32)
    cluster_id_map[coords[:, 0], coords[:, 1], coords[:, 2]] = labels

    stats = {}
    for cid in range(n_clusters):
        cvals = vals[labels == cid]
        if len(cvals) == 0:
            stats[cid] = (0.0, 1.0, 0.0)
        else:
            stats[cid] = (float(cvals.mean()), float(cvals.std() + 1e-6), float(np.log1p(len(cvals))))
    return cluster_id_map, stats


def process_subject(sub):
    try:
        download_subject_labels(sub)
    except Exception as e:
        print(f"  [skip] {sub}: missing core files ({e})")
        return None
    try:
        t1, affine, mask, cmro2, bold_pct = load_subject_core(sub)
    except Exception as e:
        print(f"  [skip] {sub}: failed to load core files ({e})")
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    x_idx = np.arange(t1.shape[0])
    side_a_x, side_b_x = x_idx < (midpoint - margin), x_idx > (midpoint + margin)

    cmap_a, stats_a = cluster_side(t1, mask, side_a_x, N_CLUSTERS, SEED)
    cmap_b, stats_b = cluster_side(t1, mask, side_b_x, N_CLUSTERS, SEED)
    if cmap_a is None or cmap_b is None:
        print(f"  [skip] {sub}: not enough voxels to cluster")
        return None

    global_mean, global_std = t1[t1 != 0].mean(), t1[t1 != 0].std() + 1e-6
    rng = np.random.default_rng(hash(sub) % (2**31))

    out = {}
    for contrast in CONTRASTS:
        label = build_targets(cmro2, bold_pct, mask, contrast)
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

        cluster_feat = np.zeros((len(used_idx), 3), dtype=np.float32)
        for i, (x, y, z) in enumerate(used_coords):
            if side_tags[i] == "A":
                cid, stats = cmap_a[x, y, z], stats_a
            else:
                cid, stats = cmap_b[x, y, z], stats_b
            mean, std, logsize = stats.get(int(cid), (0.0, 1.0, 0.0))
            cluster_feat[i] = [(mean - global_mean) / global_std, std / global_std, logsize]

        out[contrast] = {"patches": patches, "labels": used_labels, "side": side_tags, "cluster_feat": cluster_feat}

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "cluster_feat": []} for c in CONTRASTS}
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
            for key in ["patches", "labels", "side", "cluster_feat"]:
                pooled[contrast][key].append(d[key])

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        feats = np.concatenate(pooled[contrast]["cluster_feat"], axis=0)
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
                labels[test_idx], pred, probs, f"parcellation {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": acc for (c, f), acc in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(all_results.keys())
    vals = [all_results[k] for k in keys]
    ax.bar([f"{c}\n{f}" for c, f in keys], vals, color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_ylim(0, 1)
    ax.set_title(f"Data-driven parcellation (k-means, k={N_CLUSTERS}) region features")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "parcellation_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
