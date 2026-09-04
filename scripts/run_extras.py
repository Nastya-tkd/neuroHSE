"""
Three follow-ups requested after the "paths to 0.65-0.70" report, run on
the same 25-subject pooled cohort as scripts/run_pooled_cohort.py. All
three reuse PatchBOLDConditionNet (structural patch + small feature
vector) - only the feature vector changes.

1. COVARIATES (age / Hct / sex from participants.tsv) - a real experiment.
2. COARSE REGIONAL CONTEXT - NOT a real anatomical atlas. A genuine Glasser/
   HCP-MMP parcellation was requested but is not obtainable in this
   session: nilearn/NITRC/OSF (where such atlases are normally fetched
   from) and Hugging Face are all blocked by network policy, no FSL
   install ships one locally, and this session's GitHub code/repo search
   is scoped to the one attached repository, not all of GitHub - so no
   atlas file could be located or downloaded. What's implemented instead
   is a coarse geometric grid (T1 mean/std over big anterior-posterior x
   inferior-superior blocks, one per hemisphere-split side so no block
   straddles train/test) - bigger spatial context than a patch, but NOT
   anatomically informed, and reported as such rather than mislabeled.
3. ORACLE / POSITIVE CONTROL - explicitly not a scientific result. Feeds
   the model the real BOLD_percchange AND CMRO2_percchange values that
   define the label directly, so accuracy near 100% is expected by
   construction. Exists only to confirm the training pipeline itself has
   no bug suppressing achievable accuracy - i.e. that Experiments 1-2's
   chance-level numbers reflect an absence of structural signal, not a
   broken training loop. Never to be read as "structure predicts CMRO2."

Item "pretrained 3D-MRI backbone fine-tuned here" from the same list is
NOT attempted: pretrained 3D medical-imaging checkpoints (e.g. MedicalNet)
are not committed to their GitHub repos (confirmed: cloned Tencent/
MedicalNet, no .pth/.pt files, README points to Google Drive/Baidu Pan,
both blocked). Faking "pretrained" via random init would misrepresent the
result, so this item is skipped and reported as blocked, not attempted.
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
from src.covariates import load_participants
from src.model import PatchBOLDConditionNet
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold_multimodal
from src import viz
from scripts.download_real_labels import download_subject_labels, DATA_DIR

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "extras")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0
PARTICIPANTS_PATH = os.path.join(DATA_DIR, ".versions_cache", "participants.tsv")
N_Y_BINS, N_Z_BINS = 3, 2


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
    cmro2_task = cmro2[contrast]
    cmro2_control = cmro2["control"]
    bold = bold_pct[contrast]

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0

    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    binary_label = np.zeros(cmro2_task.shape, dtype=np.float32)
    conc = concordance_label(bold, pct)
    binary_label[valid] = conc[valid]
    return binary_label, pct, bold, valid


def region_bin_maps(shape, side_a_x, side_b_x):
    """Precomputes a (region_id) volume: distinct ids per (side, y_bin, z_bin)
    so no region straddles the hemisphere split."""
    _, ny, nz = shape
    y_edges = np.linspace(0, ny, N_Y_BINS + 1)
    z_edges = np.linspace(0, nz, N_Z_BINS + 1)
    y_bin = np.clip(np.digitize(np.arange(ny), y_edges) - 1, 0, N_Y_BINS - 1)
    z_bin = np.clip(np.digitize(np.arange(nz), z_edges) - 1, 0, N_Z_BINS - 1)
    return y_bin, z_bin


def compute_region_stats(t1, mask, side_mask_x):
    """side_mask_x: boolean (X,) - which x-slices belong to this side.
    Returns dict[(y_bin,z_bin)] -> (mean, std) over T1 voxels in that
    region AND this side, using the WHOLE side's mask (not just sampled
    voxels) - purely structural, no label involved, so no leakage risk."""
    y_bin, z_bin = region_bin_maps(t1.shape, None, None)
    stats = {}
    m = mask.astype(bool)
    for yb in range(N_Y_BINS):
        for zb in range(N_Z_BINS):
            y_idx = np.where(y_bin == yb)[0]
            z_idx = np.where(z_bin == zb)[0]
            sub = t1[np.ix_(np.where(side_mask_x)[0], y_idx, z_idx)]
            sub_mask = m[np.ix_(np.where(side_mask_x)[0], y_idx, z_idx)]
            vals = sub[sub_mask]
            if vals.size < 10:
                stats[(yb, zb)] = (0.0, 1.0)
            else:
                stats[(yb, zb)] = (float(vals.mean()), float(vals.std() + 1e-6))
    return stats, y_bin, z_bin


def region_features_for_coords(coords, stats, y_bin, z_bin, t1):
    feats = np.zeros((len(coords), 2), dtype=np.float32)
    brain_vals = t1[t1 != 0]
    global_mean, global_std = brain_vals.mean(), brain_vals.std() + 1e-6
    for i, (x, y, z) in enumerate(coords):
        mean, std = stats[(y_bin[y], z_bin[z])]
        feats[i] = [(mean - global_mean) / global_std, (std) / global_std]
    return feats


def process_subject(sub, participants):
    try:
        download_subject_labels(sub)
    except Exception as e:
        print(f"  [skip] {sub}: missing core files ({e})")
        return None
    if sub not in participants:
        print(f"  [skip] {sub}: no covariate row")
        return None

    try:
        t1, affine, mask, cmro2, bold_pct = load_subject_core(sub)
    except Exception as e:
        print(f"  [skip] {sub}: failed to load core files ({e})")
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    rng = np.random.default_rng(hash(sub) % (2**31))

    x_idx = np.arange(t1.shape[0])
    side_a_x = x_idx < (midpoint - margin)
    side_b_x = x_idx > (midpoint + margin)
    stats_a, y_bin, z_bin = compute_region_stats(t1, mask, side_a_x)
    stats_b, _, _ = compute_region_stats(t1, mask, side_b_x)

    out = {}
    for contrast in CONTRASTS:
        binary_label, pct, bold, valid = build_targets(cmro2, bold_pct, mask, contrast)
        n_conc, n_disc = int((binary_label > 0).sum()), int((binary_label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        coords, labels = select_labeled_coords(binary_label, mask, max_voxels=None, seed=SEED)
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

        age_mean = np.mean([p["age"] for p in participants.values()])
        age_std = np.std([p["age"] for p in participants.values()]) + 1e-6
        hct_mean = np.mean([p["hct"] for p in participants.values()])
        hct_std = np.std([p["hct"] for p in participants.values()]) + 1e-6
        row = participants[sub]
        cov_vec = np.array([(row["age"] - age_mean) / age_std, (row["hct"] - hct_mean) / hct_std, row["sex"]], dtype=np.float32)
        covariate_feat = np.tile(cov_vec, (len(used_idx), 1))

        region_feat = np.zeros((len(used_idx), 2), dtype=np.float32)
        a_mask, b_mask = side_tags == "A", side_tags == "B"
        if a_mask.any():
            region_feat[a_mask] = region_features_for_coords(used_coords[a_mask], stats_a, y_bin, z_bin, t1)
        if b_mask.any():
            region_feat[b_mask] = region_features_for_coords(used_coords[b_mask], stats_b, y_bin, z_bin, t1)

        pct_vals = pct[used_coords[:, 0], used_coords[:, 1], used_coords[:, 2]]
        bold_vals = bold[used_coords[:, 0], used_coords[:, 1], used_coords[:, 2]]
        oracle_feat = np.stack([bold_vals, pct_vals], axis=1).astype(np.float32)
        oracle_feat = (oracle_feat - oracle_feat.mean(axis=0)) / (oracle_feat.std(axis=0) + 1e-6)

        out[contrast] = {
            "patches": patches, "labels": used_labels, "side": side_tags,
            "covariate_feat": covariate_feat, "region_feat": region_feat, "oracle_feat": oracle_feat,
        }

    return out


def run_variant(name, feat_key, n_features, pooled, out_dir):
    results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        feats = np.concatenate(pooled[contrast][feat_key], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            _, hist, pred, probs = train_one_fold_multimodal(
                patches[train_idx], feats[train_idx], labels[train_idx],
                patches[test_idx], feats[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: PatchBOLDConditionNet(n_bold_features=n_features),
            )
            acc = hist["val_acc"][-1]
            print(f"  [{name}] {contrast} {fold_name}: acc={acc:.3f}")
            results[(contrast, fold_name)] = acc
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"{name}: {contrast} {fold_name}",
                os.path.join(out_dir, f"{name}_{contrast}_{fold_name}_eval.png"),
            )
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    participants = load_participants(PARTICIPANTS_PATH)
    print(f"Loaded {len(participants)} participant covariate rows")

    pooled = {c: {"patches": [], "labels": [], "side": [], "covariate_feat": [], "region_feat": [], "oracle_feat": [], "subject": []} for c in CONTRASTS}
    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub, participants)
        except Exception:
            print(f"  [error] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append((sub, "skipped"))
            continue
        log.append((sub, sorted(result.keys())))
        for contrast, d in result.items():
            for key in ["patches", "labels", "side", "covariate_feat", "region_feat", "oracle_feat"]:
                pooled[contrast][key].append(d[key])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    all_results["covariates"] = run_variant("covariates", "covariate_feat", 3, pooled, OUT_DIR)
    all_results["coarse_region"] = run_variant("coarse_region", "region_feat", 2, pooled, OUT_DIR)
    all_results["oracle_positive_control"] = run_variant("oracle_positive_control", "oracle_feat", 2, pooled, OUT_DIR)

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({k: {f"{c}|{fo}": acc for (c, fo), acc in v.items()} for k, v in all_results.items()}, f, indent=1)

    # summary plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    variants = list(all_results.keys())
    keys = sorted(set(k for v in all_results.values() for k in v.keys()))
    x = np.arange(len(keys))
    width = 0.25
    colors = {"covariates": "#2980b9", "coarse_region": "#e67e22", "oracle_positive_control": "#a6403a"}
    for i, name in enumerate(variants):
        vals = [all_results[name].get(k, np.nan) for k in keys]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors.get(name))
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Covariates / coarse-region context / oracle positive control")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "extras_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")
    print("\nNOTE: pretrained-3D-backbone item was not attempted - no accessible pretrained checkpoint "
          "(see module docstring). oracle_positive_control is a pipeline sanity check, not a finding.")


if __name__ == "__main__":
    main()
