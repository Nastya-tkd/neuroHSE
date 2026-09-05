"""
Last un-tried architectural lever from README's "Where this leaves the
project": a 3D-MRI backbone pretrained on real external data (23andme...
no - Med3D's 23-dataset multi-organ segmentation corpus), fine-tuned here,
rather than one more architecture trained from scratch on our small,
already-exhausted pooled set. Every from-scratch architecture (plain CNN,
residual CNN, transformer hybrid, real U-Net) converged on the same
chance-level result - this is the one remaining lever that brings in
genuinely external data instead of just more parameters.

Weights: Tencent/MedicalNet's resnet_50_23dataset.pth (Med3D, Chen et al.
2019), 46.2M-parameter 3D ResNet50 trunk, obtained from the user's own
GitHub release (not a network host blocked by this session's egress
policy - see conversation) and verified before loading: file header
matches the documented legacy torch.save format, and a static pickle-
opcode scan (no code execution) found only expected torch/collections
reconstruction calls, no suspicious globals.

Architecture: src/medicalnet_resnet.py reproduces the trunk (conv1/bn1/
layer1-4, Bottleneck blocks 3-4-6-3) with an EXACT state_dict key match
verified with strict=True - not a reimplementation that merely looks
similar. conv1 already takes 1 input channel (T1, same as our patches),
so no first-layer surgery is needed, unlike a typical 3-channel ImageNet
backbone. layer3/layer4 use dilation instead of stride (Med3D keeps
resolution high for segmentation), so total spatial downsampling is only
~8x, not the usual 32x - a patch_size=25 input still leaves a 4x4x4
feature map before the final pool, instead of collapsing to nothing.

Approach: FROZEN trunk (feature extraction only, no gradients, no BatchNorm
stat updates - the point is to test whether Med3D's learned general
3D-medical-image features are useful for this label, not to re-derive them)
+ a small trainable MLP head (src/model.py:PretrainedFeatureHead) on the
2048-dim pooled features. Chosen for CPU feasibility: computing 46M-param
trunk forward passes once per patch (not once per epoch) makes the many-
epoch part of training cheap. A full unfrozen fine-tune remains a possible
follow-up if this shows any signal worth chasing.
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
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold
from src.model import PretrainedFeatureHead
from src.medicalnet_resnet import load_pretrained_trunk, extract_backbone_features
from src.subject_loader import load_subject_robust
from src import viz

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_backbone")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "pretrained_weights", "resnet_50_23dataset.pth")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 25  # gives a 4x4x4x2048 trunk feature map (see module docstring); patch=9/15 leave only 2x2x2
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 300
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


def process_subject(sub, backbone):
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [error] {sub}: {e}")
        return None
    if result is None:
        print(f"  [skip] {sub}: {notes}")
        return None
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
            print(f"  [skip contrast] {sub} {contrast}: degenerate label")
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
        feats = extract_backbone_features(backbone, patches)  # (N, 2048) - patches discarded right after
        out[contrast] = {"feats": feats, "labels": labels[used_idx], "side": side_tags}

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading pretrained MedicalNet ResNet50 trunk...")
    backbone = load_pretrained_trunk(CHECKPOINT_PATH)
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Loaded, {n_params:,} params, frozen.")

    pooled = {c: {"feats": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}
    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub, backbone)
        except Exception:
            print(f"  [error] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append({"subject": sub, "status": "skipped"})
            continue
        log.append({"subject": sub, "status": "used", "contrasts": sorted(result.keys())})
        for contrast, d in result.items():
            pooled[contrast]["feats"].append(d["feats"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["feats"]:
            print(f"{contrast}: no usable subjects, skipping")
            continue
        feats = np.concatenate(pooled[contrast]["feats"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        # standardize features using ALL data's mean/std per fold's TRAIN side only, done inside the loop below
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)}/{len(pos_b)} voxels A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            mu = feats[train_idx].mean(axis=0, keepdims=True)
            sd = feats[train_idx].std(axis=0, keepdims=True) + 1e-6
            feats_norm = (feats - mu) / sd

            _, hist, pred, probs = train_one_fold(
                feats_norm[train_idx].astype(np.float32), labels[train_idx],
                feats_norm[test_idx].astype(np.float32), labels[test_idx],
                epochs=30, device="cpu", seed=SEED, model_factory=PretrainedFeatureHead,
            )
            acc = hist["val_acc"][-1]
            best_acc = max(hist["val_acc"])
            print(f"  {fold_name}: final_acc={acc:.3f} best_acc={best_acc:.3f}")
            all_results[(contrast, fold_name)] = {"final_acc": acc, "best_acc": best_acc, "n_subjects": n_subjects}
            viz.plot_training_curves(
                hist, f"pretrained-backbone {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_curves.png"),
            )
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"pretrained-backbone {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = list(all_results.keys())
    vals = [all_results[k]["final_acc"] for k in keys]
    ax.bar([f"{c}\n{f}" for c, f in keys], vals, color="#c0392b")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_ylim(0, 1)
    ax.set_title("Pretrained MedicalNet ResNet50 (frozen) + MLP head, patch=25")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "pretrained_backbone_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
