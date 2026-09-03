"""
Experiment 1 (single subject, structural data only): train a small 3D CNN on
patches from one half of the brain (e.g. left hemisphere) to predict
concordant vs discordant, test on the other half, and repeat with the halves
swapped. This is the leakage-safe validation scheme the supervisor asked for
before scaling up to more subjects / adding BOLD as an extra input.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.patches import extract_patches, labeled_voxel_coords, split_by_axis, hemisphere_midpoint
from src.model import SimplePatchCNN
from src import viz


class PatchDataset(TensorDataset):
    pass


def _normalize_patch_intensity(patches, brain_mean, brain_std):
    return (patches - brain_mean) / (brain_std + 1e-6)


def build_patch_dataset(t1_volume, label_volume, brain_mask, patch_size, max_voxels=None, seed=0):
    """
    Returns coords (N,3), patches (N,1,p,p,p) float32, labels (N,) float32 in {0,1}.
    label_volume must already be the {-1,0,+1} concordance map (0 = excluded).
    """
    coords = labeled_voxel_coords(label_volume, brain_mask)
    if max_voxels is not None and len(coords) > max_voxels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_voxels, replace=False)
        coords = coords[idx]

    raw_labels = label_volume[coords[:, 0], coords[:, 1], coords[:, 2]]
    labels = (raw_labels > 0).astype(np.float32)  # 1=concordant, 0=discordant

    patches = extract_patches(t1_volume, coords, patch_size)
    brain_vals = t1_volume[brain_mask.astype(bool)]
    patches = _normalize_patch_intensity(patches, brain_vals.mean(), brain_vals.std())
    patches = patches[:, None, :, :, :].astype(np.float32)  # add channel dim
    return coords, patches, labels


def train_one_fold(train_patches, train_labels, test_patches, test_labels,
                    epochs=15, batch_size=32, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = SimplePatchCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(train_patches), torch.from_numpy(train_labels))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_test = torch.from_numpy(test_patches).to(device)
    y_test = torch.from_numpy(test_labels).to(device)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += ((logits > 0).float() == yb).sum().item()
            total += len(yb)

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            val_acc = ((test_logits > 0).float() == y_test).float().mean().item()

        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_logits > 0).float().cpu().numpy()

    return model, history, test_pred, test_probs


def run_hemisphere_experiment(
    t1_volume, label_volume, brain_mask, affine,
    patch_size=9, axis_index=0, margin_vox=None,
    max_voxels_per_side=1500, epochs=15, out_dir="results", subject_name="subject",
    device="cpu", seed=0,
):
    """
    Runs both fold directions (side A train / side B test, and reverse) for
    one subject, saves diagnostic plots to out_dir, and returns a results dict.
    margin_vox defaults to patch_size // 2 (the minimum needed to guarantee
    no train/test patch overlap across the split).
    """
    os.makedirs(out_dir, exist_ok=True)
    if margin_vox is None:
        margin_vox = patch_size // 2

    coords, patches, labels = build_patch_dataset(
        t1_volume, label_volume, brain_mask, patch_size, max_voxels=None, seed=seed
    )
    midpoint = hemisphere_midpoint(t1_volume.shape, axis_index)
    side_a, side_b = split_by_axis(coords, axis_index, midpoint, margin_vox)

    viz.plot_patch_examples(
        patches[:, 0], labels, os.path.join(out_dir, f"{subject_name}_patch_examples.png")
    )

    mid_slice_idx = t1_volume.shape[2] // 2
    plane_mask = np.abs(coords[:, 2] - mid_slice_idx) <= 2
    axes_2d = [i for i in range(3) if i != 2]
    viz.plot_hemisphere_split(
        t1_volume[:, :, mid_slice_idx],
        coords[plane_mask][:, axes_2d],
        side_a[plane_mask], side_b[plane_mask], midpoint,
        f"{subject_name}: hemisphere split (axis {axis_index}, margin={margin_vox}vox)",
        os.path.join(out_dir, f"{subject_name}_split_axis{axis_index}.png"),
    )

    def subsample(mask):
        idx = np.where(mask)[0]
        if len(idx) > max_voxels_per_side:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=max_voxels_per_side, replace=False)
        return idx

    idx_a, idx_b = subsample(side_a), subsample(side_b)

    fold_results = {}
    for fold_name, (train_idx, test_idx) in {
        "A_train_B_test": (idx_a, idx_b),
        "B_train_A_test": (idx_b, idx_a),
    }.items():
        model, history, test_pred, test_probs = train_one_fold(
            patches[train_idx], labels[train_idx],
            patches[test_idx], labels[test_idx],
            epochs=epochs, device=device, seed=seed,
        )
        viz.plot_training_curves(
            history, f"{subject_name}: {fold_name}",
            os.path.join(out_dir, f"{subject_name}_{fold_name}_curves.png"),
        )
        viz.plot_confusion_and_roc(
            labels[test_idx], test_pred, test_probs, f"{subject_name}: {fold_name}",
            os.path.join(out_dir, f"{subject_name}_{fold_name}_eval.png"),
        )
        fold_results[fold_name] = history["val_acc"][-1]

    viz.plot_fold_accuracy_summary(
        fold_results, os.path.join(out_dir, f"{subject_name}_fold_accuracy_summary.png")
    )
    return fold_results
