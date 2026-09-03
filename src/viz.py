"""
Plotting helpers for the structural-classification experiment. Colors follow
the same red=concordant / blue=discordant convention used in the source
repo's own BlueRed colormap and in the supervisor's description.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

CONCORDANT_COLOR = "#c0392b"   # red
DISCORDANT_COLOR = "#2471a3"   # blue
NEUTRAL_COLOR = "#5d6d7e"


def plot_hemisphere_split(t1_slice, coords_2d, side_a_mask, side_b_mask, midline, title, out_path):
    """t1_slice: 2D background slice. coords_2d: (N,2) voxel coords in that
    slice's plane. Colors voxels by which split side (or dropped margin)
    they belong to."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(t1_slice.T, cmap="gray", origin="lower")
    dropped = ~(side_a_mask | side_b_mask)
    ax.scatter(*coords_2d[side_a_mask].T, s=2, color="#e67e22", label="side A (train/test)")
    ax.scatter(*coords_2d[side_b_mask].T, s=2, color="#16a085", label="side B (test/train)")
    if dropped.any():
        ax.scatter(*coords_2d[dropped].T, s=2, color="#7f8c8d", alpha=0.5, label="margin (excluded)")
    ax.axvline(midline, color="white", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_patch_examples(patches, labels, out_path, n=8):
    """Middle axial slice of n example patches, labeled concordant/discordant."""
    n = min(n, len(patches))
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    if n == 1:
        axes = [axes]
    mid = patches.shape[-1] // 2
    for ax, patch, label in zip(axes, patches[:n], labels[:n]):
        ax.imshow(patch[:, :, mid], cmap="gray")
        color = CONCORDANT_COLOR if label == 1 else DISCORDANT_COLOR
        ax.set_title("concordant" if label == 1 else "discordant", color=color, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, title, out_path):
    """history: dict with 'train_loss', 'train_acc', 'val_acc' lists."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(history["train_loss"], color=NEUTRAL_COLOR)
    ax1.set_title("training loss")
    ax1.set_xlabel("epoch")

    ax2.plot(history["train_acc"], label="train acc", color="#8e44ad")
    ax2.plot(history["val_acc"], label="test acc", color="#16a085")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    ax2.set_ylim(0, 1)
    ax2.set_title("accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_and_roc(y_true, y_pred, y_score, title, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["discordant", "concordant"], fontsize=8)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["discordant", "concordant"], fontsize=8)
    ax1.set_xlabel("predicted"); ax1.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax1.set_title("confusion matrix")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="#8e44ad", label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax2.set_xlabel("false positive rate"); ax2.set_ylabel("true positive rate")
    ax2.set_title("ROC")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fold_accuracy_summary(fold_results, out_path):
    """fold_results: dict[fold_name] -> test accuracy."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = list(fold_results.keys())
    accs = [fold_results[n] for n in names]
    bars = ax.bar(names, accs, color="#8e44ad")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range (0.65-0.70)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("structural-only classification: accuracy by fold")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
