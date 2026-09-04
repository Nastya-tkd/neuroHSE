"""
"Fine-tuning" pass requested after the architecture-diversity result: same
real pooled 25-subject data, but with random-flip augmentation, a cosine
LR schedule, and 3x the epochs (45 vs 15) - a legitimate check that the
earlier chance-level results weren't an under-training artifact - PLUS a
real U-Net (src/model.py:PatchUNet, encoder-decoder with skip connections,
reads its answer from the decoder's center-voxel output) alongside the
three architectures from Experiment 1, since a plain classifier-CNN and a
U-Net have genuinely different inductive biases (global pooling vs. dense
skip-connected reconstruction).

Patch size 15 (the "bigger context" size from the earlier extended run) so
the U-Net's two downsample/upsample stages have room to do something.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.model import SimplePatchCNN, DeeperPatchCNN, AttentionPatchCNN, PatchUNet
from src.train import train_one_fold
from src import viz
from scripts.run_pooled_extended import process_subject, CONTRASTS
from src.cohort import ALL_SUBJECTS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "finetune")
PATCH_SIZE = 15
EPOCHS = 25  # up from Experiment 1's 15 - PatchUNet costs ~12x a plain CNN per step, so this stays background-feasible
MAX_POOLED_PER_SIDE = 3000  # post-hoc cap so PatchUNet's cost stays bounded regardless of cohort size
SEED = 0

ARCHITECTURES = {
    "SimplePatchCNN": SimplePatchCNN,
    "DeeperPatchCNN": DeeperPatchCNN,
    "AttentionPatchCNN": lambda: AttentionPatchCNN(patch_size=PATCH_SIZE),
    "PatchUNet": PatchUNet,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": []} for c in CONTRASTS}

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
            pooled[contrast]["patches"].append(d["patches"][PATCH_SIZE])
            pooled[contrast]["labels"].append(d["binary_label"])
            pooled[contrast]["side"].append(d["side"])

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]

        rng = np.random.default_rng(SEED)
        if len(pos_a) > MAX_POOLED_PER_SIDE:
            pos_a = rng.choice(pos_a, size=MAX_POOLED_PER_SIDE, replace=False)
        if len(pos_b) > MAX_POOLED_PER_SIDE:
            pos_b = rng.choice(pos_b, size=MAX_POOLED_PER_SIDE, replace=False)
        print(f"\n=== {contrast}: {len(pos_a)}/{len(pos_b)} voxels A/B (capped at {MAX_POOLED_PER_SIDE}/side) ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            for arch_name, factory in ARCHITECTURES.items():
                _, hist, pred, probs = train_one_fold(
                    patches[train_idx], labels[train_idx], patches[test_idx], labels[test_idx],
                    epochs=EPOCHS, device="cpu", seed=SEED, model_factory=factory,
                    augment=True, lr_schedule=True,
                )
                acc = hist["val_acc"][-1]
                best_acc = max(hist["val_acc"])
                print(f"  {fold_name} {arch_name}: final_acc={acc:.3f} best_acc={best_acc:.3f}")
                all_results[(contrast, fold_name, arch_name)] = {"final_acc": acc, "best_acc": best_acc}
                viz.plot_training_curves(
                    hist, f"finetune {contrast} {fold_name} {arch_name}",
                    os.path.join(OUT_DIR, f"{contrast}_{fold_name}_{arch_name}_curves.png"),
                )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}|{a}": r for (c, f, a), r in all_results.items()}, f, indent=1)

    # summary
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = sorted(set((c, f) for (c, f, a) in all_results.keys()))
    x = np.arange(len(keys))
    width = 0.2
    colors = {"SimplePatchCNN": "#7f8c8d", "DeeperPatchCNN": "#e67e22", "AttentionPatchCNN": "#2980b9", "PatchUNet": "#8e44ad"}
    for i, arch_name in enumerate(ARCHITECTURES):
        vals = [all_results.get((c, f, arch_name), {}).get("final_acc", np.nan) for c, f in keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=arch_name, color=colors[arch_name])
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="target range")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Fine-tuned (augmentation + cosine LR + {EPOCHS} epochs), patch={PATCH_SIZE}, 4 architectures incl. U-Net")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "finetune_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
