"""
The one legitimate remaining thing to try, and explicitly the last one:
not a new search over configurations, but a principled combination of
architectures already established as independently null throughout this
project (SimplePatchCNN, DeeperPatchCNN, AttentionPatchCNN, PatchUNet -
src/model.py). Trains all four once each on the exact same leakage-safe
hemisphere split (structural-only, T1, patch=9 - this project's
most-used configuration, reusing scripts/run_full_cohort.py's process_
subject unchanged), then soft-votes their predicted probabilities
(simple average) into one ensemble prediction - a single, one-time
procedure, not a hyperparameter search, so it does not carry the
multiple-comparisons risk that repeatedly trying new configurations
would.

Majority-class baseline computed and reported alongside the ensemble's
own accuracy from the start, same discipline as every experiment since
the ROI-averaged correction. If four independently-null architectures
are all making uncorrelated errors, averaging them could in principle
recover a weak shared signal that no single one could - if they are
just all reflecting the same absence of signal, ensembling will do
nothing, which is the expected outcome given everything else in this
project.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.cohort import ALL_SUBJECTS
from src.model import SimplePatchCNN, DeeperPatchCNN, AttentionPatchCNN, PatchUNet
from src.train import train_one_fold
from src import viz
from scripts.run_full_cohort import process_subject, CONTRASTS, PATCH_SIZE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ensemble")
SEED = 0

ARCHITECTURES = {
    "SimplePatchCNN": SimplePatchCNN,
    "DeeperPatchCNN": DeeperPatchCNN,
    "AttentionPatchCNN": lambda: AttentionPatchCNN(patch_size=PATCH_SIZE),
    "PatchUNet": PatchUNet,
}


def train_one_fold_return_probs(train_patches, train_labels, test_patches, test_labels, model_factory, seed):
    model, hist, pred, probs = train_one_fold(
        train_patches, train_labels, test_patches, test_labels,
        epochs=15, device="cpu", seed=seed, model_factory=model_factory,
    )
    return probs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result, notes = process_subject(sub)
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
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} subjects, {len(pos_a)}/{len(pos_b)} voxels A/B ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            test_labels = labels[test_idx]
            majority_baseline = max(test_labels.mean(), 1 - test_labels.mean())

            all_probs = []
            per_arch_acc = {}
            for arch_name, factory in ARCHITECTURES.items():
                probs = train_one_fold_return_probs(
                    patches[train_idx], labels[train_idx], patches[test_idx], test_labels,
                    model_factory=factory, seed=SEED,
                )
                all_probs.append(probs)
                per_arch_acc[arch_name] = float(((probs > 0.5).astype(np.float32) == test_labels).mean())
                print(f"    {arch_name}: acc={per_arch_acc[arch_name]:.3f}")

            ensemble_probs = np.mean(all_probs, axis=0)
            ensemble_pred = (ensemble_probs > 0.5).astype(np.float32)
            ensemble_acc = float((ensemble_pred == test_labels).mean())
            beats = "YES" if ensemble_acc > majority_baseline else "no"
            print(f"  {fold_name}: ensemble_acc={ensemble_acc:.3f} majority_baseline={majority_baseline:.3f} beats_baseline={beats}")

            all_results[(contrast, fold_name)] = {
                "ensemble_acc": ensemble_acc, "majority_baseline": majority_baseline,
                "per_arch_acc": per_arch_acc, "n_train": len(train_idx), "n_test": len(test_idx), "n_subjects": n_subjects,
            }
            viz.plot_confusion_and_roc(
                test_labels, ensemble_pred, ensemble_probs, f"ensemble {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_ensemble_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(all_results.keys())
    x = np.arange(len(keys))
    width = 0.35
    vals = [all_results[k]["ensemble_acc"] for k in keys]
    base = [all_results[k]["majority_baseline"] for k in keys]
    ax.bar(x - width / 2, vals, width, label="ensemble (4 architectures)", color="#8e44ad")
    ax.bar(x + width / 2, base, width, label="majority-class baseline", color="#8e44ad", alpha=0.4, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("4-architecture soft-vote ensemble vs. its own majority-class baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "ensemble_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
