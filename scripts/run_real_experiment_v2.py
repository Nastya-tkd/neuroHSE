"""
Same real labels as scripts/run_real_experiment.py (real CMRO2 + real
BOLD_percchange, 5 subjects x 2 task contrasts), but comparing all 3
architectures from src/model.py: SimplePatchCNN (already run, chance-level),
DeeperPatchCNN (residual blocks), and AttentionPatchCNN (conv encoder +
transformer self-attention, the "no decoder needed" adaptation of a
U-Net-with-transformer to per-patch classification - see its docstring).

Per the supervisor's own contingency plan: try a deeper/attention model
before concluding there's no structural signal to find.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.model import DeeperPatchCNN, AttentionPatchCNN
from src.train import run_hemisphere_experiment
from scripts.run_real_experiment import load_subject_arrays, build_label, SUBJECTS, CONTRASTS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "real_experiment_v2")

ARCHITECTURES = {
    "DeeperPatchCNN": DeeperPatchCNN,
    "AttentionPatchCNN": AttentionPatchCNN,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}  # (sub, contrast, arch) -> fold_results

    for sub in SUBJECTS:
        arrays = load_subject_arrays(sub)
        for contrast in CONTRASTS:
            label = build_label(arrays, contrast)
            n_conc = int((label > 0).sum())
            n_disc = int((label < 0).sum())
            if min(n_conc, n_disc) < 50:
                continue

            for arch_name, model_cls in ARCHITECTURES.items():
                out_dir = os.path.join(OUT_DIR, f"{sub}_{contrast}_{arch_name}")
                results = run_hemisphere_experiment(
                    t1_volume=arrays["t1"],
                    label_volume=label,
                    brain_mask=arrays["mask"],
                    affine=arrays["affine"],
                    patch_size=9,
                    axis_index=0,
                    max_voxels_per_side=min(1500, min(n_conc, n_disc)),
                    epochs=25,
                    out_dir=out_dir,
                    subject_name=f"{sub}_{contrast}_{arch_name}",
                    model_factory=model_cls,
                )
                all_results[(sub, contrast, arch_name)] = results
                print(f"{sub} {contrast} {arch_name} -> {results}")

    # summary: mean accuracy per architecture (averaged over both fold directions)
    import json
    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{s}|{c}|{a}": r for (s, c, a), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arch_name in ARCHITECTURES:
        accs = []
        for (s, c, a), r in all_results.items():
            if a == arch_name:
                accs.extend([r["A_train_B_test"], r["B_train_A_test"]])
        accs = np.array(accs)
        print(f"{arch_name}: mean={accs.mean():.3f} std={accs.std():.3f} n={len(accs)} runs")

    labels_x = []
    means = {a: [] for a in ARCHITECTURES}
    for sub in SUBJECTS:
        for contrast in CONTRASTS:
            key_present = any((sub, contrast, a) in all_results for a in ARCHITECTURES)
            if not key_present:
                continue
            labels_x.append(f"{sub}\n{contrast}")
            for a in ARCHITECTURES:
                r = all_results.get((sub, contrast, a))
                means[a].append(np.mean([r["A_train_B_test"], r["B_train_A_test"]]) if r else np.nan)

    x = np.arange(len(labels_x))
    width = 0.35
    colors = {"DeeperPatchCNN": "#e67e22", "AttentionPatchCNN": "#2980b9"}
    for i, a in enumerate(ARCHITECTURES):
        ax.bar(x + (i - 0.5) * width, means[a], width, label=a, color=colors[a])
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="supervisor's target range")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean test accuracy (both fold directions)")
    ax.set_title("DeeperPatchCNN vs AttentionPatchCNN on real labels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "architecture_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved {os.path.join(OUT_DIR, 'architecture_comparison.png')}")


if __name__ == "__main__":
    main()
