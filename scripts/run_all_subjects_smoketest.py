"""
SMOKE TEST ACROSS ALL 5 SUBJECTS - NOT A SCIENTIFIC RESULT.

Same caveat as scripts/smoke_test.py, extended to all 5 subjects we have
real T1w for. Labels are still the synthetic T1-intensity-threshold
placeholder (no biological meaning) because real concordant/discordant
labels are still blocked on missing CBF/CBV/Hct data (see README.md).

This exists to show the pipeline is not a one-subject fluke - it runs
correctly end to end on every real subject - and to produce a cross-subject
summary plot. It does NOT show that the model can detect anything about
concordant/discordant voxels for the reasons explained in README.md and in
the chat: without a real label, there is nothing biological here to detect.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.dataio import load_nifti
from src.train import run_hemisphere_experiment
from scripts.smoke_test import make_synthetic_label

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "smoke_test_all")

SUBJECTS = ["sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for sub in SUBJECTS:
        t1_path = os.path.join(DATA_DIR, sub, "anat", f"{sub}_T1w.nii.gz")
        mask_path = os.path.join(DATA_DIR, sub, "anat", f"{sub}_brain_mask.nii.gz")

        t1, affine, _ = load_nifti(t1_path)
        mask, _, _ = load_nifti(mask_path)
        label = make_synthetic_label(t1, mask, seed=hash(sub) % (2**31))

        print(f"=== {sub}: SMOKE TEST (synthetic label) ===")
        results = run_hemisphere_experiment(
            t1_volume=t1,
            label_volume=label,
            brain_mask=mask,
            affine=affine,
            patch_size=9,
            axis_index=0,
            max_voxels_per_side=600,
            epochs=8,
            out_dir=os.path.join(OUT_DIR, sub),
            subject_name=f"{sub}_SMOKETEST",
        )
        all_results[sub] = results
        print(sub, results)

    # cross-subject summary
    fig, ax = plt.subplots(figsize=(8, 4))
    subs = list(all_results.keys())
    a_vals = [all_results[s]["A_train_B_test"] for s in subs]
    b_vals = [all_results[s]["B_train_A_test"] for s in subs]
    x = np.arange(len(subs))
    width = 0.35
    ax.bar(x - width / 2, a_vals, width, label="A train / B test", color="#8e44ad")
    ax.bar(x + width / 2, b_vals, width, label="B train / A test", color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(subs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("SMOKE TEST (synthetic labels, not biological) - all 5 subjects")
    ax.legend(fontsize=8)
    fig.tight_layout()
    summary_path = os.path.join(OUT_DIR, "cross_subject_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nCross-subject summary saved to {summary_path}")


if __name__ == "__main__":
    main()
