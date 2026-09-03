"""
SMOKE TEST - NOT A SCIENTIFIC RESULT.

Runs the full Experiment-1 pipeline (patch extraction -> hemisphere split ->
train -> eval -> plots) end-to-end on the one real structural scan we
currently have (sub-p019, T1w) so the code is proven to work before real
concordant/discordant labels are available.

Real labels need CMRO2 (Fick's principle: CBF*OEF*CaO2) and BOLD_percchange
maps for this subject, which have not been provided yet (see README.md,
section "Current data status"). Since they're missing, this script instead
builds a SYNTHETIC label from the T1 intensity itself (voxels above the
whole-brain median intensity, plus noise, are arbitrarily called
"concordant"). This has no biological meaning - it exists only to give the
CNN something learnable so we can confirm patches, splits, training and
plotting all run correctly. Do not report its accuracy as a finding.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.dataio import load_nifti
from src.train import run_hemisphere_experiment

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "subp019")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "smoke_test")


def make_synthetic_label(t1_brain, brain_mask, seed=0):
    """Placeholder label ONLY (see module docstring): not real CMRO2/BOLD
    concordance, just something spatially structured enough to sanity-check
    that the CNN can learn *something* from patches."""
    rng = np.random.default_rng(seed)
    brain_vals = t1_brain[brain_mask.astype(bool)]
    median = np.median(brain_vals)
    noise = rng.normal(0, brain_vals.std() * 0.5, size=t1_brain.shape)
    label = np.zeros(t1_brain.shape, dtype=np.float32)
    concordant = (t1_brain + noise) > median
    label[brain_mask.astype(bool)] = np.where(concordant[brain_mask.astype(bool)], 1.0, -1.0)
    return label


def main():
    t1_path = os.path.join(DATA_DIR, "subp019_descfmriprep_T1w_brain.nii.gz")
    mask_path = os.path.join(DATA_DIR, "subp019_descfmriprep_brain_mask.nii.gz")

    t1, affine, _ = load_nifti(t1_path)
    mask, mask_affine, _ = load_nifti(mask_path)

    if t1.shape != mask.shape:
        raise ValueError(
            f"T1 shape {t1.shape} != brain mask shape {mask.shape}; "
            "resample one onto the other before running the real experiment."
        )

    label = make_synthetic_label(t1, mask)

    print("SMOKE TEST: synthetic label, not a real concordant/discordant map.")
    print(f"T1 shape: {t1.shape}, labeled voxels: {int((mask > 0).sum())}")

    results = run_hemisphere_experiment(
        t1_volume=t1,
        label_volume=label,
        brain_mask=mask,
        affine=affine,
        patch_size=9,
        axis_index=0,          # left/right split
        max_voxels_per_side=800,
        epochs=8,
        out_dir=OUT_DIR,
        subject_name="subp019_SMOKETEST",
    )
    print("Smoke-test fold accuracies (synthetic labels, not scientific):", results)
    print(f"Diagnostic plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
