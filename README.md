# neuroHSE — structural classification of concordant/discordant voxels

Goal (per the supervisor's task): build a binary classifier that labels each
brain voxel as **concordant** (ΔBOLD and ΔCMRO₂ same sign, "red") or
**discordant** (opposite signs, "blue"), first from structural MRI alone,
later adding the plain BOLD signal.

Reference project/code: [`NeuroenergeticsLab/two_modes_of_hemodynamics`](https://github.com/NeuroenergeticsLab/two_modes_of_hemodynamics)
(Epp et al., *"Two distinct modes of hemodynamic responses across the human
cortex"*, preprint: https://www.biorxiv.org/content/10.1101/2023.12.08.570806).
Data: [OpenNeuro ds004873](https://openneuro.org/datasets/ds004873).

## Step 1 — labeling verified (`src/labeling.py`, `tests/test_labeling.py`)

The concordant/discordant rule was located in the source repo's merged
notebooks (`np.sign(CMRO2_percchange) * np.sign(BOLD_percchange)`, using
empirically measured `BOLD_percchange` and CMRO₂ computed via **Fick's
principle** `CBF × OEF × CaO2` — not via the Davis model) and reimplemented
faithfully in `src/labeling.py`.

**Finding while verifying it against the Davis model:** the source repo's own
`DavisBOLD()` function has a spurious trailing `-1` that breaks round-trip
consistency with its sibling `DavisCMRO2RelChange()` / `DavisCBFRelChange()`
(which correctly invert the canonical Davis et al. 1998 equation). Confirmed
by grep that `DavisBOLD()` is never actually called anywhere in the merged
notebooks — it's unused/dead code, so **this bug does not affect the
concordant/discordant labels** used by the project (those come from measured
BOLD + Fick's-principle CMRO₂, not from `DavisBOLD()`). Documented and
regression-tested in `tests/test_labeling.py::test_source_davis_bold_has_spurious_offset`;
`davis_bold_relchange_standard()` provides the corrected formula, kept
separately from the faithful reproduction.

Could not cross-check against the published methods text directly: this
session's network egress policy blocks `link.springer.com`, `pubmed`/
`ncbi.nlm.nih.gov`, `biorxiv.org` and `openneuro.org` (only GitHub is
reachable). Verification instead relies on the authors' own reference
implementation plus the algebraic self-consistency check above.

Run `python -m pytest tests/ -v` to see all checks pass.

## Step 2 — structural-only classifier (`src/patches.py`, `src/model.py`, `src/train.py`)

- `src/patches.py`: extracts a cubic patch (default 9³ voxels) around each
  labeled voxel; splits one subject's brain into two **leakage-safe** halves
  (left/right hemisphere by default, or anterior/posterior by changing
  `axis_index`) with a margin gap so no train patch can overlap a test patch.
- `src/model.py`: `SimplePatchCNN` — small 3D CNN (2 conv blocks + FC head),
  as instructed: start simple, only move to `DeeperPatchCNN` (residual
  blocks) if accuracy stays near chance.
- `src/train.py`: `run_hemisphere_experiment(...)` trains+evaluates both fold
  directions (A→train/B→test and reverse) and writes diagnostic plots
  (patch examples, split visualization, training curves, confusion
  matrix + ROC, fold-accuracy summary) via `src/viz.py`.

## Current data status

For `sub-p019` we currently have: T1w (native + brain-extracted), brain
mask, a CSF mask, one raw MEGRE echo, and three FSL first-level activation
maps (`calccontrol`, `calcrest`, `memcontrol` — all-positive, thresholded
z-type maps, **not** the signed `BOLD_percchange` the labeling code needs).

**Missing, for all 5 subjects:** the signed `BOLD_percchange` map and the
CMRO₂ maps (task + baseline) needed to compute the real concordant/discordant
label (see `src/dataio.py:SubjectPaths` for the exact expected filenames).
Producing CMRO₂ from raw multi-echo MEGRE data requires the authors' MATLAB
mq-BOLD + DSC-CBV pipeline (`qBOLD_BIDS_Hct_April21.zip` in the source repo),
which is out of scope to reimplement here — these should be pulled from the
`derivatives/` folder on OpenNeuro instead (blocked for direct download in
this session, see above).

`scripts/smoke_test.py` runs the entire pipeline above end-to-end on the real
`sub-p019` T1 using a **synthetic** placeholder label (thresholded T1
intensity + noise — no biological meaning) purely to prove patch extraction,
the hemisphere split, training and plotting all work correctly. Its printed
accuracy is not a scientific result. Once real `BOLD_percchange`/CMRO₂ maps
(or a precomputed label map) are available for the 5 patients, swap the
label source in `build_patch_dataset()` for the real concordance map and
re-run — no other code changes needed.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # verify labeling + patch logic
python scripts/smoke_test.py         # pipeline dry run (synthetic labels)
```

## Next steps (per the supervisor's plan)

1. Get real `BOLD_percchange` + CMRO₂ (or a precomputed concordance map) for
   the 5 subjects.
2. Run `run_hemisphere_experiment` with real labels; target accuracy ~0.65-0.70.
   If near chance, switch to `DeeperPatchCNN`.
3. Experiment 2: add the plain BOLD signal (no contrast agent) as an extra
   input channel/vector alongside the structural patch.
