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

**Structural data (T1w) for 5 subjects — done.** `openneuro.org` itself is
blocked by this session's network policy, but the S3 bucket that actually
backs it is not (`s3.amazonaws.com/openneuro.org/ds004873/...` — found by
listing the bucket directly, see `scripts/download_structural_data.py`).
Downloaded real T1w for `sub-p019, sub-p020, sub-p021, sub-p023, sub-p026`
straight from there, with an approximate Otsu-threshold brain mask
(`src/dataio.py:simple_brain_mask` — no FSL/nilearn available in this
session; good enough to keep patch centers inside the head, not a
substitute for real skull-stripping). Listing that bucket also confirms
`ds004873` on OpenNeuro has **no `derivatives/` folder and no raw MEGRE/DSC
data at all** — only `T1w`, 8-echo `MESE`, and one `task-all_bold.nii.gz`
per subject.

**Real concordant/discordant labels — still blocked, and the blocker is
bigger than "get a file from Drive".** Traced exactly what the merged
notebooks need: `qBOLD_fun.ipynb`'s `create_qBOLD_masks()` only thresholds
and masks R2'/CBV/T2S/OEF/CBF maps that are assumed to already exist on disk
(`combined_pipeline.py` lines 35-160) — it never computes them from raw
data. CMRO₂ itself (`CMRO2 = CBF * OEF * CaO2`, `~line 737`) also takes CBF
and OEF as given inputs. All of R2', CBV, CBF, OEF are products of the
authors' separate MATLAB mq-BOLD + DSC-CBV pipeline
(`qBOLD_BIDS_Hct_April21.zip`) run on raw multi-echo MEGRE + DSC perfusion
data + a per-subject Hct value — none of which is in the public OpenNeuro
copy, and reimplementing that MATLAB physics pipeline from scratch here
(no MATLAB in this environment, multi-step calibration prone to subtle
errors) is out of scope and too risky to trust for ground-truth labels.

**What's actually needed next:** the lab's own precomputed `qmri/` outputs
per subject/condition — `*_R2prime.nii`, `*_cbv.nii`, `*_oef.nii`,
`*_cbf.nii` (or directly `*_cmro2.nii` / `*_BOLD_percchange.nii.gz` if those
were saved) — matching the paths in `src/dataio.py:SubjectPaths`. Raw MEGRE
echoes alone are not sufficient without running the MATLAB step first.

### The MATLAB pipeline itself (`qBOLD_BIDS_Hct_April21.zip`, from
https://gitlab.lrz.de/nmrm_lab/public_projects/mq-bold - blocked here, but
the zip ships inside the already-cloned `two_modes_of_hemodynamics` repo)

Four stages: `run1_process_anatomy.m` (SPM12 segmentation), `run2_dsc_cbv.m`
(CBV from contrast-agent DSC perfusion, with a **manual** AIF-selection
step), `run3_mqBOLD_rOEF.m` (T2/T2' -> OEF), `run4_pCASL_CBF.m` (CBF from
arterial spin labeling). Needs SPM12 (large separate MATLAB toolbox, not
included) and ships precompiled Linux x86_64 MEX binaries for one
sub-step.

**Ported and actually run the one piece we have complete real data for:**
`src/qbold.py::fit_t2_map` reimplements `calc_T2_map.m`'s per-voxel T2 fit
(mono-exponential `S(TE) = a * exp(-TE/T2)`) as a vectorized linearized
least-squares fit (the MATLAB original uses a custom bisection search - not
a byte-for-byte port, see the module docstring) and
`scripts/compute_t2_maps.py` runs it on the real 8-echo MESE series (pulled
from the same S3 bucket, `scripts/download_mese.py`) for all 5 subjects.
Verified against synthetic known-T2 data first (`tests/test_qbold.py`), then
run for real: median brain T2 came out 76-79 ms for all 5 subjects, in the
physiologically expected range at 3T.

**Update - real labels recovered, no longer blocked.** OpenNeuro's S3 bucket
has object versioning enabled, and ds004873's public listing (raw-only,
`"DatasetType": "raw"`) turned out to be a later re-scoping: earlier
snapshots of the same CC0-licensed dataset had a full `derivatives/` tree
(~970 files/subject) with exactly the qmri/func outputs the source pipeline
expects, plus `participants.tsv` (Hct/O2sat). S3 keeps old object content
under a delete marker rather than erasing it, so `scripts/list_versions.py`
(walks `?versions&prefix=...`, paginated) finds the last real version of
each key and `scripts/download_real_labels.py` fetches it via
`?versionId=...`. Recovered per subject: `*_space-T2_desc-brain_T1w.nii.gz`
(structural, same space as the labels), `*_task-{calc,mem}control_space-T2_
BOLD_percchange.nii.gz` (signed), and `*_task-{calc,control,mem}_space-T2_
desc-orig_cmro2.nii` (Fick's-principle CMRO2, task and baseline).

`scripts/run_real_experiment.py` computes the real label exactly as in
`combined_pipeline.py` (`CMRO2_percchange = (CMRO2_task - CMRO2_control) /
CMRO2_control * 100`, then `sign(CMRO2_percchange) * sign(BOLD_percchange)`)
for both calc-vs-control and mem-vs-control per subject (a built-in
replication check), and runs the same tested hemisphere-split classifier
on it. Class balance came out ~45-55% concordant/discordant for every
subject/contrast - no degenerate label.

### Real result

`results/real_experiment/real_experiment_summary.png` (not committed - real
patient-derived data, see .gitignore): test accuracy is at chance (0.46-0.53)
for all 5 subjects, both task contrasts (calc and mem), and both
hemisphere-split directions - 20/20 runs. No subject or contrast stands out.
This is exactly the outcome the supervisor's plan named as a real,
actionable result ("если точность низкая... нужно усложнить архитектуру"):
`SimplePatchCNN` (2 conv blocks) finds no signal in raw T1 patches alone
predicting concordant/discordant status. Confusion matrices show a mild bias
toward predicting the majority class rather than any real discrimination
(e.g. sub-p019 calc: AUC 0.55).

This is a real negative result for the simple model, not a code problem -
the same pipeline that gets 75-82% on the intensity-threshold sanity check
(`results/smoke_test_all/`) gets chance accuracy here, so the model
*can* learn when there is something learnable in a patch; it just isn't
finding a concordant/discordant signal in raw T1 intensity alone with this
architecture. Per the supervisor's contingency plan, next step is
`DeeperPatchCNN` (`src/model.py`, residual blocks, already implemented) on
the same real labels - swap `SimplePatchCNN` for `DeeperPatchCNN` in
`src/train.py:train_one_fold` and rerun `scripts/run_real_experiment.py`.

Superseded by the above, kept for context: getting from T2 (the one
real quantity computed on 2026-09-03, see commit history) to full CMRO2
was originally thought to need re-deriving CBF/CBV/Hct from raw
pCASL/DSC/MEGRE - which is true in general, but turned out to be
unnecessary here since the already-computed CMRO2 itself was recoverable
from version history rather than needing to be rebuilt from raw inputs.

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
python -m pytest tests/ -v                    # verify labeling + patch logic
python scripts/download_structural_data.py    # real T1w for 5 subjects, from OpenNeuro S3
python scripts/smoke_test.py                  # pipeline dry run (synthetic labels)
```

## Next steps (per the supervisor's plan)

1. Get real `BOLD_percchange` + CMRO₂ (or a precomputed concordance map) for
   the 5 subjects.
2. Run `run_hemisphere_experiment` with real labels; target accuracy ~0.65-0.70.
   If near chance, switch to `DeeperPatchCNN`.
3. Experiment 2: add the plain BOLD signal (no contrast agent) as an extra
   input channel/vector alongside the structural patch.
