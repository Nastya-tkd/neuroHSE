"""
Experiment 2: plain BOLD signal as a per-voxel feature vector, alongside the
structural patch.

Input is <sub>_task-all_space-T2_filtered_func.nii.gz` - FSL FEAT's fully
preprocessed functional data (motion correction, spatial smoothing, and
temporal high-pass filtering already applied; this is the same file the
source pipeline's own first-level GLM runs on). Deliberately NOT the
minimally-preprocessed desc-preproc_bold alternative that also exists,
since the supervisor specifically asked for filtered/detrended data, not
"whatever's easiest".

What this module adds on top of that: per-voxel linear detrending (a light,
standard safety net - FEAT's high-pass filter removes slow drift in the
frequency domain, this removes any residual linear trend in the time
domain) and z-scoring (each voxel's own time series to zero mean/unit
variance), so voxels with different raw signal intensity/scanner gain are
comparable before being fed to a network - the same reasoning as
normalizing the structural patches by whole-brain T1 statistics.

What this does NOT do: field-inhomogeneity/dropout-artifact symmetrization
(signal loss near air-tissue boundaries, e.g. orbitofrontal/temporal
regions) mentioned by the supervisor as a real concern for qBOLD/EPI data.
That needs subject-specific field maps and a real distortion-correction
step, not a generic voxel-time-series operation - flagged as an open gap
rather than silently skipped.
"""

import numpy as np
from scipy.signal import detrend


def extract_bold_vectors(bold_4d, coords):
    """
    bold_4d: (X, Y, Z, T) array.
    coords: (N, 3) int array of voxel coordinates.
    Returns (N, T) float32 array, one raw time series per coordinate.
    """
    return bold_4d[coords[:, 0], coords[:, 1], coords[:, 2], :].astype(np.float32)


def normalize_bold_vectors(vectors, eps=1e-6):
    """Per-voxel linear detrend + z-score. vectors: (N, T) -> (N, T) float32."""
    detrended = detrend(vectors, axis=1, type="linear")
    mean = detrended.mean(axis=1, keepdims=True)
    std = detrended.std(axis=1, keepdims=True)
    return ((detrended - mean) / (std + eps)).astype(np.float32)


def parse_events_tsv(path):
    """Reads a BIDS events.tsv (onset, duration, trial_type columns, seconds)."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((float(row["onset"]), float(row["duration"]), row["trial_type"]))
    return rows


def condition_block_indices(events, trial_type, tr, n_timepoints, skip_seconds=4.8):
    """
    Timepoint (TR) indices falling inside blocks of `trial_type`, skipping
    the first `skip_seconds` of each block. The skip accounts for
    hemodynamic lag: BOLD signal takes ~4-6s to rise after a block starts,
    so including those TRs would mix in signal from the *previous* block's
    tail-end response - a deliberate choice, not "however it's usually
    done" (per the supervisor's emphasis on understanding each step).
    """
    indices = []
    for onset, duration, cond in events:
        if cond != trial_type:
            continue
        start = int(np.ceil((onset + skip_seconds) / tr))
        end = int(np.floor((onset + duration) / tr))
        indices.extend(range(max(start, 0), min(end, n_timepoints)))
    return sorted(set(indices))


def compute_condition_features(bold_4d, coords, events, tr, conditions=("calc", "mem", "rest"), skip_seconds=4.8):
    """
    Per-voxel, per-condition percent signal change relative to that voxel's
    whole-run temporal mean: for each condition, mean BOLD during that
    condition's blocks (lag-adjusted, see condition_block_indices) minus
    the run's grand mean, divided by the grand mean.

    "rest" is used here as the label for the non-task baseline condition
    (called "control" elsewhere in this pipeline's file naming) - the
    events.tsv block design only has calc/mem/rest trial types, no
    separate "control" label, and rest is the only non-task condition, so
    this mapping is inferred rather than given explicitly - flagged here
    rather than silently assumed.

    Returns (N, len(conditions)) float32 array.
    """
    series = bold_4d[coords[:, 0], coords[:, 1], coords[:, 2], :].astype(np.float64)  # (N, T)
    grand_mean = series.mean(axis=1)

    feats = np.zeros((series.shape[0], len(conditions)), dtype=np.float64)
    for i, cond in enumerate(conditions):
        idx = condition_block_indices(events, cond, tr, series.shape[1], skip_seconds)
        cond_mean = series[:, idx].mean(axis=1)
        feats[:, i] = (cond_mean - grand_mean) / grand_mean * 100

    return feats.astype(np.float32)
