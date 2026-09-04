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
