"""
Real T2-mapping step of the authors' MATLAB mq-BOLD pipeline
(qBOLD_BIDS_Hct_April21.zip, run3_mqBOLD_rOEF.m / calc_T2_map.m,
GitLab: https://gitlab.lrz.de/nmrm_lab/public_projects/mq-bold), ported to
Python and run on the real 8-echo MESE data pulled from OpenNeuro (see
scripts/download_mese.py).

WHAT THIS DOES: per-voxel mono-exponential T2 fit from the 8-echo MESE
series: S(TE) = a * exp(-TE / T2). The source MATLAB code (calc_T2_map.m /
fit2param.m) solves this with a custom bisection line-search on T2 combined
with a closed-form linear solve for the amplitude. Here we use the
standard, equivalent linearized fit instead: log(S) = log(a) - TE/T2, solved
per voxel by ordinary least squares (vectorized, no per-voxel Python loop).
This is a well-established, textbook approach for mono-exponential
relaxometry and should agree closely with the source's bisection result on
well-conditioned (positive, monotonically decaying) signal, but is NOT a
byte-for-byte port - flagged explicitly rather than claimed as identical.

WHAT THIS DOES NOT DO: this is only the T2 (spin-echo) half of R2'. Getting
real R2', OEF and CMRO2 additionally needs, per condition (task vs
baseline): a T2*/T2S map (from a separate multi-echo gradient-echo
acquisition - "MEGRE" - during each task condition), CBF (from pCASL), CBV
(from DSC perfusion with contrast agent), and each subject's Hct. None of
those raw inputs are available (see README.md "Current data status") - this
module intentionally stops at T2, the one piece we have complete real data
for, rather than filling the rest with placeholder numbers.
"""

import numpy as np


def fit_t2_map(echo_volumes, echo_times_ms, mask=None, t2_clip_ms=150.0):
    """
    echo_volumes: (n_echoes, X, Y, Z) array of magnitude images.
    echo_times_ms: (n_echoes,) array of echo times in milliseconds.
    mask: optional (X, Y, Z) boolean array restricting the fit (voxels
        outside are returned as T2=0). If omitted, all voxels with a
        positive first echo are used (matches the source's
        `maske = vol(:,:,:,1) > mean(vol(:,:,:,1))`-style intensity mask,
        implemented here as simply "signal present").

    Returns (t2_map_ms, amplitude_map, fit_error_percent), each shape (X,Y,Z).
    fit_error_percent is mean absolute percent deviation of the fit from the
    data, matching the source's `error`/`devmap` (i.e. the T2.error output).
    """
    echo_volumes = np.asarray(echo_volumes, dtype=np.float64)
    echo_times_ms = np.asarray(echo_times_ms, dtype=np.float64)
    n_echoes = echo_volumes.shape[0]
    if n_echoes != len(echo_times_ms):
        raise ValueError("echo_volumes and echo_times_ms length mismatch")

    shape = echo_volumes.shape[1:]
    if mask is None:
        mask = echo_volumes[0] > 0
    mask = mask.astype(bool)

    signal = echo_volumes.reshape(n_echoes, -1).T          # (n_voxels, n_echoes)
    voxel_mask = mask.reshape(-1) & np.all(signal > 0, axis=1)

    t2_map = np.zeros(signal.shape[0], dtype=np.float64)
    amp_map = np.zeros(signal.shape[0], dtype=np.float64)
    error_map = np.zeros(signal.shape[0], dtype=np.float64)

    log_signal = np.log(signal[voxel_mask])                # (n_valid, n_echoes)
    A = np.stack([np.ones(n_echoes), -echo_times_ms], axis=1)  # log(S) = log(a) - TE/T2
    coeffs, *_ = np.linalg.lstsq(A, log_signal.T, rcond=None)  # (2, n_valid)
    log_a, inv_t2 = coeffs[0], coeffs[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(inv_t2 > 0, 1.0 / inv_t2, 0.0)
    amp = np.exp(log_a)

    predicted = amp[:, None] * np.exp(-echo_times_ms[None, :] / np.where(t2[:, None] == 0, np.inf, t2[:, None]))
    observed = signal[voxel_mask]
    pct_error = 100 * np.abs(observed - predicted).sum(axis=1) / np.abs(observed).sum(axis=1)

    t2_map[voxel_mask] = np.clip(t2, 0, t2_clip_ms)
    amp_map[voxel_mask] = amp
    error_map[voxel_mask] = pct_error

    return t2_map.reshape(shape), amp_map.reshape(shape), error_map.reshape(shape)
