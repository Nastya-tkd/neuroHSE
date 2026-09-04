import numpy as np
from src.bold_features import (
    extract_bold_vectors,
    normalize_bold_vectors,
    condition_block_indices,
    compute_condition_features,
)


def test_extract_bold_vectors():
    bold = np.arange(2 * 2 * 2 * 5).reshape(2, 2, 2, 5).astype(np.float32)
    coords = np.array([[0, 0, 0], [1, 1, 1]])
    vecs = extract_bold_vectors(bold, coords)
    assert vecs.shape == (2, 5)
    np.testing.assert_array_equal(vecs[0], bold[0, 0, 0, :])
    np.testing.assert_array_equal(vecs[1], bold[1, 1, 1, :])


def test_normalize_bold_vectors_zero_mean_unit_std():
    t = np.linspace(0, 10, 50)
    trend = 3 * t + 5
    signal = trend + np.sin(t) * 2
    vectors = np.stack([signal, signal * 2 + 10], axis=0)

    normed = normalize_bold_vectors(vectors)
    np.testing.assert_allclose(normed.mean(axis=1), [0, 0], atol=1e-5)
    np.testing.assert_allclose(normed.std(axis=1), [1, 1], atol=1e-5)


def test_condition_block_indices_skips_lag_and_matches_tr():
    # one 10s "calc" block starting at t=20s, TR=2s, skip 4s of lag
    events = [(20.0, 10.0, "calc")]
    idx = condition_block_indices(events, "calc", tr=2.0, n_timepoints=20, skip_seconds=4.0)
    # block covers t in [20,30); usable window after skip is [24,30) -> TR indices 12,13,14
    assert idx == [12, 13, 14]


def test_condition_block_indices_ignores_other_conditions():
    events = [(0.0, 10.0, "rest"), (10.0, 10.0, "calc")]
    idx = condition_block_indices(events, "mem", tr=1.0, n_timepoints=20, skip_seconds=0.0)
    assert idx == []


def test_compute_condition_features_detects_elevated_condition():
    tr = 1.0
    n_t = 30
    events = [(0.0, 10.0, "rest"), (10.0, 10.0, "calc"), (20.0, 10.0, "mem")]
    # one voxel: baseline 100, +20% during calc (no lag skip so easy to reason about)
    series = np.full(n_t, 100.0)
    series[10:20] = 120.0
    bold_4d = series.reshape(1, 1, 1, n_t)
    coords = np.array([[0, 0, 0]])

    feats = compute_condition_features(
        bold_4d, coords, events, tr, conditions=("calc", "mem", "rest"), skip_seconds=0.0
    )
    assert feats.shape == (1, 3)
    calc_pct, mem_pct, rest_pct = feats[0]
    assert calc_pct > mem_pct
    assert calc_pct > rest_pct
