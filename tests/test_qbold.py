import numpy as np
from src.qbold import fit_t2_map


def test_fit_t2_map_recovers_known_t2():
    te = np.array([16, 32, 48, 64, 80, 96, 112, 128], dtype=np.float64)
    shape = (4, 4, 3)
    true_t2 = np.full(shape, 60.0)
    true_amp = np.full(shape, 1000.0)

    echoes = np.stack(
        [true_amp * np.exp(-t / true_t2) for t in te], axis=0
    )

    t2_map, amp_map, err_map = fit_t2_map(echoes, te)

    np.testing.assert_allclose(t2_map, true_t2, rtol=1e-3)
    np.testing.assert_allclose(amp_map, true_amp, rtol=1e-3)
    assert np.all(err_map < 0.1)


def test_fit_t2_map_respects_mask():
    te = np.array([16, 32, 48, 64], dtype=np.float64)
    shape = (2, 2, 1)
    amp = np.full(shape, 500.0)
    t2 = np.full(shape, 40.0)
    echoes = np.stack([amp * np.exp(-t / t2) for t in te], axis=0)

    mask = np.zeros(shape, dtype=bool)
    mask[0, 0, 0] = True

    t2_map, _, _ = fit_t2_map(echoes, te, mask=mask)
    assert t2_map[0, 0, 0] > 0
    assert t2_map[1, 1, 0] == 0
