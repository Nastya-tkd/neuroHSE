import numpy as np
from src.bold_features import extract_bold_vectors, normalize_bold_vectors


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
