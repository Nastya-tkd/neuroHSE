import numpy as np
from src.train import train_one_fold_regression


def test_regression_recovers_signal_better_than_chance():
    """Patches whose mean intensity encodes the target should be learnable
    (positive R2), same sanity-check logic as the classification smoke test."""
    rng = np.random.default_rng(0)
    n = 200
    patch_size = 5
    targets = rng.normal(0, 1, n).astype(np.float32)
    patches = np.zeros((n, 1, patch_size, patch_size, patch_size), dtype=np.float32)
    for i in range(n):
        patches[i] = targets[i] + rng.normal(0, 0.05, (patch_size,) * 3)

    split = n // 2
    _, history, pred, y_true = train_one_fold_regression(
        patches[:split], targets[:split], patches[split:], targets[split:],
        epochs=10, device="cpu", seed=0,
    )
    assert history["test_r2"][-1] > 0.5
