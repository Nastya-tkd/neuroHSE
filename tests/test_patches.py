import numpy as np
from src.patches import extract_patch, split_by_axis


def test_extract_patch_shape_and_centering():
    vol = np.arange(5 * 5 * 5).reshape(5, 5, 5).astype(np.float32)
    patch = extract_patch(vol, (2, 2, 2), patch_size=3)
    assert patch.shape == (3, 3, 3)
    assert patch[1, 1, 1] == vol[2, 2, 2]


def test_extract_patch_edge_padding():
    vol = np.ones((5, 5, 5), dtype=np.float32)
    patch = extract_patch(vol, (0, 0, 0), patch_size=3)
    assert patch.shape == (3, 3, 3)
    assert patch[1, 1, 1] == 1.0
    assert patch[0, 0, 0] == 0.0  # out of bounds -> zero padded


def test_split_by_axis_no_overlap_with_margin():
    coords = np.array([[i, 0, 0] for i in range(20)])
    side_a, side_b = split_by_axis(coords, axis_index=0, midpoint=10, margin=3)
    assert not np.any(side_a & side_b)
    assert coords[side_a][:, 0].max() < 10 - 3
    assert coords[side_b][:, 0].min() > 10 + 3
