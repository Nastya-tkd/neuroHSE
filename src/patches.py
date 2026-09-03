"""
3D patch extraction around voxels, and leakage-safe train/test splits.

Per the supervisor's instructions: split one subject's brain into two
independent halves (left/right hemisphere, or anterior/posterior) so that
neighboring voxels never end up on both sides of the train/test split -
a plain random voxel split would leak information because adjacent voxels
share almost all of their patch.
"""

import numpy as np


def extract_patch(volume, center, patch_size):
    """
    Cubic patch of side `patch_size` (must be odd) centered at voxel `center`
    = (i, j, k). Out-of-bounds edges are zero-padded.
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd so the patch has a well-defined center")
    r = patch_size // 2
    i, j, k = center
    patch = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)

    lo = np.array([i - r, j - r, k - r])
    hi = np.array([i + r, j + r, k + r])
    src_lo = np.clip(lo, 0, None)
    src_hi = np.minimum(hi, np.array(volume.shape) - 1)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)

    patch[
        dst_lo[0]:dst_hi[0] + 1, dst_lo[1]:dst_hi[1] + 1, dst_lo[2]:dst_hi[2] + 1
    ] = volume[
        src_lo[0]:src_hi[0] + 1, src_lo[1]:src_hi[1] + 1, src_lo[2]:src_hi[2] + 1
    ]
    return patch


def extract_patches(volume, centers, patch_size):
    """centers: (N, 3) int array of voxel coordinates. Returns (N, p, p, p)."""
    return np.stack([extract_patch(volume, c, patch_size) for c in centers], axis=0)


def labeled_voxel_coords(label_mask, brain_mask=None):
    """
    Voxel coordinates where label_mask is finite and non-zero (i.e. has a
    defined concordant(+1)/discordant(-1) value), optionally restricted to
    brain_mask. Returns (N, 3) int array.
    """
    valid = np.isfinite(label_mask) & (label_mask != 0)
    if brain_mask is not None:
        valid &= brain_mask.astype(bool)
    return np.argwhere(valid)


def split_by_axis(coords, axis_index, midpoint, margin):
    """
    Splits voxel coordinates into two leakage-safe groups along one axis
    (0=x/left-right, 1=y/anterior-posterior, 2=z/inferior-superior).

    Voxels within `margin` voxels of `midpoint` on either side are dropped
    entirely, so no patch on one side can overlap a patch on the other side
    (safe as long as margin >= patch_size // 2).

    Returns (mask_side_a, mask_side_b) boolean arrays over coords' first axis,
    where side_a is coord[axis_index] < midpoint - margin, side_b is
    coord[axis_index] > midpoint + margin.
    """
    vals = coords[:, axis_index]
    side_a = vals < (midpoint - margin)
    side_b = vals > (midpoint + margin)
    return side_a, side_b


def hemisphere_midpoint(volume_shape, axis_index=0):
    """Midline voxel index along the given axis, assuming the volume is
    roughly centered on the brain along that axis (true for these subject
    space images, which are not affine-registered to a symmetric template)."""
    return volume_shape[axis_index] / 2.0
