"""
Registers the real Glasser/HCP-MMP1.0 cortical parcellation (360 areas,
180/hemisphere) into each subject's own T2-space grid, via ANTsPy SyN
registration against a bundled MNI152 T1 template - no external atlas
host or registration software needed beyond two PyPI packages (antspyx,
nilearn), both reachable in this session.

Honesty note, directly from the atlas's own maintainer (see
atlas_cache/ORIGIN.md and https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL):
HCP-MMP1.0 was built and validated for *surface-based* registration
(FreeSurfer + Connectome Workbench); using it via *volumetric* MNI
registration, as this module does, is explicitly flagged by the atlas's
creators (citing Coalson, Van Essen & Glasser 2018, PNAS) as introducing
real boundary imprecision - this is a coarse volumetric approximation of
Glasser, not the methodologically preferred surface-based version. Also
note the atlas was mapped onto an ICBM2009c-like template, while the
registration target here is nilearn's bundled MNI152 template (a
different, though closely related, MNI variant) - a second source of
approximation. Reported as such, not oversold as "the" atlas.

Registration takes ~5-10s/subject (SyN, low-res 2mm template) - cheap
enough to just compute per subject and cache to disk, not worth
pre-baking into the repo.
"""

import os
import numpy as np

ATLAS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "atlas_cache")
ATLAS_LABELS_PATH = os.path.join(ATLAS_CACHE_DIR, "MNI_Glasser_HCP_v1.0.nii.gz")
MNI_TEMPLATE_PATH = os.path.join(ATLAS_CACHE_DIR, "mni152_t1_2mm.nii.gz")

_mni_template = None
_atlas_image = None


def _ensure_mni_template():
    """Bundled with nilearn (no network fetch) - cached to disk once as a plain nifti."""
    if os.path.exists(MNI_TEMPLATE_PATH):
        return
    import nibabel as nib
    from nilearn import datasets
    os.makedirs(ATLAS_CACHE_DIR, exist_ok=True)
    mni_nib = datasets.load_mni152_template(resolution=2)
    nib.save(mni_nib, MNI_TEMPLATE_PATH)


def get_subject_glasser_atlas(subject, t1_path, cache_dir=ATLAS_CACHE_DIR):
    """
    Returns an (X,Y,Z) int array, same shape/grid as the subject's T1
    (space-T2), with Glasser parcel labels (0=background/non-cortex,
    1-180=left hemisphere areas, 1000-1180=right hemisphere areas).
    Cached to disk per subject after the first call.
    """
    import ants

    out_path = os.path.join(cache_dir, f"{subject}_glasser_T2space.nii.gz")
    if os.path.exists(out_path):
        img = ants.image_read(out_path)
        return img.numpy().astype(np.int32)

    if not os.path.exists(ATLAS_LABELS_PATH):
        raise FileNotFoundError(
            f"{ATLAS_LABELS_PATH} missing - clone github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL "
            "and copy MNI_Glasser_HCP_v1.0.nii.gz there first."
        )
    _ensure_mni_template()

    subj_img = ants.image_read(t1_path)
    mni_img = ants.image_read(MNI_TEMPLATE_PATH)
    atlas_img = ants.image_read(ATLAS_LABELS_PATH)

    reg = ants.registration(fixed=subj_img, moving=mni_img, type_of_transform="SyN")
    warped = ants.apply_transforms(
        fixed=subj_img, moving=atlas_img, transformlist=reg["fwdtransforms"], interpolator="genericLabel"
    )
    os.makedirs(cache_dir, exist_ok=True)
    ants.image_write(warped, out_path)
    return warped.numpy().astype(np.int32)
