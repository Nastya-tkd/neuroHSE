"""
Downloads baseline (control-condition) CBV (cerebral blood volume) maps,
recovered via the same S3 version-history mechanism as everything else.

Motivated directly by Alexei Ossadtchi's hypothesis (relayed via the
user): capillary morphology/density determines a tissue's local oxygen-
delivery properties, and this may leave a real, physically-grounded
signature in MRI relaxation times through sub-voxel compartmental
averaging (blood vs. tissue T1/T2 differ, so a voxel's blood-volume
fraction measurably shifts its observed relaxation - the same physical
mechanism BOLD/DSC imaging itself relies on), even though individual
capillaries are far below voxel resolution. Of the maps this project has
used, CBV (blood volume) is the more direct macroscopic proxy for
capillary density/vascular morphology than CBF (flow) or T1 alone -
untested until now.

Same non-circularity logic as scripts/download_cbf_oef.py: only the
*control*-condition (baseline) CBV is used, never task-condition values -
the concordant/discordant label is defined by the change between task
and control, so a baseline map alone isn't part of that computation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.list_versions import get_or_build_version_map
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR


def download_subject_cbv(subject, condition="control", versions_cache_dir=VERSIONS_CACHE_DIR, data_dir=DATA_DIR):
    suffix = f"_task-{condition}_space-T2_cbv.nii"
    cache_path = os.path.join(versions_cache_dir, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(data_dir, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)

    matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
    if not matches:
        return None
    key = matches[0]
    version_id, last_modified = version_map[key]
    fname = key.split("/")[-1]
    out_path = os.path.join(out_dir, fname)
    download_versioned(key, version_id, out_path)
    return out_path


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["sub-p019"]
    for sub in subjects:
        p = download_subject_cbv(sub, "control")
        print(f"{sub}: {p}")
