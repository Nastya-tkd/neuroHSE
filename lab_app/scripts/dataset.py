"""Dataset utilities for the GL261 segmentation lab app.

Builds (image, mask) pairs from a CVAT 1.1 "Segmentation mask" export
(JPEGImages/ + SegmentationClass/ + labelmap.txt per animal folder) and
converts the RGB class-color masks to single-channel class-index masks.

Class layout (fixed, must match model/labels.json and app.py):
    0 = background   (0, 0, 0)
    1 = tumor         (255, 106, 77)   "Опухоль" / enhancing-tumor-like
    2 = edema         (50, 183, 250)   "Отёк" / perifocal edema-like
    3 = necrosis      (255, 0, 204)    "Некроз" / necrosis-like
"""

import os
import glob
import numpy as np
from PIL import Image

CLASS_COLORS = [
    (0, 0, 0),
    (255, 106, 77),
    (50, 183, 250),
    (255, 0, 204),
]
CLASS_NAMES = ["background", "tumor", "edema", "necrosis"]
NUM_CLASSES = len(CLASS_COLORS)


def rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Map an (H, W, 3) color mask to an (H, W) uint8 class-index mask."""
    idx = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        if class_id == 0:
            continue
        match = np.all(mask_rgb == np.array(color), axis=-1)
        idx[match] = class_id
    return idx


def index_mask_to_rgb(mask_idx: np.ndarray) -> np.ndarray:
    """Inverse of rgb_mask_to_index, for rendering overlays."""
    out = np.zeros(mask_idx.shape + (3,), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        out[mask_idx == class_id] = color
    return out


def collect_pairs(marked_root: str):
    """Return a list of dicts: {animal, image_path, mask_path} for every
    slice that has BOTH an exported image and an expert segmentation mask
    (unannotated slices, which exist in JPEGImages without a matching
    SegmentationClass file, are skipped)."""
    pairs = []
    animal_dirs = sorted(
        d for d in glob.glob(os.path.join(marked_root, "*")) if os.path.isdir(d)
    )
    for animal_dir in animal_dirs:
        animal = os.path.basename(animal_dir)
        seg_dir = os.path.join(animal_dir, "SegmentationClass")
        img_dir = os.path.join(animal_dir, "JPEGImages")
        if not (os.path.isdir(seg_dir) and os.path.isdir(img_dir)):
            continue
        for mask_path in sorted(glob.glob(os.path.join(seg_dir, "*.png"))):
            key = os.path.splitext(os.path.basename(mask_path))[0]
            image_path = os.path.join(img_dir, key + ".png")
            if os.path.exists(image_path):
                pairs.append(dict(animal=animal, image_path=image_path, mask_path=mask_path))
    return pairs


def animal_split(pairs, val_animals):
    """Subject-level split: every slice from a val animal goes to val,
    regardless of slice content, to avoid leaking a mouse's appearance
    into both train and validation."""
    val_animals = set(str(a) for a in val_animals)
    train = [p for p in pairs if p["animal"] not in val_animals]
    val = [p for p in pairs if p["animal"] in val_animals]
    return train, val


def load_pair(pair, size):
    """Load one (image, mask) pair, resized to (size, size).
    Image -> float32 [0,1] single-channel array (H, W).
    Mask  -> uint8 class-index array (H, W), nearest-neighbor resized.
    """
    img = Image.open(pair["image_path"]).convert("L").resize((size, size), Image.BILINEAR)
    mask_rgb = Image.open(pair["mask_path"]).convert("RGB").resize((size, size), Image.NEAREST)
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    mask_idx = rgb_mask_to_index(np.asarray(mask_rgb))
    return img_arr, mask_idx
