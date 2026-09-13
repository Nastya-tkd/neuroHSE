"""Train the lab baseline segmentation model on the Mice_gbm annotated
export (JPEGImages + SegmentationClass per animal). CPU-only, small
dataset -- meant to finish in well under an hour on a 4-core machine.

Usage:
    python train.py --data-root /path/to/mice_gbm/marked --epochs 80

The "marked root" is a directory containing one subfolder per animal,
each with JPEGImages/, SegmentationClass/, labelmap.txt (the CVAT 1.1
"Segmentation mask" export format used for this dataset).
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import CLASS_NAMES, NUM_CLASSES, collect_pairs, animal_split, load_pair
from model_def import SmallUNet, count_params

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "model"))


class SegDataset(Dataset):
    def __init__(self, pairs, size, augment=False):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = load_pair(self.pairs[i], self.size)
        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.3:
                k = random.choice([1, 2, 3])
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
        img_t = torch.from_numpy(img).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


def dice_loss(logits, target, num_classes, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    target_1h = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * target_1h, dims)
    union = torch.sum(probs + target_1h, dims)
    dice_per_class = (2 * inter + eps) / (union + eps)
    return 1.0 - dice_per_class.mean()


def class_pixel_weights(pairs, size, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for p in pairs:
        _, mask = load_pair(p, size)
        for c in range(num_classes):
            counts[c] += np.sum(mask == c)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def dice_per_class(logits, target, num_classes, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    out = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        out.append(1.0 if denom == 0 else (2 * inter + eps) / (denom + eps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder with one subdir per animal (JPEGImages/, SegmentationClass/)")
    ap.add_argument("--val-animals", default="8,13,17", help="comma-separated animal folder names held out for validation")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(MODEL_DIR, exist_ok=True)

    pairs = collect_pairs(args.data_root)
    if not pairs:
        raise SystemExit(f"No (image, mask) pairs found under {args.data_root}")
    val_animals = [a.strip() for a in args.val_animals.split(",") if a.strip()]
    train_pairs, val_pairs = animal_split(pairs, val_animals)
    print(f"Total pairs: {len(pairs)}  train: {len(train_pairs)}  val: {len(val_pairs)}")
    print(f"Val animals: {val_animals}")

    print("Computing class pixel weights on the training split...")
    weights = class_pixel_weights(train_pairs, args.size, NUM_CLASSES)
    print("Class weights:", dict(zip(CLASS_NAMES, [round(w, 3) for w in weights.tolist()])))

    train_ds = SegDataset(train_pairs, args.size, augment=True)
    val_ds = SegDataset(val_pairs, args.size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = SmallUNet(in_ch=1, num_classes=NUM_CLASSES).to(device)
    print(f"Model params: {count_params(model):,}")

    ce = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_mean_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for img, mask in train_dl:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = ce(logits, mask) + dice_loss(logits, mask, NUM_CLASSES)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)
        sched.step()

        model.eval()
        val_loss = 0.0
        dice_sum = np.zeros(NUM_CLASSES)
        n_batches = 0
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                loss = ce(logits, mask) + dice_loss(logits, mask, NUM_CLASSES)
                val_loss += loss.item() * img.size(0)
                dice_sum += np.array(dice_per_class(logits, mask, NUM_CLASSES))
                n_batches += 1
        val_loss /= max(len(val_ds), 1)
        dice_mean_per_class = dice_sum / max(n_batches, 1)
        mean_fg_dice = dice_mean_per_class[1:].mean()  # exclude background

        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"dice(tumor/edema/necrosis)={dice_mean_per_class[1]:.3f}/{dice_mean_per_class[2]:.3f}/{dice_mean_per_class[3]:.3f}  "
              f"mean_fg_dice={mean_fg_dice:.3f}  ({dt:.1f}s)")

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                             dice_tumor=float(dice_mean_per_class[1]),
                             dice_edema=float(dice_mean_per_class[2]),
                             dice_necrosis=float(dice_mean_per_class[3]),
                             mean_fg_dice=float(mean_fg_dice)))

        if mean_fg_dice > best_mean_dice:
            best_mean_dice = mean_fg_dice
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet_gl261.pt"))
            print(f"  -> saved new best checkpoint (mean_fg_dice={mean_fg_dice:.3f})")

    meta = dict(
        input_size=args.size,
        in_channels=1,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        base_channels=24,
        val_animals=val_animals,
        best_mean_fg_dice=best_mean_dice,
        train_pairs=len(train_pairs),
        val_pairs=len(val_pairs),
        epochs=args.epochs,
    )
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\nBest mean foreground Dice (val): {best_mean_dice:.3f}")
    print(f"Checkpoint: {os.path.join(MODEL_DIR, 'unet_gl261.pt')}")


if __name__ == "__main__":
    main()
