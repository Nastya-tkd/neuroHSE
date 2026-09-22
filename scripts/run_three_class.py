"""
Переформулирует задачу как 3-классовую (concordant / discordant /
unreliable) вместо того, чтобы принудительно относить каждый воксель к
одному из двух классов concordant/discordant - более честное отражение
результата Buchel et al. (2026) (см. README/отчёт 06j-06k) о том, что
для значительной доли вокселей истинный знак вообще статистически не
определим. Вместо того чтобы молча оставлять эти воксели в 2-классовой
задаче (неявно утверждая, что у них есть определимый истинный знак) или
молча отбрасывать их (как делали эксперименты с фильтрацией по
надёжности), здесь модели даётся явный третий вариант, и она оценивается
по тому, способна ли она вообще отличить надёжные воксели от ненадёжных,
а не только правильно определить знак у тех, что надёжны.

Класс 2 ("unreliable") = нижняя половина *обучающих* вокселей каждого
разбиения по |CMRO2_percchange| (тот же прокси по величине, что и в
run_reliability_filtered.py, та же дисциплина подбора порога только на
обучающей части, чтобы никакая информация о тестовых метках не
просачивалась в порог); классы 0/1 = discordant/concordant для вокселей
выше этого порога. Та же архитектура (SimplePatchCNN, patch=9, теперь с
n_classes=3 и CrossEntropyLoss через train_one_fold_multiclass) и тот же
протокол отбора вокселей, что и в любом другом запуске только по
структурным данным в этом проекте, для прямой сопоставимости.

Сообщается как стандартная 3-классовая точность (уровень случайного
угадывания = 1/3, а не 1/2 - это явно отмечено во врезке на сводном
графике), так и вторичное, более интерпретируемое число: бинарная
точность определения знака, ограниченная вокселями, которые в тестовой
выборке были *действительно* надёжны (строгая оценка «угадал нужную
сторону монеты И правильно распознал, что её вообще можно подбрасывать»).
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.cohort import ALL_SUBJECTS
from src.labeling import concordance_label
from src.patches import split_by_axis, hemisphere_midpoint, labeled_voxel_coords
from src.model import SimplePatchCNN
from src.train import extract_and_normalize_patches, train_one_fold_multiclass
from src import viz
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "three_class")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0
CLASS_NAMES = ["discordant", "concordant", "unreliable"]


def build_label_and_reliability(cmro2, bold_pct, mask, contrast):
    if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
        return None, None
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    label[valid] = concordance_label(bold, pct)[valid]
    reliability = np.abs(pct)
    return label, reliability


def process_subject(sub):
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [ошибка] {sub}: {e}")
        return None
    if result is None:
        print(f"  [пропуск] {sub}: {notes}")
        return None
    t1, affine, mask, cmro2, bold_pct = result

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    rng = np.random.default_rng(hash(sub) % (2**31))

    out = {}
    for contrast in CONTRASTS:
        label, reliability = build_label_and_reliability(cmro2, bold_pct, mask, contrast)
        if label is None:
            continue
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        coords = labeled_voxel_coords(label, mask)
        raw_labels = label[coords[:, 0], coords[:, 1], coords[:, 2]]
        binary_labels = (raw_labels > 0).astype(np.int64)  # 1=concordant, 0=discordant (до разделения по unreliable)
        rel_vals = reliability[coords[:, 0], coords[:, 1], coords[:, 2]]

        side_a, side_b = split_by_axis(coords, 0, midpoint, margin)

        def subsample(m):
            idx = np.where(m)[0]
            if len(idx) > MAX_VOXELS_PER_SIDE_PER_SUBJECT:
                idx = rng.choice(idx, size=MAX_VOXELS_PER_SIDE_PER_SUBJECT, replace=False)
            return idx

        idx_a, idx_b = subsample(side_a), subsample(side_b)
        used_idx = np.concatenate([idx_a, idx_b])
        used_coords = coords[used_idx]
        side_tags = np.array(["A"] * len(idx_a) + ["B"] * len(idx_b))

        patches = extract_and_normalize_patches(t1, used_coords, PATCH_SIZE, mask)

        out[contrast] = {
            "patches": patches,
            "binary_labels": binary_labels[used_idx],
            "side": side_tags,
            "reliability": rel_vals[used_idx],
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "binary_labels": [], "side": [], "reliability": [], "subject": []} for c in CONTRASTS}

    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub)
        except Exception:
            print(f"  [ошибка] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append({"subject": sub, "status": "skipped"})
            continue
        log.append({"subject": sub, "status": "used", "contrasts": sorted(result.keys())})
        for contrast, d in result.items():
            pooled[contrast]["patches"].append(d["patches"])
            pooled[contrast]["binary_labels"].append(d["binary_labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["reliability"].append(d["reliability"])
            pooled[contrast]["subject"].append(np.full(len(d["binary_labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} пациентов использовано")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        binary_labels = np.concatenate(pooled[contrast]["binary_labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        reliability = np.concatenate(pooled[contrast]["reliability"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)}/{len(pos_b)} вокселей A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            rel_threshold = np.quantile(reliability[train_idx], 0.5)
            three_class = np.where(reliability >= rel_threshold, binary_labels, 2).astype(np.int64)

            _, hist, pred, probs = train_one_fold_multiclass(
                patches[train_idx], three_class[train_idx], patches[test_idx], three_class[test_idx],
                n_classes=3, epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: SimplePatchCNN(n_classes=3),
            )
            acc = hist["val_acc"][-1]

            true_test, pred_test = three_class[test_idx], pred
            reliable_mask = true_test != 2
            if reliable_mask.sum() > 0:
                strict_acc = float((pred_test[reliable_mask] == true_test[reliable_mask]).mean())
            else:
                strict_acc = float("nan")

            print(f"  {fold_name}: точность 3 классов acc={acc:.3f} (случайный уровень=0.333)  строгая бинарная точность на надёжных acc={strict_acc:.3f}")
            all_results[(contrast, fold_name)] = {
                "three_class_acc": acc, "strict_reliable_acc": strict_acc,
                "n_train": len(train_idx), "n_test": len(test_idx), "n_subjects": n_subjects,
            }
            viz.plot_multiclass_confusion(
                true_test, pred_test, CLASS_NAMES, f"3-class {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_confusion.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(all_results.keys())
    x = np.arange(len(keys))
    width = 0.35
    three_vals = [all_results[k]["three_class_acc"] for k in keys]
    strict_vals = [all_results[k]["strict_reliable_acc"] for k in keys]
    ax.bar(x - width / 2, three_vals, width, label="точность (3 класса)", color="#8e44ad")
    ax.bar(x + width / 2, strict_vals, width, label="бинарная точность на подмножестве истинно-надёжных", color="#16a085")
    ax.axhline(1 / 3, color="gray", linestyle=":", label="случайность для 3 классов (0.333)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="случайность для бинарной (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Постановка задачи с 3 классами (concordant/discordant/unreliable)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "three_class_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
