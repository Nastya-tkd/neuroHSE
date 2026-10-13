"""
Проверка идеи, предложенной после разбора MS-DSA-NET (модель сегментации
ФКД, link.springer.com/article/10.1186/s13244-024-01803-8): дать патчевому
классификатору трансформер-путь не только на самом грубом масштабе (как в
AttentionPatchCNN), а на НЕСКОЛЬКИХ масштабах сразу (MultiScaleAttentionPatchCNN,
см. src/model.py) - облегчённый перенос их идеи "несколько трансформер-путей
на разных масштабах энкодера" без декодера, без MONAI и без MNI/FSL-регистрации
(не нужны для задачи "одна метка на патч").

Сравнивает три архитектуры при полностью идентичных вокселях/патчах (T1,
patch=9, вся пригодная когорта через load_subject_robust, как в
run_full_cohort.py):
  - SimplePatchCNN (базовая свёртка, долгоживущий baseline проекта).
  - AttentionPatchCNN (трансформер только на самом грубом масштабе).
  - MultiScaleAttentionPatchCNN (трансформер на двух масштабах).

Базовый уровень «большинство» вычисляется и сообщается наряду с точностью
всех трёх моделей с самого начала - та же дисциплина, что и везде в проекте.
Это разведочная проверка архитектуры, а не повторение гипотезы о новом
входном канале: вход везде один и тот же (T1), меняется только модель.

Только контраст "calc" и 8 эпох (не 15, как в остальных скриптах) - при
трёх моделях на fold вместо одной прогон на CPU оказался намного дольше
ожидаемого (один fold контраста calc с 15 эпохами не уложился в час
фонового времени), так что объём обязательно урезан, а не просто оставлен
"на сколько хватит времени" - то же решение и по той же причине, что и в
scripts/run_dense_segmentation.py.
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
from src.patches import split_by_axis, hemisphere_midpoint
from src.model import SimplePatchCNN, AttentionPatchCNN, MultiScaleAttentionPatchCNN
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold
from src import viz
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "multiscale_transformer")
CONTRASTS = ["calc"]  # "mem" пропущен - бюджет CPU-времени, см. docstring модуля
PATCH_SIZE = 9
TRAIN_EPOCHS = 8  # не 15 - бюджет CPU-времени, см. docstring модуля
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0
MODELS = {
    "simple_cnn": SimplePatchCNN,
    "attention_single_scale": AttentionPatchCNN,
    "attention_multiscale": MultiScaleAttentionPatchCNN,
}


def build_label(cmro2, bold_pct, mask, contrast):
    if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
        return None
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    label[valid] = concordance_label(bold, pct)[valid]
    return label


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
        label = build_label(cmro2, bold_pct, mask, contrast)
        if label is None:
            continue
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        coords, labels = select_labeled_coords(label, mask, max_voxels=None, seed=SEED)
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
        out[contrast] = {"patches": patches, "labels": labels[used_idx], "side": side_tags}

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} пациентов использовано")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            print(f"{contrast}: нет подходящих пациентов, пропуск")
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)} вокселей стороны A, {len(pos_b)} вокселей стороны B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            test_labels = labels[test_idx]
            majority_baseline = float(max(test_labels.mean(), 1 - test_labels.mean()))
            row = {"majority_baseline": majority_baseline, "n_subjects": n_subjects,
                   "n_train": len(train_idx), "n_test": len(test_idx)}

            accs = {}
            for model_name, model_factory in MODELS.items():
                _, hist, pred, probs = train_one_fold(
                    patches[train_idx], labels[train_idx], patches[test_idx], test_labels,
                    epochs=TRAIN_EPOCHS, device="cpu", seed=SEED, model_factory=model_factory,
                )
                acc = hist["val_acc"][-1]
                accs[model_name] = acc
                row[model_name] = acc
                viz.plot_confusion_and_roc(
                    test_labels, pred, probs, f"multiscale {contrast} {fold_name}: {model_name}",
                    os.path.join(OUT_DIR, f"{contrast}_{fold_name}_{model_name}_eval.png"),
                )

            beats = {m: ("ДА" if a > majority_baseline else "нет") for m, a in accs.items()}
            print(f"  {fold_name}: " + "  ".join(f"{m}={a:.3f}({beats[m]})" for m, a in accs.items())
                  + f"  базовый_уровень={majority_baseline:.3f}")
            all_results[(contrast, fold_name)] = row

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    labels_x = [f"{c}\n{f}" for (c, f) in all_results.keys()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels_x))
    width = 0.2
    colors = {"simple_cnn": "#7f8c8d", "attention_single_scale": "#2980b9", "attention_multiscale": "#c0392b"}
    for i, model_name in enumerate(MODELS.keys()):
        vals = [all_results[k][model_name] for k in all_results]
        ax.bar(x + (i - 1) * width, vals, width, label=model_name, color=colors[model_name])
    base_vals = [all_results[k]["majority_baseline"] for k in all_results]
    ax.bar(x + 2 * width, base_vals, width, label="базовый уровень «большинство»",
           color="#c0392b", alpha=0.3, hatch="//")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайность")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("Multi-scale трансформер (идея MS-DSA-NET) против однo-масштабного и обычной CNN")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "multiscale_transformer_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
