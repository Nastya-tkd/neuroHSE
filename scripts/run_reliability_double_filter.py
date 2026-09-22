"""
Объединяет два конкретных результата Buchel et al. (2026) в один
фильтр, вместо того чтобы проверять только надёжность |CMRO2_percchange|
(как уже делал scripts/run_reliability_filtered.py). Их повторный анализ
показал две вещи: (1) знак concordant/discordant для большинства
вокселей вообще нельзя статистически доверять, и (2) *там, где ему можно
доверять*, положительные ответы BOLD были преимущественно concordant с
метаболизмом, тогда как discordant случаи концентрировались в
отрицательных ответах BOLD - т.е. сама классификация надёжнее для
вокселей с положительным BOLD. Ограничение выборки вокселями,
одновременно надёжными по величине И имеющими положительный BOLD,
должно, если их вывод верен, выделить подмножество этого датасета с
наименее зашумлённой меткой из всех возможных.

Четыре уровня включения, от самого мягкого к самому строгому, на тех же
вокселях/пациентах/архитектуре, что и в run_reliability_filtered.py
(SimplePatchCNN, patch=9, только T1) для прямой сопоставимости:
  - all: каждый размеченный воксель (исходный базовый уровень проекта)
  - top50pct: |CMRO2_percchange| выше медианы обучающей части в рамках
    разбиения (то же самое, что и в эксперименте с одним фильтром,
    включено здесь снова как опорная точка в рамках одного запуска)
  - positive_bold: только BOLD_percchange > 0, без фильтра по величине
  - positive_bold_top50pct: оба условия сразу - двойной фильтр
Пороги по величине подбираются только на обучающей части каждого
разбиения, точно так же, как в run_reliability_filtered.py, поэтому
никакая информация о тестовых метках не просачивается в порог. Фильтр
по знаку BOLD не требует порога (это фиксированный критерий, а не
подобранный по данным), поэтому он в любом случае не несёт такого риска.
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
from src.train import extract_and_normalize_patches, train_one_fold
from src import viz
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reliability_double_filter")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0


def build_label_reliability_bold(cmro2, bold_pct, mask, contrast):
    if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
        return None, None, None
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    label[valid] = concordance_label(bold, pct)[valid]
    reliability = np.abs(pct)
    return label, reliability, bold


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
        label, reliability, bold = build_label_reliability_bold(cmro2, bold_pct, mask, contrast)
        if label is None:
            continue
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            continue

        coords = labeled_voxel_coords(label, mask)
        raw_labels = label[coords[:, 0], coords[:, 1], coords[:, 2]]
        labels = (raw_labels > 0).astype(np.float32)
        rel_vals = reliability[coords[:, 0], coords[:, 1], coords[:, 2]]
        bold_vals = bold[coords[:, 0], coords[:, 1], coords[:, 2]]

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
            "labels": labels[used_idx],
            "side": side_tags,
            "reliability": rel_vals[used_idx],
            "bold": bold_vals[used_idx],
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "reliability": [], "bold": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["reliability"].append(d["reliability"])
            pooled[contrast]["bold"].append(d["bold"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} пациентов использовано")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        reliability = np.concatenate(pooled[contrast]["reliability"], axis=0)
        bold = np.concatenate(pooled[contrast]["bold"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)}/{len(pos_b)} вокселей A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            train_reliability = reliability[train_idx]
            rel_threshold = np.quantile(train_reliability, 0.5)

            tiers = {
                "all": np.ones(len(patches), dtype=bool),
                "top50pct": reliability >= rel_threshold,
                "positive_bold": bold > 0,
                "positive_bold_top50pct": (bold > 0) & (reliability >= rel_threshold),
            }
            for tier_name, keep_mask in tiers.items():
                tr_keep = train_idx[keep_mask[train_idx]]
                te_keep = test_idx[keep_mask[test_idx]]
                if len(tr_keep) < 50 or len(te_keep) < 50 or len(np.unique(labels[tr_keep])) < 2 or len(np.unique(labels[te_keep])) < 2:
                    print(f"  {fold_name} {tier_name}: пропущено (слишком мало вокселей/классов после фильтрации)")
                    continue

                _, hist, pred, probs = train_one_fold(
                    patches[tr_keep], labels[tr_keep], patches[te_keep], labels[te_keep],
                    epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
                )
                acc = hist["val_acc"][-1]
                print(f"  {fold_name} {tier_name}: n_train={len(tr_keep)} n_test={len(te_keep)} acc={acc:.3f}")
                all_results[(contrast, fold_name, tier_name)] = {
                    "acc": acc, "n_train": len(tr_keep), "n_test": len(te_keep), "n_subjects": n_subjects,
                }
                if tier_name == "positive_bold_top50pct":
                    viz.plot_confusion_and_roc(
                        labels[te_keep], pred, probs, f"double-filtered {contrast} {fold_name}",
                        os.path.join(OUT_DIR, f"{contrast}_{fold_name}_{tier_name}_eval.png"),
                    )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}|{t}": r for (c, f, t), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    keys = sorted(set((c, f) for (c, f, t) in all_results.keys()))
    x = np.arange(len(keys))
    width = 0.2
    tier_names = ["all", "top50pct", "positive_bold", "positive_bold_top50pct"]
    colors = {"all": "#7f8c8d", "top50pct": "#e67e22", "positive_bold": "#2980b9", "positive_bold_top50pct": "#c0392b"}
    for i, tier in enumerate(tier_names):
        vals = [all_results.get((c, f, tier), {}).get("acc", np.nan) for c, f in keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=tier, color=colors[tier])
    ax.axhline(0.5, color="gray", linestyle=":", label="случайность")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Двойной фильтр: надёжность x положительный BOLD\n(объединяет оба результата Buchel et al. 2026)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "reliability_double_filter_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
