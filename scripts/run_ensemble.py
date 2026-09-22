"""
Единственная оставшаяся законная вещь, которую стоит попробовать, и
явно последняя: не новый перебор конфигураций, а принципиальное
объединение архитектур, каждая из которых уже показала независимый
нулевой результат на протяжении этого проекта (SimplePatchCNN,
DeeperPatchCNN, AttentionPatchCNN, PatchUNet - src/model.py). Все четыре
обучаются по одному разу на совершенно одинаковом защищённом от утечки
разбиении по полушариям (только структурные данные, T1, patch=9 - самая
часто используемая конфигурация в этом проекте, повторно используется
неизменённая process_subject из scripts/run_full_cohort.py), затем их
предсказанные вероятности объединяются мягким голосованием (простым
усреднением) в одно ансамблевое предсказание - однократная процедура, а
не подбор гиперпараметров, поэтому она не несёт риска множественных
сравнений, который присущ многократным попыткам новых конфигураций.

Базовый уровень «большинство» вычисляется и сообщается наряду с
собственной точностью ансамбля с самого начала, та же дисциплина, что и
во всех экспериментах после исправления с усреднением по ROI. Если все
четыре независимо нулевые архитектуры делают некоррелированные ошибки,
усреднение в принципе могло бы восстановить слабый общий сигнал, который
не мог уловить ни один из них по отдельности - если же они просто все
отражают одно и то же отсутствие сигнала, ансамблирование не даст
ничего, что и является ожидаемым результатом с учётом всего остального
в этом проекте.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.cohort import ALL_SUBJECTS
from src.model import SimplePatchCNN, DeeperPatchCNN, AttentionPatchCNN, PatchUNet
from src.train import train_one_fold
from src import viz
from scripts.run_full_cohort import process_subject, CONTRASTS, PATCH_SIZE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ensemble")
SEED = 0

ARCHITECTURES = {
    "SimplePatchCNN": SimplePatchCNN,
    "DeeperPatchCNN": DeeperPatchCNN,
    "AttentionPatchCNN": lambda: AttentionPatchCNN(patch_size=PATCH_SIZE),
    "PatchUNet": PatchUNet,
}


def train_one_fold_return_probs(train_patches, train_labels, test_patches, test_labels, model_factory, seed):
    model, hist, pred, probs = train_one_fold(
        train_patches, train_labels, test_patches, test_labels,
        epochs=15, device="cpu", seed=seed, model_factory=model_factory,
    )
    return probs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result, notes = process_subject(sub)
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
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)}/{len(pos_b)} вокселей A/B ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            test_labels = labels[test_idx]
            majority_baseline = max(test_labels.mean(), 1 - test_labels.mean())

            all_probs = []
            per_arch_acc = {}
            for arch_name, factory in ARCHITECTURES.items():
                probs = train_one_fold_return_probs(
                    patches[train_idx], labels[train_idx], patches[test_idx], test_labels,
                    model_factory=factory, seed=SEED,
                )
                all_probs.append(probs)
                per_arch_acc[arch_name] = float(((probs > 0.5).astype(np.float32) == test_labels).mean())
                print(f"    {arch_name}: acc={per_arch_acc[arch_name]:.3f}")

            ensemble_probs = np.mean(all_probs, axis=0)
            ensemble_pred = (ensemble_probs > 0.5).astype(np.float32)
            ensemble_acc = float((ensemble_pred == test_labels).mean())
            beats = "ДА" if ensemble_acc > majority_baseline else "нет"
            print(f"  {fold_name}: ensemble_acc={ensemble_acc:.3f} majority_baseline={majority_baseline:.3f} beats_baseline={beats}")

            all_results[(contrast, fold_name)] = {
                "ensemble_acc": ensemble_acc, "majority_baseline": majority_baseline,
                "per_arch_acc": per_arch_acc, "n_train": len(train_idx), "n_test": len(test_idx), "n_subjects": n_subjects,
            }
            viz.plot_confusion_and_roc(
                test_labels, ensemble_pred, ensemble_probs, f"ensemble {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_ensemble_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(all_results.keys())
    x = np.arange(len(keys))
    width = 0.35
    vals = [all_results[k]["ensemble_acc"] for k in keys]
    base = [all_results[k]["majority_baseline"] for k in keys]
    ax.bar(x - width / 2, vals, width, label="ensemble (4 architectures)", color="#8e44ad")
    ax.bar(x + width / 2, base, width, label="majority-class baseline", color="#8e44ad", alpha=0.4, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("4-architecture soft-vote ensemble vs. its own majority-class baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "ensemble_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
