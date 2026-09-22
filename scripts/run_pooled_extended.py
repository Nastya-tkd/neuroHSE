"""
Два дополнительных эксперимента на тех же реальных данных объединённой
когорты, что и в scripts/run_pooled_cohort.py (25/40 пациентов, см. этот
скрипт и README.md для объяснения), оба по прямому запросу пользователя:

1. Более широкий пространственный контекст: patch_size 15 (~30x30x50 мм
   физически, против исходных 9 -> ~18x18x30 мм) вместо небольшого
   локального куба - шаг «признаки в масштабе всего ROI», названный в
   плане самого руководителя как запасной вариант на случай, если одно
   лишь «больше данных» не станет решением.
2. Регрессия вместо классификации: предсказание непрерывного значения
   CMRO2_percchange напрямую (src/train.py:train_one_fold_regression)
   вместо приведения его к знаку и объединения со знаком BOLD.
   Сохраняет информацию о величине, которую бинарная метка отбрасывает,
   и представляет собой содержательно другой вопрос («насколько сильно
   изменился CMRO2» против «изменился ли он в ту же сторону, что и
   BOLD»).

Данные каждого пациента скачиваются ОДИН РАЗ, а патчи извлекаются ОБОИХ
размеров (9 и 15), так что все 4 комбинации (классификация/регрессия x
патч 9/15) напрямую сопоставимы в рамках одного прохода по данным, а не
четырёх отдельных, потенциально несогласованных скачиваний. То же
защищённое от утечки разбиение объединённой когорты по полушариям, что
и раньше.
"""

import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.cohort import ALL_SUBJECTS
from src.dataio import load_nifti
from src.labeling import concordance_label
from src.patches import split_by_axis, hemisphere_midpoint
from src.bold_features import parse_events_tsv
from src.model import SimplePatchCNN
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold, train_one_fold_regression
from src import viz
from scripts.download_real_labels import download_subject_labels, DATA_DIR
from scripts.download_bold import download_subject_bold
from scripts.download_events import download_subject_events

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pooled_extended")
CONTRASTS = ["calc", "mem"]
PATCH_SIZES = [9, 15]
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 400
SEED = 0


def load_subject_core(sub):
    d = os.path.join(DATA_DIR, sub, "derivatives")
    t1, affine, _ = load_nifti(os.path.join(d, f"{sub}_space-T2_desc-brain_T1w.nii.gz"))
    mask_raw, _, _ = load_nifti(os.path.join(d, f"{sub}_BrMsk_CSF_30slices.nii.gz"))
    mask = (mask_raw > 0.5).astype(np.uint8)

    cmro2 = {}
    for cond in ["control", "calc", "mem"]:
        v, _, _ = load_nifti(os.path.join(d, f"{sub}_task-{cond}_space-T2_desc-orig_cmro2.nii"))
        cmro2[cond] = v.squeeze()

    bold_pct = {}
    for c in ["calc", "mem"]:
        v, _, _ = load_nifti(os.path.join(d, f"{sub}_task-{c}control_space-T2_BOLD_percchange.nii.gz"))
        bold_pct[c] = v

    return t1, affine, mask, cmro2, bold_pct


def build_targets(cmro2, bold_pct, mask, contrast):
    """Возвращает (binary_label, continuous_pct) - та же маска валидности,
    что и в экспериментах с классификацией, поэтому обе целевые
    переменные вычисляются на одном и том же множестве вокселей для
    честного сравнения."""
    cmro2_task = cmro2[contrast]
    cmro2_control = cmro2["control"]
    bold = bold_pct[contrast]

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0

    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    binary_label = np.zeros(cmro2_task.shape, dtype=np.float32)
    conc = concordance_label(bold, pct)
    binary_label[valid] = conc[valid]

    continuous = np.zeros(cmro2_task.shape, dtype=np.float32)
    continuous[valid] = pct[valid]
    return binary_label, continuous, valid


def process_subject(sub):
    """Возвращает {contrast: {"patches": {9: arr, 15: arr}, "binary_label":...,
    "continuous_target":..., "side":...}} или None, если пациент пропущен."""
    try:
        download_subject_labels(sub)
        download_subject_events(sub)
    except Exception as e:
        print(f"  [пропуск] {sub}: отсутствуют базовые файлы ({e})")
        return None

    try:
        t1, affine, mask, cmro2, bold_pct = load_subject_core(sub)
    except Exception as e:
        print(f"  [пропуск] {sub}: не удалось загрузить базовые файлы ({e})")
        return None

    targets_by_contrast = {}
    for contrast in CONTRASTS:
        binary_label, continuous, valid = build_targets(cmro2, bold_pct, mask, contrast)
        n_conc, n_disc = int((binary_label > 0).sum()), int((binary_label < 0).sum())
        if min(n_conc, n_disc) < 20:
            print(f"  [пропуск контраста] {sub} {contrast}: вырожденная метка")
            continue
        targets_by_contrast[contrast] = (binary_label, continuous)

    if not targets_by_contrast:
        return None

    # events.tsv строго не нужен для этих двух экспериментов (на этот раз
    # BOLD не используется как входной признак), но мы всё равно требуем
    # его, чтобы множество вокселей/пациентов оставалось идентичным
    # предыдущему эксперименту с условием BOLD для сопоставимости, и
    # чтобы пропускать пациентов согласованно с тем запуском.
    try:
        download_subject_bold(sub)  # скачивается, но сразу не используется; обеспечивает сопоставимое множество пациентов
    except Exception as e:
        print(f"  [пропуск] {sub}: отсутствует BOLD ({e})")
        return None
    bold_path = os.path.join(DATA_DIR, sub, "derivatives", f"{sub}_task-all_space-T2_filtered_func.nii.gz")

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    max_patch = max(PATCH_SIZES)
    margin = max_patch // 2  # используем отступ для самого большого патча, чтобы ОБА размера патчей оставались защищёнными от утечки
    rng = np.random.default_rng(hash(sub) % (2**31))

    out = {}
    for contrast, (binary_label, continuous) in targets_by_contrast.items():
        coords, labels = select_labeled_coords(binary_label, mask, max_voxels=None, seed=SEED)
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

        patches_by_size = {
            p: extract_and_normalize_patches(t1, used_coords, p, mask) for p in PATCH_SIZES
        }
        continuous_target = continuous[used_coords[:, 0], used_coords[:, 1], used_coords[:, 2]]

        out[contrast] = {
            "patches": patches_by_size,
            "binary_label": labels[used_idx],
            "continuous_target": continuous_target,
            "side": side_tags,
        }

    if os.path.exists(bold_path):
        os.remove(bold_path)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {
        c: {
            "patches": {p: [] for p in PATCH_SIZES},
            "binary_label": [], "continuous_target": [], "side": [], "subject": [],
        }
        for c in CONTRASTS
    }

    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub)
        except Exception:
            print(f"  [ошибка] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append((sub, "skipped"))
            continue
        log.append((sub, sorted(result.keys())))
        for contrast, d in result.items():
            for p in PATCH_SIZES:
                pooled[contrast]["patches"][p].append(d["patches"][p])
            pooled[contrast]["binary_label"].append(d["binary_label"])
            pooled[contrast]["continuous_target"].append(d["continuous_target"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["binary_label"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["binary_label"]:
            print(f"{contrast}: нет подходящих пациентов, пропуск")
            continue
        patches = {p: np.concatenate(pooled[contrast]["patches"][p], axis=0) for p in PATCH_SIZES}
        binary_label = np.concatenate(pooled[contrast]["binary_label"], axis=0)
        continuous_target = np.concatenate(pooled[contrast]["continuous_target"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a = np.where(side == "A")[0]
        pos_b = np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)}/{len(pos_b)} вокселей A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            for patch_size in PATCH_SIZES:
                p = patches[patch_size]

                _, hist_cls, pred_cls, probs_cls = train_one_fold(
                    p[train_idx], binary_label[train_idx], p[test_idx], binary_label[test_idx],
                    epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
                )
                acc = hist_cls["val_acc"][-1]

                _, hist_reg, pred_reg, ytrue_reg = train_one_fold_regression(
                    p[train_idx], continuous_target[train_idx], p[test_idx], continuous_target[test_idx],
                    epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
                )
                r2 = hist_reg["test_r2"][-1]
                corr = float(np.corrcoef(ytrue_reg, pred_reg)[0, 1]) if np.std(pred_reg) > 1e-8 else 0.0

                print(f"  {fold_name} patch={patch_size}: точность классификации acc={acc:.3f}  регрессия R2={r2:.3f} r={corr:.3f}")
                key = (contrast, fold_name, patch_size)
                all_results[key] = {"classification_acc": acc, "regression_r2": r2, "regression_corr": corr}

                viz.plot_regression_scatter(
                    ytrue_reg, pred_reg, r2, f"pooled {contrast} {fold_name} patch={patch_size}",
                    os.path.join(OUT_DIR, f"{contrast}_{fold_name}_patch{patch_size}_regression.png"),
                )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}|{p}": r for (c, f, p), r in all_results.items()}, f, indent=1)

    # сводные графики
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels_x = sorted(set((c, f) for (c, f, p) in all_results.keys()))
    x = np.arange(len(labels_x))
    width = 0.35
    for i, p in enumerate(PATCH_SIZES):
        accs = [all_results.get((c, f, p), {}).get("classification_acc", np.nan) for (c, f) in labels_x]
        axes[0].bar(x + (i - 0.5) * width, accs, width, label=f"patch={p}")
    axes[0].axhline(0.5, color="gray", linestyle=":", label="случайность")
    axes[0].axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{c}\n{f}" for c, f in labels_x], fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Точность классификации по размеру патча")
    axes[0].legend(fontsize=7)

    for i, p in enumerate(PATCH_SIZES):
        r2s = [all_results.get((c, f, p), {}).get("regression_r2", np.nan) for (c, f) in labels_x]
        axes[1].bar(x + (i - 0.5) * width, r2s, width, label=f"patch={p}")
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{c}\n{f}" for c, f in labels_x], fontsize=8)
    axes[1].set_title("R2 регрессии по размеру патча")
    axes[1].legend(fontsize=7)

    fig.suptitle(f"Объединённая выборка ({sum(1 for s,r in log if r != 'skipped')} пациентов): больший патч + регрессия")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "pooled_extended_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
