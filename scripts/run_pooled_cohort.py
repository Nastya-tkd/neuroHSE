"""
Увеличенная версия Экспериментов 1 и 2: вместо обучения одной CNN на пациента
(n=5, ~1500 вокселей на сторону каждый - слишком большая ёмкость для очень
малого объёма данных) воксели объединяются по всей валидной выборке (40
пациентов, src/cohort.py) и обучается ОДНА модель на "полушарии A по всем
пациентам" против "полушария B по всем пациентам". Это естественный способ
реально использовать "больше пациентов" для требовательной к данным CNN,
а не просто повторять один и тот же прогон на малом N для одного пациента
40 раз.

Также вход по BOLD переключается с сырого временного ряда из 400 точек
времени, использовавшегося в Эксперименте 2, на компактный вектор
compute_condition_features - процентное изменение сигнала по каждому
условию (calc/mem/rest) - те самые "признаки BOLD по условиям", которые
запрашивал пользователь.

Обрабатывает пациентов по одному: скачивает небольшие файлы производных
данного пациента + BOLD-объём filtered_func размером ~300-450 МБ,
извлекает патчи и признаки condition-BOLD для его субсэмплированных
вокселей, затем УДАЛЯЕТ большой BOLD-файл перед переходом к следующему
пациенту (диск ограничен ~1 большим файлом за раз, а не 40 x 400 МБ
одновременно). Пациенты, у которых отсутствует какой-либо требуемый файл
в истории версий S3, пропускаются с логируемой причиной, а не молча
отбрасываются.

Запускает по две объединённые модели на каждый контраст и каждое
направление разбиения: SimplePatchCNN (только структура, для прямого
сопоставимого сравнения при этом большем N) и PatchBOLDConditionNet
(структура + признаки BOLD по условию).
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
from src.bold_features import compute_condition_features, parse_events_tsv
from src.model import SimplePatchCNN, PatchBOLDConditionNet
from src.train import (
    select_labeled_coords,
    extract_and_normalize_patches,
    train_one_fold,
    train_one_fold_multimodal,
)
from src import viz
from scripts.download_real_labels import download_subject_labels, DATA_DIR
from scripts.download_bold import download_subject_bold
from scripts.download_events import download_subject_events

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pooled_cohort")
CONTRASTS = ["calc", "mem"]
CONDITIONS = ("calc", "mem", "rest")
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
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


def build_label(cmro2, bold_pct, mask, contrast):
    cmro2_task = cmro2[contrast]
    cmro2_control = cmro2["control"]
    bold = bold_pct[contrast]

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0

    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    label = np.zeros(cmro2_task.shape, dtype=np.float32)
    conc = concordance_label(bold, pct)
    label[valid] = conc[valid]
    return label


def process_subject(sub):
    """Возвращает {contrast: {"patches":..., "bold_feat":..., "labels":..., "side": массив 'A'/'B'}}
    либо None, если пациента нужно пропустить."""
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

    labels_by_contrast = {}
    for contrast in CONTRASTS:
        label = build_label(cmro2, bold_pct, mask, contrast)
        n_conc, n_disc = int((label > 0).sum()), int((label < 0).sum())
        if min(n_conc, n_disc) < 20:
            print(f"  [пропуск контраста] {sub} {contrast}: вырожденная метка (conc={n_conc}, disc={n_disc})")
            continue
        labels_by_contrast[contrast] = label

    if not labels_by_contrast:
        return None

    try:
        bold_path = download_subject_bold(sub)
        events_path = os.path.join(DATA_DIR, sub, "raw", f"{sub}_task-all_events.tsv")
        bold_4d, _, bold_img = load_nifti(bold_path)
        tr = float(bold_img.header.get_zooms()[3])
        events = parse_events_tsv(events_path)
    except Exception as e:
        print(f"  [пропуск] {sub}: отсутствуют BOLD/события ({e})")
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    rng = np.random.default_rng(hash(sub) % (2**31))

    out = {}
    for contrast, label in labels_by_contrast.items():
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
        bold_feat = compute_condition_features(bold_4d, used_coords, events, tr, conditions=CONDITIONS)

        out[contrast] = {
            "patches": patches,
            "bold_feat": bold_feat,
            "labels": labels[used_idx],
            "side": side_tags,
        }

    del bold_4d
    os.remove(bold_path)  # освободить место на диске перед следующим пациентом
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "bold_feat": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["patches"].append(d["patches"])
            pooled[contrast]["bold_feat"].append(d["bold_feat"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            print(f"{contrast}: нет пригодных пациентов, пропуск")
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        bold_feat = np.concatenate(pooled[contrast]["bold_feat"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a = np.where(side == "A")[0]
        pos_b = np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)} вокселей стороны A, {len(pos_b)} вокселей стороны B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            # базовый уровень только по структуре при этом большем N
            _, hist_struct, pred_struct, probs_struct = train_one_fold(
                patches[train_idx], labels[train_idx], patches[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
            )
            acc_struct = hist_struct["val_acc"][-1]

            # структура + признаки BOLD по условию
            _, hist_combo, pred_combo, probs_combo = train_one_fold_multimodal(
                patches[train_idx], bold_feat[train_idx], labels[train_idx],
                patches[test_idx], bold_feat[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: PatchBOLDConditionNet(n_bold_features=len(CONDITIONS)),
            )
            acc_combo = hist_combo["val_acc"][-1]

            print(f"  {fold_name}: только структура={acc_struct:.3f}  структура+BOLD-условие={acc_combo:.3f}")
            all_results[(contrast, fold_name)] = {"structural_only": acc_struct, "structural_plus_bold": acc_combo}

            viz.plot_confusion_and_roc(
                labels[test_idx], pred_combo, probs_combo, f"pooled {contrast} {fold_name}: structural+BOLD",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_structural_plus_bold_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    # сводный график
    labels_x = [f"{c}\n{f}" for (c, f) in all_results.keys()]
    struct_vals = [all_results[k]["structural_only"] for k in all_results]
    combo_vals = [all_results[k]["structural_plus_bold"] for k in all_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels_x))
    width = 0.35
    ax.bar(x - width / 2, struct_vals, width, label="только структура", color="#7f8c8d")
    ax.bar(x + width / 2, combo_vals, width, label="структура + BOLD (по условию)", color="#2980b9")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайный уровень")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон научного руководителя")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title(f"Объединённая выборка ({len(ALL_SUBJECTS)} пациентов): структура vs структура+BOLD")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "pooled_cohort_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено {out_path}")


if __name__ == "__main__":
    main()
