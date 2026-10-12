"""
Толщина коры как дополнительный входной канал наряду с T1 - прямая
проверка "surface-adjacent" гипотезы без самого FreeSurfer, напрямую
отвечающая на вопрос "а что если дело не в интенсивности T1, а в
геометрии/толщине ткани". Методология и честные приближения (сегментация
k-means, передискретизация до 1.5мм, 25 итераций DiReCT вместо 45) - см.
src/cortical_thickness.py.

Та же логика отсутствия цикличности и тот же leakage-safe протокол, что и
в run_cbf_oef.py/run_cbv.py: толщина - структурная, не физиологическая
величина, но она тоже не часть определения метки concordant/discordant
(sign(BOLD_percchange) x sign(CMRO2_percchange)) - независимый
геометрический признак ткани.

Запускает две модели на контраст на направление разбиения, при полностью
идентичных вокселях/патчах:
  - только структурные данные (T1, 1 канал).
  - структура + толщина коры (T1+thickness, 2 канала, совмещённые и
    независимо нормализованные по каждому каналу).

Базовый уровень «большинство» вычисляется и сообщается наряду с точностью
обеих моделей с самого начала, та же дисциплина, что и везде после
поправки по ROI-усреднению.
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
from src.model import SimplePatchCNN
from src.train import (
    select_labeled_coords,
    extract_and_normalize_patches,
    extract_and_normalize_multichannel_patches,
    train_one_fold,
)
from src import viz
from src.subject_loader import load_subject_robust
from src.cortical_thickness import get_subject_thickness_map
from src.glasser_atlas_v2 import ATLAS_CACHE_DIR
from scripts.download_native_t1 import download_subject_native_t1
from scripts.download_real_labels import download_versioned, DATA_DIR, VERSIONS_CACHE_DIR
from scripts.list_versions import get_or_build_version_map

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cortical_thickness")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 9
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 500
SEED = 0


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


def download_subject_native_t1_brain(subject):
    """desc-fmriprep_T1w_brain.nii.gz (уже скальп-стрипнутый) - нужен для
    сегментации тканей; desc-fmriprep_T1w.nii.gz (не стрипнутый) используется
    отдельно в glasser_atlas_v2.py для регистрации атласа."""
    suffix = "_desc-fmriprep_T1w_brain.nii.gz"
    cache_path = os.path.join(VERSIONS_CACHE_DIR, f"{subject}.json")
    version_map = get_or_build_version_map(f"ds004873/derivatives/{subject}/", cache_path)
    out_dir = os.path.join(DATA_DIR, subject, "derivatives")
    os.makedirs(out_dir, exist_ok=True)
    matches = [k for k in version_map if k.split("/")[-1] == subject + suffix]
    if not matches:
        return None
    key = matches[0]
    version_id, _ = version_map[key]
    out_path = os.path.join(out_dir, key.split("/")[-1])
    download_versioned(key, version_id, out_path)
    return out_path


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

    t2_t1_cache_path = os.path.join(ATLAS_CACHE_DIR, f"{sub}_t1_for_reg.nii.gz")
    if not os.path.exists(t2_t1_cache_path):
        import nibabel as nib
        os.makedirs(ATLAS_CACHE_DIR, exist_ok=True)
        nib.save(nib.Nifti1Image(t1.astype(np.float32), affine), t2_t1_cache_path)

    native_t1_brain_path = download_subject_native_t1_brain(sub)
    if native_t1_brain_path is None:
        print(f"  [пропуск] {sub}: нет полного T1 brain (fMRIPrep anat/)")
        return None

    try:
        thickness = get_subject_thickness_map(sub, t2_t1_cache_path, native_t1_brain_path)
    except Exception as e:
        print(f"  [пропуск] {sub}: расчёт толщины не удался ({e})")
        return None
    if thickness.shape != t1.shape:
        print(f"  [пропуск] {sub}: форма толщины {thickness.shape} != форма t1 {t1.shape}")
        return None

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

        patches_1ch = extract_and_normalize_patches(t1, used_coords, PATCH_SIZE, mask)
        patches_2ch = extract_and_normalize_multichannel_patches(
            [t1, thickness], used_coords, PATCH_SIZE, mask
        )

        out[contrast] = {
            "patches_1ch": patches_1ch,
            "patches_2ch": patches_2ch,
            "labels": labels[used_idx],
            "side": side_tags,
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches_1ch": [], "patches_2ch": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["patches_1ch"].append(d["patches_1ch"])
            pooled[contrast]["patches_2ch"].append(d["patches_2ch"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} пациентов использовано (есть толщина коры + >=1 контраст)")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches_1ch"]:
            print(f"{contrast}: нет подходящих пациентов, пропуск")
            continue
        patches_1ch = np.concatenate(pooled[contrast]["patches_1ch"], axis=0)
        patches_2ch = np.concatenate(pooled[contrast]["patches_2ch"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)} вокселей сторона A, {len(pos_b)} сторона B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            test_labels = labels[test_idx]
            majority_baseline = float(max(test_labels.mean(), 1 - test_labels.mean()))

            _, hist_struct, pred_struct, probs_struct = train_one_fold(
                patches_1ch[train_idx], labels[train_idx], patches_1ch[test_idx], test_labels,
                epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
            )
            acc_struct = hist_struct["val_acc"][-1]

            _, hist_thick, pred_thick, probs_thick = train_one_fold(
                patches_2ch[train_idx], labels[train_idx], patches_2ch[test_idx], test_labels,
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: SimplePatchCNN(in_channels=2),
            )
            acc_thick = hist_thick["val_acc"][-1]

            beats = "ДА" if acc_thick > majority_baseline else "нет"
            print(f"  {fold_name}: только-T1={acc_struct:.3f}  T1+толщина={acc_thick:.3f}  "
                  f"базовый_уровень={majority_baseline:.3f}  превышает={beats}")
            all_results[(contrast, fold_name)] = {
                "structural_only": acc_struct, "structural_plus_thickness": acc_thick,
                "majority_baseline": majority_baseline, "n_subjects": n_subjects,
                "n_train": len(train_idx), "n_test": len(test_idx),
            }
            viz.plot_confusion_and_roc(
                test_labels, pred_thick, probs_thick, f"Толщина коры {contrast} {fold_name}: T1+толщина",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_thickness_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    labels_x = [f"{c}\n{f}" for (c, f) in all_results.keys()]
    struct_vals = [all_results[k]["structural_only"] for k in all_results]
    thick_vals = [all_results[k]["structural_plus_thickness"] for k in all_results]
    base_vals = [all_results[k]["majority_baseline"] for k in all_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels_x))
    width = 0.25
    ax.bar(x - width, struct_vals, width, label="только T1", color="#7f8c8d")
    ax.bar(x, thick_vals, width, label="T1 + толщина коры", color="#c0392b")
    ax.bar(x + width, base_vals, width, label="базовый уровень «большинство»", color="#c0392b", alpha=0.35, hatch="//")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайность")
    ax.set_xticks(x); ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("Только T1 против T1+толщина коры (surface-adjacent гипотеза)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "thickness_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
