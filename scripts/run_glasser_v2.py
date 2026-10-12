"""
Повтор эксперимента с настоящим атласом Glasser (run_glasser.py), но с
улучшенной регистрацией атласа (src/glasser_atlas_v2.py): вместо одного
плохо обусловленного шага MNI -> обрезанная до 30 срезов T2-сетка,
используется цепочка MNI -> полный непорезанный T1 пациента (найден в
derivatives/anat/, никогда раньше не использовался) -> T2-сетка. Прямая
проверка на реальных данных: насколько устойчивы (или нет) результаты
voxel-level Glasser-эксперимента к качеству регистрации атласа - честно
раскрытая неточность границ в run_glasser.py/glasser_atlas.py может быть
реальной причиной шума в назначении парцеллов, а не просто оговоркой
"для порядка".

Количественная проверка перед прогоном классификации (см.
scripts/compare_atlas_registrations.py): на sub-p019 старая и новая
регистрация дают почти одинаковый общий Dice с маской мозга (~0.64-0.65),
но там, где обе согласны, что воксель - кора, конкретный парцелл совпадает
только в ~50% случаев - то есть сама по себе неточность границ велика
настолько, что имеет смысл проверить её влияние на результат
классификации напрямую, а не просто упомянуть как ограничение.

Во всём остальном идентично run_glasser.py: та же архитектура
(PatchBOLDConditionNet), тот же протокол защиты от утечки, тот же
3-признаковый дескриптор региона (средний T1, std T1, log-размер).
Базовый уровень «большинство» считается и указывается с самого начала.
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
from src.model import PatchBOLDConditionNet
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold_multimodal
from src import viz
from src.subject_loader import load_subject_robust
from src.glasser_atlas_v2 import get_subject_glasser_atlas_v2, ATLAS_CACHE_DIR
from scripts.download_native_t1 import download_subject_native_t1

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "glasser_v2")
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


def region_stats_for_side(t1, mask, atlas, side_mask_x):
    xs = np.where(side_mask_x)[0]
    m = mask.astype(bool)
    coords = np.argwhere(m)
    coords = coords[np.isin(coords[:, 0], xs)]
    if len(coords) == 0:
        return {}
    vals = t1[coords[:, 0], coords[:, 1], coords[:, 2]]
    ids = atlas[coords[:, 0], coords[:, 1], coords[:, 2]]
    stats = {}
    for pid in np.unique(ids):
        sel = ids == pid
        cvals = vals[sel]
        stats[int(pid)] = (float(cvals.mean()), float(cvals.std() + 1e-6), float(np.log1p(sel.sum())))
    return stats


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

    t1_cache_path = os.path.join(ATLAS_CACHE_DIR, f"{sub}_t1_for_reg.nii.gz")
    if not os.path.exists(t1_cache_path):
        import nibabel as nib
        os.makedirs(ATLAS_CACHE_DIR, exist_ok=True)
        nib.save(nib.Nifti1Image(t1.astype(np.float32), affine), t1_cache_path)

    native_t1_path = download_subject_native_t1(sub)
    if native_t1_path is None:
        print(f"  [пропуск] {sub}: нет полного T1 (fMRIPrep anat/)")
        return None

    try:
        atlas = get_subject_glasser_atlas_v2(sub, t1_cache_path, native_t1_path)
    except Exception as e:
        print(f"  [пропуск] {sub}: регистрация атласа (v2) не удалась ({e})")
        return None
    if atlas.shape != t1.shape:
        print(f"  [пропуск] {sub}: форма атласа {atlas.shape} != форма t1 {t1.shape}")
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE // 2
    x_idx = np.arange(t1.shape[0])
    side_a_x, side_b_x = x_idx < (midpoint - margin), x_idx > (midpoint + margin)

    stats_a = region_stats_for_side(t1, mask, atlas, side_a_x)
    stats_b = region_stats_for_side(t1, mask, atlas, side_b_x)
    global_mean, global_std = t1[t1 != 0].mean(), t1[t1 != 0].std() + 1e-6
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
        used_labels = labels[used_idx]

        patches = extract_and_normalize_patches(t1, used_coords, PATCH_SIZE, mask)

        region_feat = np.zeros((len(used_idx), 3), dtype=np.float32)
        for i, (x, y, z) in enumerate(used_coords):
            pid = int(atlas[x, y, z])
            stats = stats_a if side_tags[i] == "A" else stats_b
            mean, std, logsize = stats.get(pid, (0.0, 1.0, 0.0))
            region_feat[i] = [(mean - global_mean) / global_std, std / global_std, logsize]

        out[contrast] = {"patches": patches, "labels": used_labels, "side": side_tags, "region_feat": region_feat}

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches": [], "labels": [], "side": [], "region_feat": []} for c in CONTRASTS}
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
            for key in ["patches", "labels", "side", "region_feat"]:
                pooled[contrast][key].append(d[key])

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for _, v in log if v != "skipped")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} пациентов использовано")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches"]:
            continue
        patches = np.concatenate(pooled[contrast]["patches"], axis=0)
        feats = np.concatenate(pooled[contrast]["region_feat"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {len(pos_a)}/{len(pos_b)} вокселей A/B ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            test_labels = labels[test_idx]
            majority_baseline = float(max(test_labels.mean(), 1 - test_labels.mean()))

            _, hist, pred, probs = train_one_fold_multimodal(
                patches[train_idx], feats[train_idx], labels[train_idx],
                patches[test_idx], feats[test_idx], test_labels,
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: PatchBOLDConditionNet(n_bold_features=3),
            )
            acc = hist["val_acc"][-1]
            beats = "ДА" if acc > majority_baseline else "нет"
            print(f"  {fold_name}: acc={acc:.3f} базовый_уровень={majority_baseline:.3f} превышает={beats}")
            all_results[(contrast, fold_name)] = {"acc": acc, "majority_baseline": majority_baseline, "n_train": len(train_idx), "n_test": len(test_idx)}
            viz.plot_confusion_and_roc(
                test_labels, pred, probs, f"Glasser v2 {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1, default=float)

    keys = list(all_results.keys())
    vals = [all_results[k]["acc"] for k in keys]
    base = [all_results[k]["majority_baseline"] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    width = 0.35
    ax.bar(x - width/2, vals, width, label="модель (атлас v2)", color="#8e44ad")
    ax.bar(x + width/2, base, width, label="базовый уровень «большинство»", color="#8e44ad", alpha=0.4, hatch="//")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайность")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{f}" for c, f in keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Glasser v2 (через полный T1): точность против базового уровня")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "glasser_v2_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
