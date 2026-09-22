"""
Меняет единицу классификации с вокселя на настоящую анатомическую
область (парцел Glasser/HCP-MMP1.0), напрямую мотивировано результатом
Buchel et al. (2026) о том, что большая часть нулевого результата этого
проекта может быть шумом меток отдельных вокселей, а не отсутствием
структурного сигнала (см. README «Независимый литературный контекст» /
раздел отчёта 06j-06k). Усреднение внутри анатомически настоящей области
по десяткам-сотням вокселей механически подавляет именно этот тип шума
(та же статистическая логика, по которой групповые/ROI-анализы в
нейровизуализации надёжнее анализов на уровне отдельного вокселя) - это
прямое и недорогое продолжение этого результата, а не новая гипотеза.

Для каждого пациента, каждого контраста, каждого настоящего парцела
Glasser (используется тот же зарегистрированный атлас, что и в
scripts/run_glasser.py, кэшированный на диск): сырые CMRO2_task и
CMRO2_control усредняются по всем валидным вокселям этого парцела перед
вычислением одного процентного изменения на уровне ROI (стандартный
способ вычисления контрастов на уровне ROI - сначала усредняется
физиологическая величина, затем берётся одно отношение, а не
усредняются много зашумлённых повоксельных отношений), и отдельно
усредняется BOLD_percchange (сырой базовый уровень BOLD недоступен на
этой стадии обработки датасета, поэтому усредняется его повоксельное
процентное изменение). Метка ROI = sign(среднее BOLD_percchange) x
sign(ROI CMRO2_percchange), то же определение, что и везде в этом
проекте, только вычисленное по среднему области, а не по одному
вокселю. Парцелы с числом валидных вокселей (на своей стороне разбиения
по полушариям) меньше MIN_VOXELS_PER_ROI отбрасываются как слишком
маленькие/зашумлённые для доверия среднему.

Классифицирует каждую область по её собственному структурному профилю
(среднее T1, стандартное отклонение T1, логарифм размера - те же 3
признака, что использовались в run_glasser.py) через RegionMLP - маленький
MLP, а не 3D CNN, поскольку входными данными здесь служит сводка по
области, а не пространственный патч. Тот же защищённый от утечки
протокол разбиения по полушариям (обучение на областях стороны A, тест
на стороне B, и наоборот).
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
from src.patches import hemisphere_midpoint
from src.model import RegionMLP
from src.train import train_one_fold
from src import viz
from src.subject_loader import load_subject_robust
from src.glasser_atlas import get_subject_glasser_atlas, ATLAS_CACHE_DIR

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "roi_averaged")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE_MARGIN = 9  # используется только для отступа разбиения по полушариям, патчи здесь не извлекаются
MIN_VOXELS_PER_ROI = 20
SEED = 0


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
    try:
        atlas = get_subject_glasser_atlas(sub, t1_cache_path)
    except Exception as e:
        print(f"  [пропуск] {sub}: регистрация атласа не удалась ({e})")
        return None
    if atlas.shape != t1.shape:
        print(f"  [пропуск] {sub}: несовпадение формы атласа")
        return None

    midpoint = hemisphere_midpoint(t1.shape, axis_index=0)
    margin = PATCH_SIZE_MARGIN // 2
    x_idx = np.arange(t1.shape[0])
    side_a_x, side_b_x = x_idx < (midpoint - margin), x_idx > (midpoint + margin)
    m = mask.astype(bool)
    global_mean, global_std = t1[t1 != 0].mean(), t1[t1 != 0].std() + 1e-6

    out = {}
    for contrast in CONTRASTS:
        if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
            continue
        cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
        valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & m

        feats, labels, sides = [], [], []
        for side_name, side_x in [("A", side_a_x), ("B", side_b_x)]:
            side_mask = np.zeros_like(m)
            side_mask[side_x, :, :] = True
            region_valid = valid & side_mask
            coords = np.argwhere(region_valid)
            if len(coords) == 0:
                continue
            pids = atlas[coords[:, 0], coords[:, 1], coords[:, 2]]
            for pid in np.unique(pids):
                if pid == 0:
                    continue
                sel = pids == pid
                if sel.sum() < MIN_VOXELS_PER_ROI:
                    continue
                vx = coords[sel]
                roi_bold = bold[vx[:, 0], vx[:, 1], vx[:, 2]].mean()
                roi_cmro2_task = cmro2_task[vx[:, 0], vx[:, 1], vx[:, 2]].mean()
                roi_cmro2_control = cmro2_control[vx[:, 0], vx[:, 1], vx[:, 2]].mean()
                if roi_cmro2_control == 0:
                    continue
                roi_pct = (roi_cmro2_task - roi_cmro2_control) / roi_cmro2_control * 100
                if not np.isfinite(roi_pct) or roi_bold == 0:
                    continue
                roi_label = 1.0 if concordance_label(np.array([roi_bold]), np.array([roi_pct]))[0] > 0 else 0.0

                t1_vals = t1[vx[:, 0], vx[:, 1], vx[:, 2]]
                feat = [
                    (t1_vals.mean() - global_mean) / global_std,
                    t1_vals.std() / global_std,
                    float(np.log1p(len(vx))),
                ]
                feats.append(feat)
                labels.append(roi_label)
                sides.append(side_name)

        if len(feats) == 0:
            continue
        n_conc, n_disc = sum(1 for l in labels if l > 0), sum(1 for l in labels if l == 0)
        if min(n_conc, n_disc) < 5:
            continue
        out[contrast] = {
            "features": np.array(feats, dtype=np.float32),
            "labels": np.array(labels, dtype=np.float32),
            "side": np.array(sides),
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"features": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}
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
        n_rois = {c: len(d["labels"]) for c, d in result.items()}
        log.append({"subject": sub, "status": "used", "n_rois": n_rois})
        for contrast, d in result.items():
            pooled[contrast]["features"].append(d["features"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\n{n_used}/{len(ALL_SUBJECTS)} пациентов внесли >=1 пригодный ROI")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["features"]:
            continue
        feats = np.concatenate(pooled[contrast]["features"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        n_conc, n_disc = int((labels > 0).sum()), int((labels == 0).sum())
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)}/{len(pos_b)} ROI A/B, {n_conc} concordant / {n_disc} discordant ===")

        for fold_name, (train_idx, test_idx) in {"A_train_B_test": (pos_a, pos_b), "B_train_A_test": (pos_b, pos_a)}.items():
            _, hist, pred, probs = train_one_fold(
                feats[train_idx], labels[train_idx], feats[test_idx], labels[test_idx],
                epochs=30, batch_size=16, device="cpu", seed=SEED,
                model_factory=lambda: RegionMLP(in_dim=3),
            )
            acc = hist["val_acc"][-1]
            print(f"  {fold_name}: n_train={len(train_idx)} n_test={len(test_idx)} acc={acc:.3f}")
            all_results[(contrast, fold_name)] = {"acc": acc, "n_train": len(train_idx), "n_test": len(test_idx), "n_subjects": n_subjects}
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"ROI-averaged {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    keys = list(all_results.keys())
    vals = [all_results[k]["acc"] for k in keys]
    ax.bar([f"{c}\n{f}" for c, f in keys], vals, color="#16a085")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайность")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон")
    ax.set_ylim(0, 1)
    ax.set_title(f"Классификация с усреднением по ROI (реальные парцелы Glasser, >={MIN_VOXELS_PER_ROI} вокселей)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "roi_averaged_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
