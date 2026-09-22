"""
Единственный по-настоящему новый, ещё не заблокированный рычаг: базовые
(в покое, control-условие) карты CBF и OEF в качестве дополнительных
входных каналов вместе с T1, вместо очередного способа смотреть на
анатомическую интенсивность.

Мотивация (см. обсуждение в README/отчёте): интенсивность T1 отражает
состав ткани (содержание миелина/воды/макромолекул) - она напрямую не
кодирует физиологические величины, которые механически определяют,
движутся ли BOLD и CMRO2 в одну сторону или в противоположные (базовая
перфузия, базовая экстракция кислорода, коэффициент нейрососудистого
сопряжения). Базовые карты CBF и OEF гораздо ближе напрямую к этим
величинам, и - в отличие от настоящего анатомического атласа - они уже
есть в собственной истории версий S3 этого датасета, а не на заблокированном
внешнем хосте.

Нециркулярность: используются только CBF/OEF *control*-условия (состояние
покоя), значения условия задачи никогда не используются. Метка
concordant/discordant определяется *изменением* между задачей и control
(sign(BOLD_percchange) x sign(CMRO2_percchange)); значения только базового
уровня не участвуют в этом вычислении, поэтому это настоящий эксперимент с
физиологической ковариатой, а не утечка метки. (Базовый CMRO2_control =
CBF_control x OEF_control x CaO2 по принципу Фика - то же самое отношение,
которое исходный конвейер использовал для изначального вычисления базового
CMRO2, - поэтому базовые значения CBF/OEF физиологически предшествуют
метке, а не являются её пересказом.)

Запускает две модели на каждый контраст и каждое направление разбиения,
на абсолютно тех же вокселях/патчах, для честного сравнения:
  - только структура (T1, 1 канал) - тот же базовый уровень, что и в
    каждом более раннем эксперименте, пересчитанный здесь заново для
    честного сравнения, а не повторное использование старых чисел (другая,
    ограниченная наличием CBF/OEF подвыборка пациентов).
  - структура + базовые CBF/OEF (T1+CBF+OEF, 3 канала, совмещённые и
    независимо нормализованные по каждому каналу).
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
from src.model import SimplePatchCNN
from src.train import (
    select_labeled_coords,
    extract_and_normalize_patches,
    extract_and_normalize_multichannel_patches,
    train_one_fold,
)
from src import viz
from src.subject_loader import load_subject_robust
from scripts.download_cbf_oef import download_subject_cbf_oef

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cbf_oef")
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


def load_cbf_oef(sub, mask, expected_shape):
    """Базовые CBF/OEF, замаскированные по мозгу и очищенные от нефинитного
    шума вне маски (эти исходные qmri-карты не были заранее лишены черепа)."""
    files = download_subject_cbf_oef(sub)
    if len(files) < 2:
        return None
    d = os.path.join("data", sub, "derivatives")
    cbf, _, _ = load_nifti(os.path.join(d, f"{sub}_task-control_space-T2_cbf.nii"))
    oef, _, _ = load_nifti(os.path.join(d, f"{sub}_task-control_space-T2_oef.nii"))
    cbf, oef = cbf.squeeze(), oef.squeeze()
    if cbf.shape != expected_shape or oef.shape != expected_shape:
        return None
    m = mask.astype(bool)
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0) * m
    oef = np.nan_to_num(oef, nan=0.0, posinf=0.0, neginf=0.0) * m
    return cbf, oef


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

    cbf_oef = load_cbf_oef(sub, mask, t1.shape)
    if cbf_oef is None:
        print(f"  [пропуск] {sub}: нет пригодных базовых CBF/OEF")
        return None
    cbf, oef = cbf_oef

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
        patches_3ch = extract_and_normalize_multichannel_patches(
            [t1, cbf, oef], used_coords, PATCH_SIZE, mask
        )

        out[contrast] = {
            "patches_1ch": patches_1ch,
            "patches_3ch": patches_3ch,
            "labels": labels[used_idx],
            "side": side_tags,
        }

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled = {c: {"patches_1ch": [], "patches_3ch": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}

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
            pooled[contrast]["patches_3ch"].append(d["patches_3ch"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    n_used = sum(1 for e in log if e["status"] == "used")
    print(f"\nИспользовано пациентов: {n_used}/{len(ALL_SUBJECTS)} (есть базовые CBF/OEF + >=1 контраст)")

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["patches_1ch"]:
            print(f"{contrast}: нет пригодных пациентов, пропуск")
            continue
        patches_1ch = np.concatenate(pooled[contrast]["patches_1ch"], axis=0)
        patches_3ch = np.concatenate(pooled[contrast]["patches_3ch"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)} вокселей стороны A, {len(pos_b)} вокселей стороны B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            _, hist_struct, pred_struct, probs_struct = train_one_fold(
                patches_1ch[train_idx], labels[train_idx], patches_1ch[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED, model_factory=SimplePatchCNN,
            )
            acc_struct = hist_struct["val_acc"][-1]

            _, hist_phys, pred_phys, probs_phys = train_one_fold(
                patches_3ch[train_idx], labels[train_idx], patches_3ch[test_idx], labels[test_idx],
                epochs=15, device="cpu", seed=SEED,
                model_factory=lambda: SimplePatchCNN(in_channels=3),
            )
            acc_phys = hist_phys["val_acc"][-1]

            print(f"  {fold_name}: только структура(T1)={acc_struct:.3f}  +базовые CBF/OEF={acc_phys:.3f}")
            all_results[(contrast, fold_name)] = {
                "structural_only": acc_struct, "structural_plus_cbf_oef": acc_phys, "n_subjects": n_subjects,
            }
            viz.plot_confusion_and_roc(
                labels[test_idx], pred_phys, probs_phys, f"CBF/OEF {contrast} {fold_name}: T1+baseline CBF/OEF",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_cbf_oef_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    labels_x = [f"{c}\n{f}" for (c, f) in all_results.keys()]
    struct_vals = [all_results[k]["structural_only"] for k in all_results]
    phys_vals = [all_results[k]["structural_plus_cbf_oef"] for k in all_results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels_x))
    width = 0.35
    ax.bar(x - width / 2, struct_vals, width, label="только T1", color="#7f8c8d")
    ax.bar(x + width / 2, phys_vals, width, label="T1 + базовые CBF/OEF", color="#c0392b")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайный уровень")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("Только T1 против T1+базовые CBF/OEF (физиологическая структура)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "cbf_oef_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено {out_path}")


if __name__ == "__main__":
    main()
