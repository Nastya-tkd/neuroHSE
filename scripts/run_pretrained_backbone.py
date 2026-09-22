"""
Последний неопробованный архитектурный рычаг из раздела README "Куда это
привело проект": 3D-MRI backbone, предобученный на реальных внешних данных
(23andme... нет - на корпусе Med3D для мультиорганной сегментации из 23
датасетов), дообученный здесь, а не ещё одна архитектура, обученная с нуля
на нашей маленькой, уже исчерпанной объединённой выборке. Каждая архитектура,
обученная с нуля (обычная CNN, остаточная CNN, гибрид с трансформером,
настоящий U-Net), сошлась к одному и тому же результату на уровне случайного
угадывания - это единственный оставшийся рычаг, который привносит подлинно
внешние данные, а не просто увеличивает число параметров.

Веса: resnet_50_23dataset.pth от Tencent/MedicalNet (Med3D, Chen et al.
2019), ствол 3D ResNet50 с 46,2 млн параметров, получен из собственного
GitHub-релиза пользователя (это не сетевой хост, заблокированный политикой
исходящих соединений этой сессии - см. переписку) и проверен перед
загрузкой: заголовок файла соответствует документированному устаревшему
формату torch.save, а статическое сканирование pickle-опкодов (без
выполнения кода) обнаружило только ожидаемые вызовы восстановления
torch/collections, без подозрительных глобальных объектов.

Архитектура: src/medicalnet_resnet.py воспроизводит ствол (conv1/bn1/
layer1-4, блоки Bottleneck 3-4-6-3) с ТОЧНЫМ совпадением ключей state_dict,
проверенным через strict=True - это не реализация "по мотивам", а точное
воспроизведение. conv1 уже принимает 1 входной канал (T1, как и наши
патчи), поэтому хирургия первого слоя не требуется, в отличие от типичного
3-канального ImageNet-backbone. layer3/layer4 используют dilation вместо
stride (Med3D сохраняет высокое разрешение для сегментации), поэтому общее
понижение пространственного разрешения составляет всего ~8x, а не обычные
32x - вход с patch_size=25 всё ещё оставляет карту признаков 4x4x4 перед
финальным пулингом, а не схлопывается в ничто.

Подход: ЗАМОРОЖЕННЫЙ ствол (только извлечение признаков, без градиентов, без
обновления статистик BatchNorm - цель в том, чтобы проверить, полезны ли
для данной метки общие 3D-медицинские признаки, выученные Med3D, а не
переоткрыть их заново) + небольшая обучаемая MLP-голова
(src/model.py:PretrainedFeatureHead) поверх 2048-мерных агрегированных
признаков. Выбрано ради выполнимости на CPU: вычисление прямого прохода
46-миллионного ствола один раз на патч (а не один раз на эпоху) делает
многоэпоховую часть обучения дешёвой. Полное дообучение без заморозки
остаётся возможным следующим шагом, если здесь обнаружится сигнал, который
стоит развивать.
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
from src.train import select_labeled_coords, extract_and_normalize_patches, train_one_fold
from src.model import PretrainedFeatureHead
from src.medicalnet_resnet import load_pretrained_trunk, extract_backbone_features
from src.subject_loader import load_subject_robust
from src import viz

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_backbone")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "pretrained_weights", "resnet_50_23dataset.pth")
CONTRASTS = ["calc", "mem"]
PATCH_SIZE = 25  # даёт карту признаков ствола 4x4x4x2048 (см. docstring модуля); patch=9/15 оставляют только 2x2x2
MAX_VOXELS_PER_SIDE_PER_SUBJECT = 300
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


def process_subject(sub, backbone):
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [ошибка] {sub}: {e}")
        return None
    if result is None:
        print(f"  [пропуск] {sub}: {notes}")
        return None
    t1, affine, mask, cmro2, bold_pct = result
    if notes:
        print(f"  [примечания] {sub}: {notes}")

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
            print(f"  [пропуск контраста] {sub} {contrast}: вырожденная метка")
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
        feats = extract_backbone_features(backbone, patches)  # (N, 2048) - патчи сразу же отбрасываются
        out[contrast] = {"feats": feats, "labels": labels[used_idx], "side": side_tags}

    return out if out else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Загрузка предобученного ствола MedicalNet ResNet50...")
    backbone = load_pretrained_trunk(CHECKPOINT_PATH)
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Загружено, {n_params:,} параметров, заморожен.")

    pooled = {c: {"feats": [], "labels": [], "side": [], "subject": []} for c in CONTRASTS}
    log = []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            result = process_subject(sub, backbone)
        except Exception:
            print(f"  [ошибка] {sub}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            log.append({"subject": sub, "status": "skipped"})
            continue
        log.append({"subject": sub, "status": "used", "contrasts": sorted(result.keys())})
        for contrast, d in result.items():
            pooled[contrast]["feats"].append(d["feats"])
            pooled[contrast]["labels"].append(d["labels"])
            pooled[contrast]["side"].append(d["side"])
            pooled[contrast]["subject"].append(np.full(len(d["labels"]), sub))

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)

    all_results = {}
    for contrast in CONTRASTS:
        if not pooled[contrast]["feats"]:
            print(f"{contrast}: нет пригодных пациентов, пропуск")
            continue
        feats = np.concatenate(pooled[contrast]["feats"], axis=0)
        labels = np.concatenate(pooled[contrast]["labels"], axis=0)
        side = np.concatenate(pooled[contrast]["side"], axis=0)
        n_subjects = len(set(np.concatenate(pooled[contrast]["subject"], axis=0).tolist()))

        # стандартизация признаков с использованием mean/std только TRAIN-стороны каждого разбиения, выполняется внутри цикла ниже
        pos_a, pos_b = np.where(side == "A")[0], np.where(side == "B")[0]
        print(f"\n=== {contrast}: {n_subjects} пациентов, {len(pos_a)}/{len(pos_b)} вокселей A/B ===")

        for fold_name, (train_idx, test_idx) in {
            "A_train_B_test": (pos_a, pos_b),
            "B_train_A_test": (pos_b, pos_a),
        }.items():
            mu = feats[train_idx].mean(axis=0, keepdims=True)
            sd = feats[train_idx].std(axis=0, keepdims=True) + 1e-6
            feats_norm = (feats - mu) / sd

            _, hist, pred, probs = train_one_fold(
                feats_norm[train_idx].astype(np.float32), labels[train_idx],
                feats_norm[test_idx].astype(np.float32), labels[test_idx],
                epochs=30, device="cpu", seed=SEED, model_factory=PretrainedFeatureHead,
            )
            acc = hist["val_acc"][-1]
            best_acc = max(hist["val_acc"])
            print(f"  {fold_name}: итоговая_точность={acc:.3f} лучшая_точность={best_acc:.3f}")
            all_results[(contrast, fold_name)] = {"final_acc": acc, "best_acc": best_acc, "n_subjects": n_subjects}
            viz.plot_training_curves(
                hist, f"pretrained-backbone {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_curves.png"),
            )
            viz.plot_confusion_and_roc(
                labels[test_idx], pred, probs, f"pretrained-backbone {contrast} {fold_name}",
                os.path.join(OUT_DIR, f"{contrast}_{fold_name}_eval.png"),
            )

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump({f"{c}|{f}": r for (c, f), r in all_results.items()}, f, indent=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = list(all_results.keys())
    vals = [all_results[k]["final_acc"] for k in keys]
    ax.bar([f"{c}\n{f}" for c, f in keys], vals, color="#c0392b")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайный уровень")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон")
    ax.set_ylim(0, 1)
    ax.set_title("Предобученный MedicalNet ResNet50 (заморожен) + MLP-голова, patch=25")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "pretrained_backbone_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено {out_path}")


if __name__ == "__main__":
    main()
