"""
Более прямой перенос идеи MS-DSA-NET (link.springer.com/article/10.1186/
s13244-024-01803-8, модель сегментации очагов ФКД): вместо классификации
одного патча 9^3 вокруг вокселя (как во всех остальных экспериментах
проекта и в scripts/run_multiscale_transformer.py) сеть
(src/dense_model.py::DenseConcordanceNet, encoder-decoder + трансформер на
боттлнеке) выдаёт плотную карту concordant/discordant на ВЕСЬ объём мозга
одним проходом, как и в оригинальной статье.

Защита от утечки для плотного подхода принципиально отличается от
патчевой (src/patches.py::split_by_axis с margin = patch_size // 2): сеть
с трансформером на боттлнеке имеет глобальное рецептивное поле внутри
видимой половины, так что вырезание узкой полосы вокруг границы (как для
патчей) не защитило бы от утечки. Вместо этого вход целиком обнуляется на
невидимой стороне ПЕРЕД подачей в сеть и на этапе обучения, и на этапе
теста (src/patches.py::side_mask_volume) - ноль одинаков у всех пациентов
и не несёт информации об анатомии пациента с той стороны, так что никакая
реальная воксельная структура «невидимой» половины никогда физически не
попадает в сеть ни на одном из двух этапов. Небольшой margin вокруг
границы (MARGIN_VOX) здесь не требование защиты от утечки (ноль уже
достаточен), а чисто косметическая мера - не обучать/оценивать сеть на
вокселях, чья входная окрестность прямо на границе содержит резкий
переход "реальные данные/ноль".

Это разведочный, ограниченный по объёму эксперимент (как и было
согласовано): только контраст "calc" (доступен у большинства когорты, в
отличие от "mem", ограниченного исходными ~25 пациентами) и небольшое
число эпох - полная сеть с глобальным вниманием на весь мозг при ~39
независимых обучающих объектах (целые объёмы, не патчи с подвыборкой) -
серьёзный риск переобучения, который стоит явно держать в уме при
интерпретации результата, а не повод не проверять вовсе.
"""

import os
import sys
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.cohort import ALL_SUBJECTS
from src.labeling import concordance_label
from src.patches import side_mask_volume
from src.dense_model import DenseConcordanceNet
from src.subject_loader import load_subject_robust

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dense_segmentation")
CONTRAST = "calc"
CANONICAL_SHAPE = (112, 112, 35)  # максимум по когорте на каждой оси (см. проверку формы перед запуском)
MARGIN_VOX = 3  # косметический буфер у границы, не требование защиты от утечки (см. docstring модуля)
EPOCHS = 6
BATCH_SIZE = 4
BASE_CHANNELS = 4
LR = 1e-3
SEED = 0


def build_label_and_valid(cmro2, bold_pct, mask, contrast):
    """Как build_label в остальных скриптах, но возвращает ещё и булеву
    маску valid отдельно - для плотного подхода нужно явно знать, какие
    воксели размечены, а не только итоговый знак метки."""
    if contrast not in cmro2 or "control" not in cmro2 or contrast not in bold_pct:
        return None, None
    cmro2_task, cmro2_control, bold = cmro2[contrast], cmro2["control"], bold_pct[contrast]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (cmro2_task - cmro2_control) / cmro2_control * 100
    pct[~np.isfinite(pct)] = 0
    valid = (cmro2_control != 0) & (cmro2_task != 0) & (bold != 0) & mask.astype(bool)
    raw_label = np.zeros(cmro2_task.shape, dtype=np.float32)
    raw_label[valid] = concordance_label(bold, pct)[valid]
    label01 = (raw_label > 0).astype(np.float32)  # 1=concordant, 0=discordant - то же соглашение, что в select_labeled_coords
    return label01, valid


def pad_to_canonical(volume, canonical_shape, fill=0):
    pad = [(0, c - s) for s, c in zip(volume.shape, canonical_shape)]
    if any(p[1] < 0 for p in pad):
        return None  # объём больше канонической формы - такого в этой когорте не встречается, но на всякий случай
    return np.pad(volume, pad, mode="constant", constant_values=fill)


def prepare_subject(sub):
    """Возвращает (t1_norm_padded, valid_padded, label_padded) формы
    CANONICAL_SHAPE, нормализованные по интенсивности статистикой мозга
    САМОГО пациента до паддинга (как и везде в проекте - см.
    src/train.py::extract_and_normalize_patches), или None при отсутствии
    нужных данных."""
    try:
        result, notes = load_subject_robust(sub)
    except Exception as e:
        print(f"  [ошибка] {sub}: {e}")
        return None
    if result is None:
        print(f"  [пропуск] {sub}: {notes}")
        return None
    t1, affine, mask, cmro2, bold_pct = result

    label01, valid = build_label_and_valid(cmro2, bold_pct, mask, CONTRAST)
    if label01 is None or int(valid.sum()) < 40:
        print(f"  [пропуск] {sub}: нет пригодной разметки для контраста {CONTRAST}")
        return None

    brain_vals = t1[mask.astype(bool)]
    t1_norm = (t1 - brain_vals.mean()) / (brain_vals.std() + 1e-6)

    t1_p = pad_to_canonical(t1_norm, CANONICAL_SHAPE, fill=0.0)
    valid_p = pad_to_canonical(valid.astype(np.float32), CANONICAL_SHAPE, fill=0.0) > 0.5
    label_p = pad_to_canonical(label01, CANONICAL_SHAPE, fill=0.0)
    if t1_p is None:
        print(f"  [пропуск] {sub}: форма {t1.shape} больше канонической {CANONICAL_SHAPE}")
        return None
    return t1_p.astype(np.float32), valid_p, label_p.astype(np.float32)


def run_fold(fold_name, train_mask, test_mask, subjects_data, epochs=EPOCHS):
    """train_mask/test_mask: булевы массивы формы CANONICAL_SHAPE (видимая
    + размеченная половина для обучения и для теста соответственно).
    Для каждого пациента вход на обучении обнуляется вне train_mask, вход
    на тесте обнуляется вне test_mask - сеть никогда не получает реальные
    воксели противоположной стороны ни на одном из двух этапов."""
    torch.manual_seed(SEED)
    model = DenseConcordanceNet(img_size=CANONICAL_SHAPE, base_channels=BASE_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_mask_t = torch.from_numpy(train_mask.astype(np.float32))
    test_mask_t = torch.from_numpy(test_mask.astype(np.float32))

    t1_all = torch.stack([torch.from_numpy(d[0]) for d in subjects_data])  # (N, D, H, W)
    valid_all = torch.stack([torch.from_numpy(d[1].astype(np.float32)) for d in subjects_data])
    label_all = torch.stack([torch.from_numpy(d[2]) for d in subjects_data])
    n = t1_all.shape[0]

    rng = np.random.default_rng(SEED)
    history_train_loss = []
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        epoch_losses = []
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = (t1_all[idx] * train_mask_t).unsqueeze(1)  # (B,1,D,H,W), обнулено вне train_mask
            loss_mask = (valid_all[idx] * train_mask_t) > 0.5
            if loss_mask.sum() == 0:
                continue
            target = label_all[idx]

            logits = model(x).squeeze(1)  # (B,D,H,W)
            loss = F.binary_cross_entropy_with_logits(logits[loss_mask], target[loss_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history_train_loss.append(mean_loss)
        print(f"    [{fold_name}] эпоха {epoch+1}/{epochs}: train_loss={mean_loss:.4f}")

    model.eval()
    n_correct, n_total, n_pos_pred, n_pos_true = 0, 0, 0, 0
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            idx = np.arange(start, min(start + BATCH_SIZE, n))
            x = (t1_all[idx] * test_mask_t).unsqueeze(1)
            eval_mask = (valid_all[idx] * test_mask_t) > 0.5
            if eval_mask.sum() == 0:
                continue
            target = label_all[idx]
            logits = model(x).squeeze(1)
            pred = (torch.sigmoid(logits) > 0.5).float()
            n_correct += int((pred[eval_mask] == target[eval_mask]).sum())
            n_total += int(eval_mask.sum())
            n_pos_pred += int(pred[eval_mask].sum())
            n_pos_true += int(target[eval_mask].sum())

    test_acc = n_correct / n_total if n_total else float("nan")
    majority_baseline = max(n_pos_true, n_total - n_pos_true) / n_total if n_total else float("nan")
    return {
        "test_acc": test_acc, "majority_baseline": majority_baseline,
        "n_test_voxels": n_total, "train_loss_history": history_train_loss,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    subjects_data, used_subjects, log = [], [], []
    for i, sub in enumerate(ALL_SUBJECTS):
        print(f"[{i+1}/{len(ALL_SUBJECTS)}] {sub}")
        try:
            d = prepare_subject(sub)
        except Exception:
            print(f"  [ошибка] {sub}:\n{traceback.format_exc()}")
            d = None
        if d is None:
            log.append({"subject": sub, "status": "skipped"})
            continue
        subjects_data.append(d)
        used_subjects.append(sub)
        log.append({"subject": sub, "status": "used"})

    with open(os.path.join(OUT_DIR, "subject_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=str)
    print(f"\n{len(used_subjects)}/{len(ALL_SUBJECTS)} пациентов использовано (контраст {CONTRAST})")

    midpoint = CANONICAL_SHAPE[0] / 2.0
    mask_a, mask_b = side_mask_volume(CANONICAL_SHAPE, axis_index=0, midpoint=midpoint, margin=MARGIN_VOX)

    all_results = {}
    for fold_name, (train_side, test_side) in {
        "A_train_B_test": (mask_a, mask_b),
        "B_train_A_test": (mask_b, mask_a),
    }.items():
        print(f"\n=== {fold_name} ===")
        result = run_fold(fold_name, train_side, test_side, subjects_data)
        beats = "ДА" if result["test_acc"] > result["majority_baseline"] else "нет"
        print(f"  {fold_name}: точность_на_тесте={result['test_acc']:.3f}  "
              f"базовый_уровень={result['majority_baseline']:.3f}  превышает={beats}  "
              f"n_вокселей_теста={result['n_test_voxels']}")
        all_results[fold_name] = result

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=1, default=float)

    elapsed = time.time() - t_start
    print(f"\nГотово за {elapsed/60:.1f} мин. Сохранено: {os.path.join(OUT_DIR, 'all_results.json')}")


if __name__ == "__main__":
    main()
