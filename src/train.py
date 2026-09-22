"""
Эксперимент 1 (один пациент, только структурные данные): обучить небольшую
3D CNN на патчах из одной половины мозга (например, левого полушария)
предсказывать concordant vs discordant, протестировать на другой половине,
затем повторить с половинами, поменянными местами. Это защищённая от
утечки схема валидации, которую попросил руководитель перед масштабированием
на больше пациентов / добавлением BOLD как дополнительного входа.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.patches import extract_patches, labeled_voxel_coords, split_by_axis, hemisphere_midpoint
from src.model import SimplePatchCNN, PatchBOLDNet
from src.bold_features import extract_bold_vectors, normalize_bold_vectors
from src import viz


class PatchDataset(TensorDataset):
    pass


def _normalize_patch_intensity(patches, brain_mean, brain_std):
    return (patches - brain_mean) / (brain_std + 1e-6)


def select_labeled_coords(label_volume, brain_mask, max_voxels=None, seed=0):
    """
    Только координаты + метки, без патчей (дёшево - извлечение патчей было
    тем, что разрывало память при выполнении для каждого из ~150k+ размеченных
    вокселей пациента вместо лишь той небольшой части, что реально
    используется для обучения/теста). Возвращает coords - целочисленный
    массив (N,3), labels - массив (N,) float32 со значениями {0,1}.
    """
    coords = labeled_voxel_coords(label_volume, brain_mask)
    if max_voxels is not None and len(coords) > max_voxels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_voxels, replace=False)
        coords = coords[idx]
    raw_labels = label_volume[coords[:, 0], coords[:, 1], coords[:, 2]]
    labels = (raw_labels > 0).astype(np.float32)  # 1=concordant, 0=discordant
    return coords, labels


def extract_and_normalize_patches(t1_volume, coords, patch_size, brain_mask):
    """patches (N,1,p,p,p) float32, нормализованные по интенсивности статистикой всего мозга."""
    patches = extract_patches(t1_volume, coords, patch_size)
    brain_vals = t1_volume[brain_mask.astype(bool)]
    patches = _normalize_patch_intensity(patches, brain_vals.mean(), brain_vals.std())
    return patches[:, None, :, :, :].astype(np.float32)  # add channel dim


def extract_and_normalize_multichannel_patches(volumes, coords, patch_size, brain_mask):
    """
    То же самое, что extract_and_normalize_patches, но сразу для нескольких
    совмещённых (co-registered) объёмов (например, T1 + базовый CBF +
    базовый OEF), каждый нормализуется независимо собственной статистикой
    всего мозга перед объединением в каналы - физические единицы и
    масштабы интенсивности T1, CBF (мл/100г/мин) и OEF (доля) сильно
    отличаются, так что общая константа нормализации позволила бы одному
    каналу доминировать просто за счёт масштаба.
    volumes: список массивов (X,Y,Z) одинаковой формы, уже совмещённых.
    Возвращает (N, C, p, p, p) float32.
    """
    channels = []
    for vol in volumes:
        patches = extract_patches(vol, coords, patch_size)
        brain_vals = vol[brain_mask.astype(bool)]
        channels.append(_normalize_patch_intensity(patches, brain_vals.mean(), brain_vals.std()))
    return np.stack(channels, axis=1).astype(np.float32)  # (N, C, p, p, p)


def augment_patch_batch(xb, rng_state=None):
    """
    Случайное отражение вдоль каждой пространственной оси (измерения
    2,3,4 из (N,1,p,p,p)), независимо по каждой оси, применяется сразу ко
    всему батчу. Стандартная, сохраняющая метку аугментация для 3D CNN на
    маленьких патчах: модель видит только локальное содержимое патча
    (никогда не абсолютное положение в мозге), поэтому зеркалирование
    патча никак не меняет смысл метки.
    """
    for dim in (2, 3, 4):
        if torch.rand(1).item() < 0.5:
            xb = torch.flip(xb, dims=[dim])
    return xb


def train_one_fold(train_patches, train_labels, test_patches, test_labels,
                    epochs=15, batch_size=32, lr=1e-3, device="cpu", seed=0,
                    model_factory=SimplePatchCNN, augment=False, lr_schedule=False):
    torch.manual_seed(seed)
    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if lr_schedule else None
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(train_patches), torch.from_numpy(train_labels))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_test = torch.from_numpy(test_patches).to(device)
    y_test = torch.from_numpy(test_labels).to(device)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            if augment:
                xb = augment_patch_batch(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += ((logits > 0).float() == yb).sum().item()
            total += len(yb)
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            val_acc = ((test_logits > 0).float() == y_test).float().mean().item()

        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_logits > 0).float().cpu().numpy()

    return model, history, test_pred, test_probs


def train_one_fold_multiclass(train_x, train_labels, test_x, test_labels,
                               n_classes, epochs=15, batch_size=32, lr=1e-3, device="cpu", seed=0,
                               model_factory=None):
    """Та же схема, что и train_one_fold, но для >2 целочисленных меток
    классов (0..n_classes-1) через CrossEntropyLoss - используется для
    3-классовой постановки concordant/discordant/unreliable. train_labels/
    test_labels: массивы int64. model_factory должна возвращать модуль, чей
    forward даёт логиты (N, n_classes) (например, RegionMLP(n_classes=3)
    или любую патч-модель с out_features последнего слоя, равным
    n_classes)."""
    torch.manual_seed(seed)
    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_labels).long())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_test = torch.from_numpy(test_x).to(device)
    y_test = torch.from_numpy(test_labels).long().to(device)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            val_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_pred = test_logits.argmax(dim=1).cpu().numpy()

    return model, history, test_pred, test_probs


def run_hemisphere_experiment(
    t1_volume, label_volume, brain_mask, affine,
    patch_size=9, axis_index=0, margin_vox=None,
    max_voxels_per_side=1500, epochs=15, out_dir="results", subject_name="subject",
    device="cpu", seed=0, model_factory=SimplePatchCNN,
):
    """
    Прогоняет оба направления разбиения (сторона A - train / сторона B -
    test, и наоборот) для одного пациента, сохраняет диагностические
    графики в out_dir и возвращает словарь результатов. margin_vox по
    умолчанию равен patch_size // 2 (минимум, необходимый, чтобы
    гарантировать отсутствие пересечения патчей train/test по обе стороны
    разбиения).
    """
    os.makedirs(out_dir, exist_ok=True)
    if margin_vox is None:
        margin_vox = patch_size // 2

    # Координаты + метки для каждого размеченного вокселя (дёшево - патчей пока нет).
    coords, labels = select_labeled_coords(label_volume, brain_mask, max_voxels=None, seed=seed)
    midpoint = hemisphere_midpoint(t1_volume.shape, axis_index)
    side_a, side_b = split_by_axis(coords, axis_index, midpoint, margin_vox)

    mid_slice_idx = t1_volume.shape[2] // 2
    plane_mask = np.abs(coords[:, 2] - mid_slice_idx) <= 2
    axes_2d = [i for i in range(3) if i != 2]
    viz.plot_hemisphere_split(
        t1_volume[:, :, mid_slice_idx],
        coords[plane_mask][:, axes_2d],
        side_a[plane_mask], side_b[plane_mask], midpoint,
        f"{subject_name}: hemisphere split (axis {axis_index}, margin={margin_vox}vox)",
        os.path.join(out_dir, f"{subject_name}_split_axis{axis_index}.png"),
    )

    def subsample(mask):
        idx = np.where(mask)[0]
        if len(idx) > max_voxels_per_side:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=max_voxels_per_side, replace=False)
        return idx

    # Только начиная отсюда извлекаются патчи - и только для ~2*max_voxels_per_side
    # реально используемых вокселей, а не для каждого размеченного вокселя в мозге.
    idx_a, idx_b = subsample(side_a), subsample(side_b)
    used_idx = np.concatenate([idx_a, idx_b])
    used_patches = extract_and_normalize_patches(t1_volume, coords[used_idx], patch_size, brain_mask)
    used_labels = labels[used_idx]
    pos_a = np.arange(len(idx_a))
    pos_b = np.arange(len(idx_a), len(idx_a) + len(idx_b))

    viz.plot_patch_examples(
        used_patches[:8, 0], used_labels[:8], os.path.join(out_dir, f"{subject_name}_patch_examples.png")
    )

    fold_results = {}
    for fold_name, (train_idx, test_idx) in {
        "A_train_B_test": (pos_a, pos_b),
        "B_train_A_test": (pos_b, pos_a),
    }.items():
        model, history, test_pred, test_probs = train_one_fold(
            used_patches[train_idx], used_labels[train_idx],
            used_patches[test_idx], used_labels[test_idx],
            epochs=epochs, device=device, seed=seed, model_factory=model_factory,
        )
        viz.plot_training_curves(
            history, f"{subject_name}: {fold_name}",
            os.path.join(out_dir, f"{subject_name}_{fold_name}_curves.png"),
        )
        viz.plot_confusion_and_roc(
            used_labels[test_idx], test_pred, test_probs, f"{subject_name}: {fold_name}",
            os.path.join(out_dir, f"{subject_name}_{fold_name}_eval.png"),
        )
        fold_results[fold_name] = history["val_acc"][-1]

    viz.plot_fold_accuracy_summary(
        fold_results, os.path.join(out_dir, f"{subject_name}_fold_accuracy_summary.png")
    )
    return fold_results


def train_one_fold_regression(train_patches, train_targets, test_patches, test_targets,
                               epochs=15, batch_size=32, lr=1e-3, device="cpu", seed=0,
                               model_factory=SimplePatchCNN):
    """
    Альтернативная к train_one_fold постановка: регрессировать непрерывное
    значение CMRO2_percchange напрямую (сырой скалярный выход
    SimplePatchCNN, без сигмоиды) вместо классификации его знака в
    сочетании со знаком BOLD. Сохраняет информацию о величине, которую
    выбрасывает бинарная метка concordant/discordant.

    Целевые значения стандартизуются с использованием среднего/std только
    по ОБУЧАЮЩЕЙ выборке (статистика тестовой выборки не просачивается в
    обучение) перед подгонкой; итоговые MSE/R2 приводятся в этой
    стандартизованной шкале, напрямую сравнимой между разбиениями/контрастами.
    """
    torch.manual_seed(seed)
    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    train_mean = float(train_targets.mean())
    train_std = float(train_targets.std()) + 1e-6
    train_targets_norm = ((train_targets - train_mean) / train_std).astype(np.float32)
    test_targets_norm = ((test_targets - train_mean) / train_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(train_patches), torch.from_numpy(train_targets_norm))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_test = torch.from_numpy(test_patches).to(device)
    y_test = torch.from_numpy(test_targets_norm).to(device)

    history = {"train_loss": [], "test_mse": [], "test_r2": []}
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(x_test)
            mse = loss_fn(test_pred, y_test).item()
            ss_res = ((y_test - test_pred) ** 2).sum().item()
            ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        history["train_loss"].append(float(np.mean(losses)))
        history["test_mse"].append(mse)
        history["test_r2"].append(r2)

    model.eval()
    with torch.no_grad():
        test_pred = model(x_test).cpu().numpy()

    return model, history, test_pred, test_targets_norm


def train_one_fold_multimodal(train_patches, train_bold, train_labels,
                               test_patches, test_bold, test_labels,
                               epochs=15, batch_size=32, lr=1e-3, device="cpu", seed=0,
                               model_factory=None, bold_len=None):
    """Тот же цикл обучения, что и train_one_fold, но для двухветвевой
    модели (патч, вектор признаков) - по умолчанию PatchBOLDNet (сырой
    временной ряд), или, например, PatchBOLDConditionNet (компактные
    признаки по условию) через model_factory."""
    torch.manual_seed(seed)
    if model_factory is None:
        model_factory = lambda: PatchBOLDNet(bold_len=bold_len or train_bold.shape[1])
    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_patches), torch.from_numpy(train_bold), torch.from_numpy(train_labels)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_test_patch = torch.from_numpy(test_patches).to(device)
    x_test_bold = torch.from_numpy(test_bold).to(device)
    y_test = torch.from_numpy(test_labels).to(device)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        losses, correct, total = [], 0, 0
        for xb_patch, xb_bold, yb in train_dl:
            xb_patch, xb_bold, yb = xb_patch.to(device), xb_bold.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb_patch, xb_bold)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += ((logits > 0).float() == yb).sum().item()
            total += len(yb)

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test_patch, x_test_bold)
            val_acc = ((test_logits > 0).float() == y_test).float().mean().item()

        history["train_loss"].append(float(np.mean(losses)))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_patch, x_test_bold)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_logits > 0).float().cpu().numpy()

    return model, history, test_pred, test_probs


def run_hemisphere_experiment_multimodal(
    t1_volume, bold_4d, label_volume, brain_mask, affine,
    patch_size=9, axis_index=0, margin_vox=None,
    max_voxels_per_side=1500, epochs=15, out_dir="results", subject_name="subject",
    device="cpu", seed=0,
):
    """
    Эксперимент 2: та же защищённая от утечки схема разбиения по
    полушариям, что и в run_hemisphere_experiment, но каждый воксель
    представлен ОБОИМИ - своим структурным патчем И своим (детрендированным,
    z-нормализованным) BOLD-временным рядом из bold_4d, подаваемыми в две
    ветви PatchBOLDNet.
    """
    os.makedirs(out_dir, exist_ok=True)
    if margin_vox is None:
        margin_vox = patch_size // 2

    coords, labels = select_labeled_coords(label_volume, brain_mask, max_voxels=None, seed=seed)
    midpoint = hemisphere_midpoint(t1_volume.shape, axis_index)
    side_a, side_b = split_by_axis(coords, axis_index, midpoint, margin_vox)

    def subsample(mask):
        idx = np.where(mask)[0]
        if len(idx) > max_voxels_per_side:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=max_voxels_per_side, replace=False)
        return idx

    idx_a, idx_b = subsample(side_a), subsample(side_b)
    used_idx = np.concatenate([idx_a, idx_b])
    used_coords = coords[used_idx]

    used_patches = extract_and_normalize_patches(t1_volume, used_coords, patch_size, brain_mask)
    used_bold = normalize_bold_vectors(extract_bold_vectors(bold_4d, used_coords))
    used_labels = labels[used_idx]

    pos_a = np.arange(len(idx_a))
    pos_b = np.arange(len(idx_a), len(idx_a) + len(idx_b))

    fold_results = {}
    for fold_name, (train_idx, test_idx) in {
        "A_train_B_test": (pos_a, pos_b),
        "B_train_A_test": (pos_b, pos_a),
    }.items():
        model, history, test_pred, test_probs = train_one_fold_multimodal(
            used_patches[train_idx], used_bold[train_idx], used_labels[train_idx],
            used_patches[test_idx], used_bold[test_idx], used_labels[test_idx],
            epochs=epochs, device=device, seed=seed,
        )
        viz.plot_training_curves(
            history, f"{subject_name}: {fold_name}",
            os.path.join(out_dir, f"{subject_name}_{fold_name}_curves.png"),
        )
        viz.plot_confusion_and_roc(
            used_labels[test_idx], test_pred, test_probs, f"{subject_name}: {fold_name}",
            os.path.join(out_dir, f"{subject_name}_{fold_name}_eval.png"),
        )
        fold_results[fold_name] = history["val_acc"][-1]

    viz.plot_fold_accuracy_summary(
        fold_results, os.path.join(out_dir, f"{subject_name}_fold_accuracy_summary.png")
    )
    return fold_results
