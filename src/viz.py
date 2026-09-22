"""
Вспомогательные функции построения графиков для эксперимента структурной
классификации. Цвета следуют тому же соглашению red=concordant /
blue=discordant, что используется в собственной цветовой карте BlueRed
исходного репозитория и в описании руководителя.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

CONCORDANT_COLOR = "#c0392b"   # красный
DISCORDANT_COLOR = "#2471a3"   # синий
NEUTRAL_COLOR = "#5d6d7e"


def plot_hemisphere_split(t1_slice, coords_2d, side_a_mask, side_b_mask, midline, title, out_path):
    """t1_slice: 2D фоновый срез. coords_2d: координаты вокселей (N,2) в
    плоскости этого среза. Раскрашивает воксели по тому, к какой стороне
    разбиения (или отброшенному промежутку) они относятся."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(t1_slice.T, cmap="gray", origin="lower")
    dropped = ~(side_a_mask | side_b_mask)
    ax.scatter(*coords_2d[side_a_mask].T, s=2, color="#e67e22", label="сторона A (train/test)")
    ax.scatter(*coords_2d[side_b_mask].T, s=2, color="#16a085", label="сторона B (test/train)")
    if dropped.any():
        ax.scatter(*coords_2d[dropped].T, s=2, color="#7f8c8d", alpha=0.5, label="отступ (исключено)")
    ax.axvline(midline, color="white", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_patch_examples(patches, labels, out_path, n=8):
    """Средний аксиальный срез n примеров патчей, подписанных concordant/discordant."""
    n = min(n, len(patches))
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    if n == 1:
        axes = [axes]
    mid = patches.shape[-1] // 2
    for ax, patch, label in zip(axes, patches[:n], labels[:n]):
        ax.imshow(patch[:, :, mid], cmap="gray")
        color = CONCORDANT_COLOR if label == 1 else DISCORDANT_COLOR
        ax.set_title("concordant" if label == 1 else "discordant", color=color, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, title, out_path):
    """history: словарь со списками 'train_loss', 'train_acc', 'val_acc'."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(history["train_loss"], color=NEUTRAL_COLOR)
    ax1.set_title("функция потерь на обучении")
    ax1.set_xlabel("эпоха")

    ax2.plot(history["train_acc"], label="точн. на обучении", color="#8e44ad")
    ax2.plot(history["val_acc"], label="точн. на тесте", color="#16a085")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="случайность")
    ax2.set_ylim(0, 1)
    ax2.set_title("точность")
    ax2.set_xlabel("эпоха")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_and_roc(y_true, y_pred, y_score, title, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["discordant", "concordant"], fontsize=8)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["discordant", "concordant"], fontsize=8)
    ax1.set_xlabel("предсказано"); ax1.set_ylabel("истина")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax1.set_title("матрица ошибок")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="#8e44ad", label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax2.set_xlabel("доля ложноположительных"); ax2.set_ylabel("доля истинноположительных")
    ax2.set_title("ROC")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multiclass_confusion(y_true, y_pred, class_names, title, out_path):
    """Общая матрица ошибок для n классов (используется для 3-классовой
    постановки concordant/discordant/unreliable - plot_confusion_and_roc
    выше жёстко рассчитана на 2 класса плюс ROC-кривую, что не применимо,
    когда появляется третий класс)."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("предсказано"); ax.set_ylabel("истина")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fold_accuracy_summary(fold_results, out_path):
    """fold_results: dict[fold_name] -> точность на test."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = list(fold_results.keys())
    accs = [fold_results[n] for n in names]
    bars = ax.bar(names, accs, color="#8e44ad")
    ax.axhline(0.5, color="gray", linestyle=":", label="случайность")
    ax.axhspan(0.65, 0.70, color="#16a085", alpha=0.15, label="целевой диапазон (0.65-0.70)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("точность на тесте")
    ax.set_title("классификация только по структуре: точность по разбиениям")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regression_scatter(y_true, y_pred, r2, title, out_path):
    """Предсказанное vs истинное (стандартизованное) CMRO2_percchange, одна точка на тестовый воксель."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.15, color="#8e44ad")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="gray", linestyle=":", linewidth=1, label="идеальное предсказание")
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set_xlabel("истинное CMRO2 %change (стандартизовано)")
    ax.set_ylabel("предсказанное CMRO2 %change (стандартизовано)")
    ax.set_title(f"{title}\nR2 = {r2:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
