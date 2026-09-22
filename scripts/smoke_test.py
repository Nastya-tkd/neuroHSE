"""
SMOKE-ТЕСТ - НЕ НАУЧНЫЙ РЕЗУЛЬТАТ.

Запускает полный конвейер Эксперимента 1 (извлечение патчей -> разбиение по
полушариям -> обучение -> оценка -> графики) от начала до конца на
единственном реальном структурном скане, который у нас сейчас есть
(sub-p019, T1w), чтобы доказать работоспособность кода до появления
реальных меток concordant/discordant.

Для реальных меток нужны карты CMRO2 (принцип Фика: CBF*OEF*CaO2) и
BOLD_percchange для этого пациента, которые пока не предоставлены (см.
README.md, раздел "Текущий статус данных"). Поскольку их нет, этот скрипт
вместо этого строит СИНТЕТИЧЕСКУЮ метку из самой интенсивности T1 (воксели
выше медианной интенсивности по всему мозгу, плюс шум, произвольно
называются "concordant"). Это не имеет биологического смысла - существует
только для того, чтобы дать CNN что-то обучаемое и подтвердить, что
патчи, разбиения, обучение и построение графиков работают корректно. Не
представляйте её точность как научный результат.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.dataio import load_nifti
from src.train import run_hemisphere_experiment

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "subp019")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "smoke_test")


def make_synthetic_label(t1_brain, brain_mask, seed=0):
    """ТОЛЬКО плейсхолдер-метка (см. docstring модуля): не реальная
    concordance по CMRO2/BOLD, а просто что-то достаточно пространственно
    структурированное, чтобы проверить, что CNN способна выучить *хоть
    что-то* по патчам."""
    rng = np.random.default_rng(seed)
    brain_vals = t1_brain[brain_mask.astype(bool)]
    median = np.median(brain_vals)
    noise = rng.normal(0, brain_vals.std() * 0.5, size=t1_brain.shape)
    label = np.zeros(t1_brain.shape, dtype=np.float32)
    concordant = (t1_brain + noise) > median
    label[brain_mask.astype(bool)] = np.where(concordant[brain_mask.astype(bool)], 1.0, -1.0)
    return label


def main():
    t1_path = os.path.join(DATA_DIR, "subp019_descfmriprep_T1w_brain.nii.gz")
    mask_path = os.path.join(DATA_DIR, "subp019_descfmriprep_brain_mask.nii.gz")

    t1, affine, _ = load_nifti(t1_path)
    mask, mask_affine, _ = load_nifti(mask_path)

    if t1.shape != mask.shape:
        raise ValueError(
            f"форма T1 {t1.shape} != форме маски мозга {mask.shape}; "
            "передискретизируйте одно в другое перед запуском реального эксперимента."
        )

    label = make_synthetic_label(t1, mask)

    print("SMOKE-ТЕСТ: синтетическая метка, а не реальная карта concordant/discordant.")
    print(f"Форма T1: {t1.shape}, размеченных вокселей: {int((mask > 0).sum())}")

    results = run_hemisphere_experiment(
        t1_volume=t1,
        label_volume=label,
        brain_mask=mask,
        affine=affine,
        patch_size=9,
        axis_index=0,          # разбиение слева/справа
        max_voxels_per_side=800,
        epochs=8,
        out_dir=OUT_DIR,
        subject_name="subp019_SMOKETEST",
    )
    print("Точность разбиений smoke-теста (синтетические метки, не научный результат):", results)
    print(f"Диагностические графики записаны в {OUT_DIR}")


if __name__ == "__main__":
    main()
