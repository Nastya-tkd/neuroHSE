"""GL261 Segmentation Assistant -- local lab application.

Run with:  streamlit run app.py
(or double-click run_app.bat on Windows)

Doctors/lab staff upload a contrast-enhanced T1-weighted MRI slice
(PNG/JPG or DICOM) and get back the three-class segmentation described
in the GL261-MRI-Segmentation paper: enhancing-tumor-like, perifocal
edema-like, and necrosis-like regions, with per-class area and a
downloadable report. Runs fully offline, CPU-only.
"""

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from scripts.model_def import SmallUNet
from scripts.dataset import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "unet_gl261.pt")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

CLASS_LABELS_RU = {"background": "Фон", "tumor": "Опухоль", "edema": "Отёк", "necrosis": "Некроз"}

st.set_page_config(page_title="GL261 Segmentation Assistant", layout="wide", page_icon="🩻")


# ---------------------------------------------------------------- model ----
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = SmallUNet(in_ch=meta["in_channels"], num_classes=meta["num_classes"],
                       base=meta.get("base_channels", 24))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def read_uploaded_image(uploaded_file):
    """Returns (grayscale PIL image, pixel_spacing_mm or None, source label)."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".dcm") or name.endswith(".ima"):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        arr = ds.pixel_array.astype(np.float32)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
        spacing = None
        if "PixelSpacing" in ds:
            spacing = float(ds.PixelSpacing[0])
        return img, spacing, "DICOM"
    else:
        img = Image.open(io.BytesIO(data)).convert("L")
        return img, None, "image"


def run_inference(model, meta, gray_img: Image.Image):
    size = meta["input_size"]
    resized = gray_img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    pred = np.argmax(probs, axis=0).astype(np.uint8)
    confidence = probs.max(axis=0)
    return pred, confidence, probs


def make_overlay(gray_img: Image.Image, pred: np.ndarray, alpha=0.5):
    size = pred.shape[0]
    base = gray_img.resize((size, size), Image.BILINEAR).convert("RGB")
    base_arr = np.asarray(base).astype(np.float32)
    overlay_arr = base_arr.copy()
    color_mask = np.zeros_like(base_arr)
    for class_id, color in enumerate(CLASS_COLORS):
        if class_id == 0:
            continue
        color_mask[pred == class_id] = color
    mask_any = pred != 0
    overlay_arr[mask_any] = (1 - alpha) * base_arr[mask_any] + alpha * color_mask[mask_any]
    return Image.fromarray(overlay_arr.astype(np.uint8))


def class_stats(pred: np.ndarray, spacing_mm):
    total_px = pred.size
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        if class_id == 0:
            continue
        px = int(np.sum(pred == class_id))
        pct = 100.0 * px / total_px
        mm2 = px * (spacing_mm ** 2) if spacing_mm else None
        rows.append(dict(class_id=class_id, name=name, px=px, pct=pct, mm2=mm2))
    return rows


# --------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.title("🩻 GL261 Segmentation Assistant")
    st.caption("Локальный, офлайн, CPU. По методике GL261-MRI-Segmentation.")

    model, meta = load_model()
    if model is None:
        st.error("Модель не найдена. Ожидается файл model/unet_gl261.pt "
                 "(положите его рядом с приложением после обучения).")
    else:
        st.success(f"Модель загружена ({sum(p.numel() for p in model.parameters()):,} параметров)")
        with st.expander("О модели"):
            st.write(f"Вход: {meta['input_size']}×{meta['input_size']} px, оттенки серого")
            st.write(f"Обучено на {meta['train_pairs']} срезах, валидация на {meta['val_pairs']} "
                     f"срезах ({meta['val_pairs']} животных вне обучающей выборки)")
            st.write(f"Средний Dice (валидация, без фона): **{meta['best_mean_fg_dice']:.2f}**")

    st.divider()
    st.subheader("Параметры")
    manual_spacing = st.number_input(
        "Размер пикселя (мм), если не DICOM", min_value=0.0, max_value=5.0,
        value=0.0, step=0.01, format="%.3f",
        help="Для PNG/JPG приложение не знает физический масштаб снимка. "
             "Укажите PixelSpacing из протокола сканирования, чтобы получить площадь в мм² "
             "(0 = показывать только пиксели и проценты).")
    conf_threshold = st.slider(
        "Порог уверенности", min_value=0.0, max_value=0.95, value=0.0, step=0.05,
        help="Пиксели с максимальной вероятностью класса ниже порога относятся к фону "
             "(снижает ложные срабатывания ценой чувствительности; см. §4.3 статьи).")

    st.divider()
    st.caption("⚠️ Исследовательский прототип. Не для клинической диагностики — "
               "регионы определены по МРТ-паттерну, а не по гистологии.")


# ----------------------------------------------------------------- main ----
st.header("Анализ среза")

uploaded = st.file_uploader(
    "Загрузите T1-срез с контрастированием (PNG, JPG или DICOM)",
    type=["png", "jpg", "jpeg", "dcm", "ima"],
)

if uploaded is not None and model is not None:
    gray_img, dicom_spacing, source = read_uploaded_image(uploaded)
    spacing_mm = dicom_spacing if dicom_spacing else (manual_spacing or None)

    if st.button("Запустить анализ", type="primary"):
        with st.spinner("Выполняется сегментация..."):
            pred, confidence, probs = run_inference(model, meta, gray_img)
            if conf_threshold > 0:
                pred = np.where(confidence < conf_threshold, 0, pred).astype(np.uint8)
            overlay = make_overlay(gray_img, pred)
            stats = class_stats(pred, spacing_mm)

        st.session_state["result"] = dict(
            filename=uploaded.name, gray_img=gray_img, overlay=overlay,
            pred=pred, stats=stats, spacing_mm=spacing_mm, source=source,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

if "result" in st.session_state:
    r = st.session_state["result"]

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Исходный срез**")
        st.image(r["gray_img"], use_container_width=True)
    with col_img2:
        st.markdown("**Результат сегментации**")
        st.image(r["overlay"], use_container_width=True)
        legend_cols = st.columns(3)
        for col, name in zip(legend_cols, ["tumor", "edema", "necrosis"]):
            color = CLASS_COLORS[CLASS_NAMES.index(name)]
            col.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background:rgb{color};border-radius:2px;margin-right:6px"></span>'
                f'{CLASS_LABELS_RU[name]}',
                unsafe_allow_html=True,
            )

    st.subheader("Количественная оценка")
    unit_note = "мм²" if r["spacing_mm"] else "% площади среза"
    stat_cols = st.columns(len(r["stats"]) + 1)
    total_px = 0
    total_mm2 = 0.0
    for col, row in zip(stat_cols, r["stats"]):
        total_px += row["px"]
        value = f"{row['mm2']:.1f} мм²" if row["mm2"] is not None else f"{row['pct']:.1f}%"
        if row["mm2"] is not None:
            total_mm2 += row["mm2"]
        col.metric(CLASS_LABELS_RU[row["name"]], value,
                   f"{row['px']:,} px".replace(",", " "), delta_color="off")
    total_value = f"{total_mm2:.1f} мм²" if r["spacing_mm"] else f"{100*total_px/r['pred'].size:.1f}%"
    stat_cols[-1].metric("Суммарно (очаг)", total_value,
                         f"{total_px:,} px".replace(",", " "), delta_color="off")

    if not r["spacing_mm"]:
        st.info("Площадь показана в пикселях/процентах. Укажите размер пикселя в мм "
                "в боковой панели (или загрузите DICOM с тегом PixelSpacing), чтобы получить мм².")

    df = pd.DataFrame([
        {
            "Регион": CLASS_LABELS_RU[row["name"]],
            "Площадь, px": row["px"],
            "Площадь, %": round(row["pct"], 2),
            "Площадь, мм²": round(row["mm2"], 2) if row["mm2"] is not None else None,
        }
        for row in r["stats"]
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Ограничения")
    st.markdown(
        "- Регионы определены по виду на МРТ и экспертной разметке; без совмещения с "
        "гистологией их нельзя считать прямым измерением некроза, отёка или инвазии.\n"
        "- Модель обучена на 15 мышах — это исследовательский, а не валидированный "
        "клинический инструмент.\n"
        "- Результат относится к загруженному изображению как есть (без переоценки "
        "качества скана, артефактов, ориентации среза)."
    )

    report_txt = (
        f"GL261 Segmentation Assistant — отчёт\n"
        f"Файл: {r['filename']}\n"
        f"Дата анализа: {r['timestamp']}\n"
        f"Источник: {r['source']}\n"
        f"Масштаб: {r['spacing_mm']} мм/px\n\n"
        + df.to_string(index=False)
        + "\n\nИсследовательский прототип. Не для клинической диагностики.\n"
    )
    st.download_button(
        "Скачать отчёт (.txt)", report_txt,
        file_name=f"report_{os.path.splitext(r['filename'])[0]}.txt",
    )

elif model is not None:
    st.info("Загрузите снимок и нажмите «Запустить анализ», чтобы увидеть результат.")
