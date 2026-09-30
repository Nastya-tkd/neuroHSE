#!/usr/bin/env python3
"""Готовит реальные МРТ из OpenNeuro ds007045 (Filimonova et al., CC0) для веб-просмотрщика СТОП-РАК.

Вход  (--raw): папка с файлами  sub-XXX_{t1,t1ce,t2,flair}.nii.gz  и  sub-XXX_mask.nii.gz
               (MNI152, skull-stripped, N4 — derivatives/processing и derivatives/segmentation датасета)
Выход (--out): stoprak/data/<sub>/{t1,t1ce,t2,flair,seg,depth_all,depth_enh}.nii.gz (uint8, RAS+, обрезка по мозгу)
               и case.json с метаданными.

Метки сегментации датасета (BraTS): 1 — некроз/неусиливающееся ядро, 2 — перифокальный отек, 3 — усиливающаяся опухоль.
"""
import argparse, json, csv, os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

MODS = ["t1", "t1ce", "t2", "flair"]

def load_ras(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj).astype(np.float32), img.affine

def to_u8(vol, brain):
    v = vol[brain]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.6)
    out = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1) * 254 + 1
    out[~brain] = 0
    return out.astype(np.uint8)

def save(arr, affine, path):
    img = nib.Nifti1Image(arr, affine)
    img.header.set_data_dtype(arr.dtype)
    nib.save(img, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--participants", required=True); ap.add_argument("subs", nargs="+")
    a = ap.parse_args()
    parts = {r["participant_id"]: r for r in csv.DictReader(open(a.participants), delimiter="\t")}
    for sub in a.subs:
        vols = {}
        for m in MODS:
            vols[m], aff = load_ras(f"{a.raw}/{sub}_{m}.nii.gz")
        seg = np.asarray(nib.as_closest_canonical(nib.load(f"{a.raw}/{sub}_mask.nii.gz")).dataobj).astype(np.uint8)
        brain = (vols["t1"] > 0) | (vols["t2"] > 0) | (vols["flair"] > 0)
        brain = ndi.binary_opening(brain, iterations=1)
        idx = np.argwhere(brain | (seg > 0)); lo = np.maximum(idx.min(0) - 3, 0); hi = idx.max(0) + 4
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        aff2 = aff.copy(); aff2[:3, 3] = aff[:3, :3] @ lo + aff[:3, 3]
        brain = brain[sl]; seg = seg[sl]
        d = os.path.join(a.out, sub); os.makedirs(d, exist_ok=True)
        for m in MODS:
            save(to_u8(vols[m][sl], brain), aff2, f"{d}/{m}.nii.gz")
        save(seg, aff2, f"{d}/seg.nii.gz")
        whole = seg > 0; enh = seg == 3
        zooms = np.abs(np.diag(aff)[:3])
        de = np.clip(ndi.distance_transform_edt(whole, sampling=zooms) * 4, 0, 255).astype(np.uint8)
        dh = np.clip(ndi.distance_transform_edt(enh, sampling=zooms) * 4, 0, 255).astype(np.uint8)
        save(de, aff2, f"{d}/depth_all.nii.gz"); save(dh, aff2, f"{d}/depth_enh.nii.gz")
        p = parts.get(sub, {})
        meta = dict(id=sub, shape=list(seg.shape), spacing=[float(x) for x in zooms], affine=aff2.tolist(),
                    age=int(p.get("age", 0) or 0), mgmt=int(p.get("MGMT", -1) or 0), field=p.get("field_strength", "").replace(",", "."),
                    scanner=p.get("scanner_model", ""), source="OpenNeuro ds007045 (CC0)")
        json.dump(meta, open(f"{d}/case.json", "w"), ensure_ascii=False, indent=1)
        print(sub, seg.shape, {int(k): int((seg == k).sum()) for k in (1, 2, 3)})

if __name__ == "__main__":
    main()
