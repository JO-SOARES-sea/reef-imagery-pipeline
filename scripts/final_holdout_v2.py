#!/usr/bin/env python3
"""
scripts/final_holdout_v2.py — Teste final do pré-registo v2 (docs/HOLDOUT_PREREGISTO_v2.md). CORRER UMA SÓ VEZ.

Critério 1–2: o modelo gravado por run_h4b_classifier.py é aplicado à zona VALADO (imagens de desenvolvimento,
órbita R037) e comparado, só dentro da caixa do holdout v2, com a verdade COSMO LTHNOCR1 (2018 vs 2020): rocha estável
vs liso, estratificação por profundidade (0,5 m, ×5), blocos 500 m. AUC ≥ 0,75 e ECE ≤ 0,10.
Critério 3: Greta (zona IF, nunca treinada como positivo): fração da crista acima do limiar FPR 5 % do fundo da IF a
13–17 m (a > 300 m da crista).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.reef_eval import depth_stratified_sample, evaluate  # noqa: E402

_b = importlib.util.spec_from_file_location("h4b", PROJECT_ROOT / "scripts" / "run_h4b_classifier.py")
h4b = importlib.util.module_from_spec(_b)
_b.loader.exec_module(h4b)
_c = importlib.util.spec_from_file_location("ct", PROJECT_ROOT / "scripts" / "cosmo_truth_rasters.py")
ct = importlib.util.module_from_spec(_c)
_c.loader.exec_module(ct)
NOCR = "/Volumes/4G_WD_BLACK/BACKUP/bulk_download_2026-09-30_22-06-05/"


def nocr_truth(z: dict) -> np.ndarray:
    fa, fb = sorted(glob.glob(NOCR + "LTHNOCR1_2018*_DEM.tif"))[0], sorted(glob.glob(NOCR + "LTHNOCR1_2020*_DEM.tif"))[0]
    ra, za, ta, crs, _ = ct.relief(Path(fa))
    rb, zb, tb, _, _ = ct.relief(Path(fb))
    o = {}
    for k, arr, tr, rs in (("ra", ra, ta, Resampling.max), ("rb", rb, tb, Resampling.max), ("za", za, ta, Resampling.average), ("zb", zb, tb, Resampling.average)):
        dst = np.full(z["shape"], np.nan, np.float32)
        reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=z["T"], dst_crs="EPSG:32629", resampling=rs, src_nodata=np.nan, dst_nodata=np.nan)
        o[k] = dst
    both = np.isfinite(o["ra"]) & np.isfinite(o["rb"]) & np.isfinite(o["za"]) & np.isfinite(o["zb"])
    cls = np.full(z["shape"], np.nan, np.float32)
    cls[both & (o["ra"] < 0.15) & (o["rb"] < 0.15)] = 0
    cls[both & (o["ra"] >= 0.3) & (o["rb"] >= 0.3) & (np.abs(o["za"] - o["zb"]) <= 0.3)] = 1
    return cls


def predict_zone(m, feats, z) -> np.ndarray:
    ok = z["sea"].astype(bool) & (z["depth"] >= 3) & (z["depth"] <= 25)
    for f in feats:
        ok &= np.isfinite(z["F"][f])
    P = np.full(z["shape"], np.nan, np.float32)
    ii = np.flatnonzero(ok.ravel())
    P.ravel()[ii] = m.predict_proba(np.column_stack([z["F"][f].ravel()[ii] for f in feats]))[:, 1]
    return P


def main() -> None:
    ap = argparse.ArgumentParser(description="Teste final do pré-registo v2 (uma só vez)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    b = joblib.load(a.model)
    m, feats = b["model"], b["features"]
    res = {"model": str(a.model)}

    z = h4b.zone_features("valado")
    cls = nocr_truth(z)
    box = h4b.holdout2_mask(z)
    dep = z["depth"]
    ok = z["sea"].astype(bool) & box & (dep >= 3) & (dep <= 25)
    for f in feats:
        ok &= np.isfinite(z["F"][f])
    pi, ni = np.flatnonzero(((cls == 1) & ok).ravel()), np.flatnonzero(((cls == 0) & ok).ravel())
    d = dep.ravel()
    ip, ineg = depth_stratified_sample(d[pi], d[ni], 0.5, 5, max_pos=5000, seed=a.seed)
    idx = np.r_[pi[ip], ni[ineg]]
    y = np.r_[np.ones(len(ip), int), np.zeros(len(ineg), int)]
    pr = m.predict_proba(np.column_stack([z["F"][f].ravel()[idx] for f in feats]))[:, 1]
    rr, cc = np.unravel_index(idx, z["shape"])
    r = evaluate(y, pr, d[idx], z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10, 500.0, baseline=-z["F"]["med"].ravel()[idx],
                 prob=pr, n_boot=1000, seed=a.seed).row()
    r["crit1_auc_ge_0.75"] = bool(r["auc"] >= 0.75)
    r["crit2_ece_le_0.10"] = bool(r["ece"] <= 0.10)
    res["holdout_v2_LTHNOCR1"] = r
    print("HOLDOUT v2 LTHNOCR1:", json.dumps(r, ensure_ascii=False))

    zif = h4b.zone_features("if")
    P = predict_zone(m, feats, zif)
    t = Transformer.from_crs(4326, 32629, always_xy=True)
    se_, nw = np.array(t.transform(-7.978251, 36.972730)), np.array(t.transform(-7.996629, 36.984876))
    L = np.linalg.norm(nw - se_); u = (nw - se_) / L; nrm = np.array([-u[1], u[0]])
    crest = []
    for s_ in np.arange(0, L + 1, 30):
        vals = []
        for o in np.arange(-50, 51, 10):
            x, yv = se_ + u * s_ + nrm * o
            r_, c_ = int((zif["T"].f - yv) // 10), int((x - zif["T"].c) // 10)
            if 0 <= r_ < P.shape[0] and 0 <= c_ < P.shape[1] and np.isfinite(P[r_, c_]):
                vals.append(P[r_, c_])
        crest.append(max(vals) if vals else np.nan)
    rr, cc = np.indices(P.shape)
    X_, Y_ = zif["T"].c + (cc + .5) * 10, zif["T"].f - (rr + .5) * 10
    dl = np.abs((X_ - se_[0]) * nrm[0] + (Y_ - se_[1]) * nrm[1]); al = (X_ - se_[0]) * u[0] + (Y_ - se_[1]) * u[1]
    far = ~((dl < 300) & (al > -500) & (al < L + 500))
    bg = P[np.isfinite(P) & far & (zif["depth"] >= 13) & (zif["depth"] <= 17) & ~zif["excl"]]
    thr = float(np.nanquantile(bg, 0.95))
    frac = float(np.nanmean(np.array(crest) > thr))
    res["greta"] = {"threshold_fpr5": thr, "crest_fraction_above": frac, "crit3_ge_0.5": bool(frac >= 0.5)}
    print("GRETA:", res["greta"])
    (PROJECT_ROOT / "outputs" / "eval" / "FINAL_HOLDOUT_v2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
