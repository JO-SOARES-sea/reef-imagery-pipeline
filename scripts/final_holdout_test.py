#!/usr/bin/env python3
"""
scripts/final_holdout_test.py — Teste final pré-registado (docs/HOLDOUT_PREREGISTO.md). CORRER UMA SÓ VEZ.

Espacial: o modelo congelado (`--model`) é aplicado à zona FZW (Barreta–Culatra, imagens de desenvolvimento
2017–2024), e comparado com a verdade COSMO LTHBAIF1 (2018 vs 2020): rocha estável (relevo ≥ 0,3 m nos dois anos,
|Δz| ≤ 0,3 m) contra fundo liso (< 0,15 m nos dois), só dentro da caixa do holdout, negativos estratificados por
profundidade (0,5 m, ×5), IC por blocos de 500 m. Critério 2: AUC ≥ 0,75. Também: ganho vs mediana z, ECE.

Temporal: para cada zona de desenvolvimento, mapa de persistência só com passagens ≥ 2025-01-01 (gerado antes com
`build_persistence_map.py --date-range 2025-01-01/2026-10-31 --name <zona>_hold`). Uma deteção de desenvolvimento
(P ≥ 0,5) persiste se a mediana z de 2025–2026 tiver o mesmo sinal da de 2017–2024 e |z| ≥ 0,5. Compara-se com a
mesma taxa em fundo emparelhado por profundidade.
"""

from __future__ import annotations

import argparse
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
from src.constants import HOLDOUT_BBOX_WGS84  # noqa: E402

_s = importlib.util.spec_from_file_location("h4", PROJECT_ROOT / "scripts" / "run_h4_classifier.py")
h4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(h4)
_c = importlib.util.spec_from_file_location("ct", PROJECT_ROOT / "scripts" / "cosmo_truth_rasters.py")
ct = importlib.util.module_from_spec(_c)
_c.loader.exec_module(ct)
PDIR = h4.PDIR
OUT = PROJECT_ROOT / "outputs" / "eval"
BAIF = Path("/Volumes/4G_WD_BLACK/BACKUP/bulk_download_2026-10-02_02-35-48")


def baif_truth(z: dict) -> np.ndarray:
    fa, fb = sorted(BAIF.glob("LTHBAIF1_2018*_DEM.tif"))[0], sorted(BAIF.glob("LTHBAIF1_2020*_DEM.tif"))[0]
    ra, za, ta, crs, _ = ct.relief(fa)
    rb, zb, tb, _, _ = ct.relief(fb)
    out = {}
    for name, arr, tr, rs in (("ra", ra, ta, Resampling.max), ("rb", rb, tb, Resampling.max),
                              ("za", za, ta, Resampling.average), ("zb", zb, tb, Resampling.average)):
        dst = np.full(z["shape"], np.nan, np.float32)
        reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=z["T"], dst_crs="EPSG:32629", resampling=rs,
                  src_nodata=np.nan, dst_nodata=np.nan)
        out[name] = dst
    both = np.isfinite(out["ra"]) & np.isfinite(out["rb"]) & np.isfinite(out["za"]) & np.isfinite(out["zb"])
    cls = np.full(z["shape"], np.nan, np.float32)
    cls[both & (out["ra"] < 0.15) & (out["rb"] < 0.15)] = 0
    cls[both & (out["ra"] >= 0.3) & (out["rb"] >= 0.3) & (np.abs(out["za"] - out["zb"]) <= 0.3)] = 1
    return cls


def holdout_box_mask(z: dict) -> np.ndarray:
    W, S, E, N = HOLDOUT_BBOX_WGS84
    t = Transformer.from_crs(4326, 32629, always_xy=True)
    (x0, y0), (x1, y1) = t.transform(W, S), t.transform(E, N)
    rr, cc = np.indices(z["shape"])
    X = z["T"].c + (cc + .5) * 10
    Y = z["T"].f - (rr + .5) * 10
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def main() -> None:
    p = argparse.ArgumentParser(description="Teste final pré-registado (uma só vez)")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    bundle = joblib.load(a.model)
    m, feats = bundle["model"], bundle["features"]
    res = {"model": str(a.model), "features": feats}

    # ---- espacial: FZW (contém LTHBAIF1)
    z = h4.zone_features("fzw", bundle.get("with_h2", False))
    cls = baif_truth(z)
    box = holdout_box_mask(z)
    dep = z["depth"]
    ok = z["sea"].astype(bool) & box & (dep >= 3) & (dep <= 25) & np.isfinite(z["ALL_med_B03"])
    for f in feats:
        ok &= np.isfinite(z["F"][f])
    pi = np.flatnonzero(((cls == 1) & ok).ravel())
    ni = np.flatnonzero(((cls == 0) & ok).ravel())
    d = dep.ravel()
    ip, ineg = depth_stratified_sample(d[pi], d[ni], 0.5, 5, max_pos=5000, seed=a.seed)
    idx = np.r_[pi[ip], ni[ineg]]
    y = np.r_[np.ones(len(ip), int), np.zeros(len(ineg), int)]
    X = np.column_stack([z["F"][f].ravel()[idx] for f in feats])
    pr = m.predict_proba(X)[:, 1]
    rr, cc = np.unravel_index(idx, z["shape"])
    xs, ys = z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10
    base = -z["ALL_med_B03"].ravel()[idx]
    r = evaluate(y, pr, d[idx], xs, ys, 500.0, baseline=base, prob=pr, n_boot=1000, seed=a.seed).row()
    r["criterion_2_auc_ge_0.75"] = bool(r["auc"] >= 0.75)
    res["spatial_LTHBAIF1"] = r
    print("ESPACIAL LTHBAIF1:", json.dumps(r, ensure_ascii=False))

    # ---- temporal: zonas de desenvolvimento com mapas 2025–2026
    temporal = {}
    for zone in ("ancao", "se", "valado", "if"):
        hold = PDIR / f"{zone}_hold_maps.npz"
        if not hold.exists():
            temporal[zone] = "sem mapa 2025–2026"
            continue
        zd = h4.zone_features(zone, bundle.get("with_h2", False))
        zh = np.load(hold)
        okz = zd["sea"].astype(bool) & (zd["depth"] >= 3) & (zd["depth"] <= 25) & np.isfinite(zh["ALL_med_B03"])
        for f in feats:
            okz &= np.isfinite(zd["F"][f])
        okz &= ~holdout_box_mask(zd)
        P = np.full(zd["shape"], np.nan, np.float32)
        ii = np.flatnonzero(okz.ravel())
        P.ravel()[ii] = m.predict_proba(np.column_stack([zd["F"][f].ravel()[ii] for f in feats]))[:, 1]
        det = okz & (P >= 0.5)
        zdev, zhold = zd["ALL_med_B03"], zh["ALL_med_B03"]
        persist = (np.sign(zdev) == np.sign(zhold)) & (np.abs(zhold) >= 0.5)
        dd = zd["depth"].ravel()
        di, bi = np.flatnonzero(det.ravel()), np.flatnonzero((okz & ~det).ravel())
        ip2, ib2 = depth_stratified_sample(dd[di], dd[bi], 0.5, 5, max_pos=20000, seed=a.seed)
        rate_det = float(persist.ravel()[di[ip2]].mean()) if len(ip2) else None
        rate_bg = float(persist.ravel()[bi[ib2]].mean()) if len(ib2) else None
        temporal[zone] = {"n_detected_px": int(det.sum()), "persist_rate_detected": rate_det, "persist_rate_background": rate_bg,
                          "n_hold_passes": int(json.loads((PDIR / f"{zone}_hold_maps.json").read_text())["n_passes"])}
        print(f"TEMPORAL {zone}: {temporal[zone]}")
    res["temporal_2025_2026"] = temporal
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FINAL_HOLDOUT.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
