#!/usr/bin/env python3
"""
scripts/build_h2_albedo.py — Hipótese H2: índice de fundo invariante à profundidade (Lyzenga 1978/1981) por passagem.

Para cada passagem limpa já usada na pilha de desenvolvimento da zona (`<zona>_stack_passes.csv`):
  X_i = ln(R_i − R∞_i), com R∞ = mediana da banda em água > 25 m;
  ki/kj pelo método da covariância (Lyzenga 1981) em água de 3–15 m da própria passagem:
      a = (var Xi − var Xj) / (2 cov(Xi, Xj)),  ki/kj = a + sqrt(a² + 1);
  DII = X_i − (ki/kj) X_j, com i = B02, j = B03.
Depois remove-se a tendência residual com a profundidade (mediana por classe de 1 m) e normaliza-se por desvio
robusto. A mediana sobre as passagens dá `<zona>_h2.npz` (dii_anom). Escuro = albedo baixo (rocha com algas) à
mesma profundidade, positivo = areia clara.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer as pc
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDIR = PROJECT_ROOT / "data" / "processed" / "persistence"
_spec = importlib.util.spec_from_file_location("bcw", PROJECT_ROOT / "scripts" / "build_clear_water_composite.py")
bcw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcw)


def dii_pass(item, grid, sea, depth):
    try:
        off = bcw._boa_offset_dn(item)
        rd = lambda b, rs=Resampling.bilinear: bcw._read_on_grid(item.assets[b].href, grid, rs)
        b2, b3, b8 = (bcw._to_reflectance(rd(b), off) for b in ("B02", "B03", "B08"))
        scl = rd("SCL", Resampling.nearest)
    except Exception:
        return None
    valid = sea & (b3 > 0) & ~np.isin(scl, bcw.SCL_INVALID) & (b8 < 0.03)
    deep = valid & (depth > 25)
    shallow = valid & (depth >= 3) & (depth <= 15)
    if deep.sum() < 500 or shallow.sum() < 2000:
        return None
    r2, r3 = np.median(b2[deep]), np.median(b3[deep])
    with np.errstate(all="ignore"):
        X2, X3 = np.log(b2 - r2), np.log(b3 - r3)
    m = shallow & np.isfinite(X2) & np.isfinite(X3)
    if m.sum() < 2000:
        return None
    c = np.cov(X2[m], X3[m])
    a = (c[0, 0] - c[1, 1]) / (2 * c[0, 1])
    kk = a + np.sqrt(a * a + 1)
    dii = np.where(valid & np.isfinite(X2) & np.isfinite(X3), X2 - kk * X3, np.nan)
    # remove a tendência residual com a profundidade (mediana por classe de 1 m) e normaliza
    out = np.full(dii.shape, np.nan, np.float32)
    bins = np.floor(depth).astype(int)
    ok = np.isfinite(dii) & (depth >= 1) & (depth <= 30)
    for b in np.unique(bins[ok]):
        sel = ok & (bins == b)
        if sel.sum() > 50:
            out[sel] = dii[sel] - np.median(dii[sel])
    s = 1.4826 * np.nanmedian(np.abs(out[ok]))
    return (out / max(s, 1e-9)).astype(np.float16), float(kk)


def main() -> None:
    p = argparse.ArgumentParser(description="H2: índice de Lyzenga por passagem")
    p.add_argument("--zone", required=True, help="ex.: ancao_dev")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    meta = json.loads((PDIR / f"{a.zone}_maps.json").read_text())
    z = np.load(PDIR / f"{a.zone}_maps.npz")
    depth, sea = z["depth"], z["sea"].astype(bool)
    grid = {"crs": "EPSG:32629", "transform": from_origin(meta["x0"], meta["y1"], 10, 10), "width": depth.shape[1], "height": depth.shape[0]}
    import rasterio
    grid["crs"] = rasterio.crs.CRS.from_epsg(32629)
    ids = set(pd.read_csv(PDIR / f"{a.zone}_stack_passes.csv").id)
    client = Client.open(bcw.STAC_URL, modifier=pc.sign_inplace)
    items = [it for it in client.search(collections=["sentinel-2-l2a"], ids=sorted(ids)).items()]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        res = [r for r in pool.map(lambda it: dii_pass(it, grid, sea, depth), items) if r is not None]
    stack = np.stack([r[0] for r in res]).astype(np.float32)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0).astype(np.float32)
    np.savez_compressed(PDIR / f"{a.zone}_h2.npz", dii_anom=med)
    print(json.dumps({"zone": a.zone, "passes": len(res), "of": len(items), "k_ratio_median": float(np.median([r[1] for r in res]))}))


if __name__ == "__main__":
    main()
