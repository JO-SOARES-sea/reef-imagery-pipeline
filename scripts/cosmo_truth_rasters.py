#!/usr/bin/env python3
"""
scripts/cosmo_truth_rasters.py — Verdade COSMO a 10 m para o conjunto de teste T3 (docs/HOLDOUT_PREREGISTO.md).

Para um setor COSMO (dois anos de DEM multifeixe 0,30 m, cotas ZH), calcula a 0,9 m o relevo local
(DEM − média móvel de 25 m) e agrega-o por máximo a uma grelha UTM 29N de 10 m alinhada a múltiplos de 10 m
(as grelhas de satélite). Classes por célula:
  rocha estável = relevo ≥ 0,3 m nos dois anos e |Δcota| ≤ 0,3 m (critério DL-025 do UV);
  fundo liso     = relevo < 0,15 m nos dois anos;
  resto           = sem rótulo (nodata).
O setor do holdout v2 (LTHNOCR1, docs/HOLDOUT_PREREGISTO_v2.md) é recusado.

Saída: data/processed/cosmo_truth/<setor>_truth10m.tif (bandas: rel_a, rel_b, z_a, z_b, classe 1/0/255).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "data" / "processed" / "cosmo_truth"
B = Path("/Volumes/4G_WD_BLACK/BACKUP")
SECTORS = {
    "LTHQUGA1": (B / "bulk_download_2026-09-28_04-23-11/LTHQUGA1_20181003_20181013_DEM.tif",
                 B / "bulk_download_2026-09-28_04-23-11/LTHQUGA1_20250612_20250616_DEM.tif"),
    "LTHFA1": (B / "bulk_download_2026-10-20_22-57-24/LTHFA1_20181010_20181014_DEM.tif",
               B / "bulk_download_2026-10-20_22-57-24/LTHFA1_20250617_DEM.tif"),
    "LTHPEAL1": (B / "bulk_download_2026-09-28_04-21-05/LTHPEAL1_20181020_20181024_DEM.tif",
                 B / "bulk_download_2026-09-28_04-21-05/LTHPEAL1_20201113_20201117_DEM.tif"),
}
# Holdout v1 (LTHBAIF1) foi gasto em 2026-10-26 e passou a dado de treino; o holdout v2 é o LTHNOCR1
SECTORS["LTHBAIF1"] = (B / "bulk_download_2026-10-02_02-35-48/LTHBAIF1_20181016_20181022_DEM.tif", B / "bulk_download_2026-10-02_02-35-48/LTHBAIF1_20201023_20201029_DEM.tif")
HOLDOUT_SECTORS = {"LTHNOCR1"}


def relief(path: Path, dec: int = 3):
    with rasterio.open(path) as src:
        dem = src.read(1, out_shape=(src.height // dec, src.width // dec), resampling=Resampling.average).astype(np.float32)
        dem[dem <= -9000] = np.nan
        t = src.transform * src.transform.scale(src.width / dem.shape[1], src.height / dem.shape[0])
        crs, bounds = src.crs, src.bounds
    ok = np.isfinite(dem)
    w = int(round(25 / abs(t.a)))
    num = ndi.uniform_filter(np.where(ok, dem, 0), w)
    den = ndi.uniform_filter(ok.astype(np.float32), w)
    low = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
    return np.where(ok & np.isfinite(low), dem - low, np.nan).astype(np.float32), dem, t, crs, bounds


def main() -> None:
    p = argparse.ArgumentParser(description="Verdade COSMO a 10 m (T3)")
    p.add_argument("sector", choices=sorted(SECTORS) + sorted(HOLDOUT_SECTORS))
    a = p.parse_args()
    if a.sector in HOLDOUT_SECTORS:
        raise SystemExit(f"{a.sector} é o holdout pré-registado: não é processado no desenvolvimento.")
    fa, fb = SECTORS[a.sector]
    ra, za, ta, crs, bounds = relief(fa)
    rb, zb, tb, _, _ = relief(fb)
    l, b_, r, t = transform_bounds(crs, "EPSG:32629", *bounds)
    x0, y1 = np.floor(l / 10) * 10, np.ceil(t / 10) * 10
    W, H = int(np.ceil((r - x0) / 10)), int(np.ceil((y1 - b_) / 10))
    T = from_origin(x0, y1, 10, 10)
    out = {}
    for name, arr, tr, rs in (("rel_a", ra, ta, Resampling.max), ("rel_b", rb, tb, Resampling.max),
                              ("z_a", za, ta, Resampling.average), ("z_b", zb, tb, Resampling.average)):
        dst = np.full((H, W), np.nan, np.float32)
        reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=T, dst_crs="EPSG:32629", resampling=rs,
                  src_nodata=np.nan, dst_nodata=np.nan)
        out[name] = dst
    both = np.isfinite(out["rel_a"]) & np.isfinite(out["rel_b"]) & np.isfinite(out["z_a"]) & np.isfinite(out["z_b"])
    rock = both & (out["rel_a"] >= 0.3) & (out["rel_b"] >= 0.3) & (np.abs(out["z_a"] - out["z_b"]) <= 0.3)
    smooth = both & (out["rel_a"] < 0.15) & (out["rel_b"] < 0.15)
    cls = np.full((H, W), 255, np.float32); cls[smooth] = 0; cls[rock] = 1
    OUT.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "dtype": "float32", "count": 5, "crs": "EPSG:32629", "transform": T, "width": W, "height": H,
            "nodata": np.nan, "compress": "deflate"}
    with rasterio.open(OUT / f"{a.sector}_truth10m.tif", "w", **prof) as dst:
        for i, k in enumerate(("rel_a", "rel_b", "z_a", "z_b"), 1):
            dst.write(out[k], i); dst.set_band_description(i, k)
        dst.write(cls, 5); dst.set_band_description(5, "classe: 1 rocha estável, 0 liso, 255 sem rótulo")
        dst.update_tags(years=f"{fa.name} | {fb.name}")
    print(f"{a.sector}: rocha estável {int(rock.sum())} células, liso {int(smooth.sum())}, grelha {W}x{H}")


if __name__ == "__main__":
    main()
