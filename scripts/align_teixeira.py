#!/usr/bin/env python3
"""
scripts/align_teixeira.py — Estima o deslocamento dos polígonos de rocha de Teixeira (Fig. 20) pela imagem do Ancão.

Ajuste: pilha dev ANCAO (anomalia de escurecimento sem treino, sem componente de profundidade).
Verificações independentes do ajuste:
  1. Quarteira: marcas GPS fora da cena do Ancão — a distância à rocha tem de baixar mais do que num controlo aleatório.
  2. COSMO LTHQUGA1 (LiDAR da APA): concordância entre a rocha de Teixeira e a rocha estável, antes e depois.

Uso:
    python scripts/align_teixeira.py              # só mede e escreve outputs/eval/teixeira_alignment.json
    python scripts/align_teixeira.py --write      # escreve também o GeoJSON alinhado (o original fica intacto)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.teixeira_alignment import depth_detrend, estimate_shift, pixel_shift_to_metres, shift_polygons  # noqa: E402
from src.constants import HOLDOUT2_BBOX_WGS84, HOLDOUT_BBOX_WGS84  # noqa: E402

PDIR = PROJECT_ROOT / "data" / "processed" / "persistence"
TRUTH = PROJECT_ROOT / "data" / "processed" / "cosmo_truth"
TEIXEIRA = PROJECT_ROOT / "data" / "external" / "eia3587" / "fig20_teixeira2025_layers.geojson"
ALIGNED = TEIXEIRA.with_name("fig20_teixeira2025_layers_alinhado.geojson")
GPS = PROJECT_ROOT / "data" / "vector" / "GPS_10_2026.csv"
OUT = PROJECT_ROOT / "outputs" / "eval" / "teixeira_alignment.json"
CRS = 32629
DEPTH_RANGE = (3.0, 20.0)
MAX_SHIFT_PX = 30
GPS_CATEGORIES = ("fishing_mark", "rock", "waypoint", "general")  # naufrágios fora


def _rasterize(geoms, shape, T) -> np.ndarray:
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=T).astype(bool)


def _holdout_mask(shape, T) -> np.ndarray:
    boxes = []
    for W, S, E, N in (HOLDOUT_BBOX_WGS84, HOLDOUT2_BBOX_WGS84):
        hb = gpd.GeoSeries.from_xy([W, E], [S, N], crs=4326).to_crs(CRS)
        boxes.append(box(hb.x.min(), hb.y.min(), hb.x.max(), hb.y.max()))
    return _rasterize(boxes, shape, T)


def _dist_to_rock(points: gpd.GeoSeries, rock_union) -> np.ndarray:
    return np.asarray(points.distance(rock_union))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="escreve o GeoJSON alinhado")
    args = ap.parse_args()

    meta = json.loads((PDIR / "ancao_dev_maps.json").read_text())
    z = dict(np.load(PDIR / "ancao_dev_maps.npz"))
    shape = z["depth"].shape
    T = from_origin(meta["x0"], meta["y1"], 10, 10)

    lith = gpd.read_file(TEIXEIRA).set_crs(4326, allow_override=True).to_crs(CRS)
    rock_g = lith[lith.layer.str.startswith("rocha")]
    rock_mask = ndi.binary_erosion(_rasterize(rock_g.geometry, shape, T), iterations=2)

    valid = z["sea"] & (z["depth"] >= DEPTH_RANGE[0]) & (z["depth"] <= DEPTH_RANGE[1]) & ~_holdout_mask(shape, T)
    resid = depth_detrend(-z["ALL_med_B03"], z["depth"], valid)

    est = estimate_shift(rock_mask, resid, valid, max_shift_px=MAX_SHIFT_PX)
    dx, dy = pixel_shift_to_metres(est["dr"], est["dc"], 10.0)

    # Estabilidade: o mesmo ajuste em duas metades da cena (oeste/este) tem de concordar
    half = {}
    mid = shape[1] // 2
    for name, sl in (("oeste", slice(0, mid)), ("este", slice(mid, shape[1]))):
        m = np.zeros(shape, bool)
        m[:, sl] = True
        e = estimate_shift(rock_mask & m, resid, valid & m, max_shift_px=MAX_SHIFT_PX)
        half[name] = {"dr": e["dr"], "dc": e["dc"], "gain": e["score_at_peak"] - e["score_at_zero"]}

    shifted = shift_polygons(list(rock_g.geometry), dx, dy)
    rock_before = rock_g.geometry.union_all()
    rock_after = gpd.GeoSeries(shifted, crs=CRS).union_all()

    # Verificação 1 — Quarteira: marcas GPS fora da cena do Ancão
    W, S, E, N = meta["bbox_wgs84"]
    gps = pd.read_csv(GPS)
    gps = gps[gps.category.isin(GPS_CATEGORIES)]
    g = gpd.GeoDataFrame(gps, geometry=gpd.points_from_xy(gps.longitude, gps.latitude), crs=4326)
    outside = ~((g.longitude >= W) & (g.longitude <= E) & (g.latitude >= S) & (g.latitude <= N))
    hull = lith.union_all().convex_hull
    g = g[outside].to_crs(CRS)
    g = g[g.geometry.within(hull)]
    d0, d1 = _dist_to_rock(g.geometry, rock_before), _dist_to_rock(g.geometry, rock_after)
    rng = np.random.default_rng(0)
    minx, miny, maxx, maxy = hull.bounds
    ctrl = []
    while len(ctrl) < 3000:
        p = gpd.points_from_xy([rng.uniform(minx, maxx)], [rng.uniform(miny, maxy)])[0]
        if hull.contains(p):
            ctrl.append(p)
    cs = gpd.GeoSeries(ctrl, crs=CRS)
    c0, c1 = _dist_to_rock(cs, rock_before), _dist_to_rock(cs, rock_after)

    def summ(a: np.ndarray) -> dict:
        return {"n": int(len(a)), "em_rocha_pct": round(float((a == 0).mean() * 100), 1),
                "ate_200m_pct": round(float((a <= 200).mean() * 100), 1), "mediana_m": round(float(np.median(a)), 0)}

    quarteira = {"gps_antes": summ(d0), "gps_depois": summ(d1), "controlo_antes": summ(c0), "controlo_depois": summ(c1)}

    # Verificação 2 — COSMO LTHQUGA1 (independente do satélite)
    with rasterio.open(TRUTH / "LTHQUGA1_truth10m.tif") as src:
        cls = np.full(shape, np.nan, np.float32)
        reproject(src.read(5), cls, src_transform=src.transform, src_crs=src.crs, dst_transform=T, dst_crs=f"EPSG:{CRS}",
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    foot = np.isfinite(cls)
    cosmo_rock = cls == 1

    def agreement(geoms) -> dict:
        tx = _rasterize(geoms, shape, T) & foot
        inter = int((tx & cosmo_rock).sum())
        return {"pix_teixeira": int(tx.sum()), "pix_cosmo_rocha": int(cosmo_rock.sum()),
                "precisao": round(inter / max(1, int(tx.sum())), 3), "recall": round(inter / max(1, int(cosmo_rock.sum())), 3),
                "iou": round(inter / max(1, int((tx | cosmo_rock).sum())), 3)}

    cosmo = {"antes": agreement(list(rock_g.geometry)), "depois": agreement(shifted)}

    result = {
        "ajuste_ancao_dev": {"dr_px": est["dr"], "dc_px": est["dc"], "dx_este_m": dx, "dy_norte_m": dy,
                             "residuo_medio_no_pico": est["score_at_peak"], "residuo_medio_sem_deslocamento": est["score_at_zero"]},
        "estabilidade_meia_cena": half,
        "verificacao_1_quarteira": quarteira,
        "verificacao_2_cosmo_quga1": cosmo,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.write:
        out = lith.copy()
        out["geometry"] = shift_polygons(list(lith.geometry), dx, dy)
        out.to_crs(4326).to_file(ALIGNED, driver="GeoJSON")
        print(f"GeoJSON alinhado escrito em {ALIGNED}")


if __name__ == "__main__":
    main()
