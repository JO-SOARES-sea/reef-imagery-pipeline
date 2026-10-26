#!/usr/bin/env python3
"""
scripts/extract_sl3_soundings.py — Batimetria própria do barco a partir dos registos Lowrance `.sl3` (hipótese H6).

Para cada registo: frames do canal de sonda (2 = downscan, mesma profundidade que o primário), deduplicados por posição,
reamostrados a cada ~``--step-m`` m ao longo do trilho. Calcula o relevo ao longo do percurso
(profundidade − mediana móvel de ``--window-m`` m; positivo = acima do fundo envolvente) e a rugosidade (desvio-padrão
local). Marca os pontos dentro do holdout pré-registado (`src.constants.HOLDOUT_BBOX_WGS84`), que não podem entrar no
desenvolvimento.

A profundidade é a lida pela sonda (abaixo do transdutor, sem correção de maré): serve para relevo relativo, não para
cotas absolutas.

Saída: data/external/own_soundings/sl3_soundings.csv (+ .gpkg).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.constants import HOLDOUT_BBOX_WGS84  # noqa: E402
from src.navigation.lowrance_sl3 import iter_frames  # noqa: E402

OUT = PROJECT_ROOT / "data" / "external" / "own_soundings"


def process(path: str, step_m: float, window_m: float) -> pd.DataFrame:
    rows = [(f.lat, f.lon, f.depth_m) for f in iter_frames(path, channels={2})]
    d = pd.DataFrame(rows, columns=["lat", "lon", "depth_m"])
    d = d[(d.depth_m > 0.5) & (d.depth_m < 120) & d.lat.between(36, 38) & d.lon.between(-9.5, -7)]
    x, y = Transformer.from_crs(4326, 32629, always_xy=True).transform(d.lon.values, d.lat.values)
    d["x"], d["y"] = x, y
    seg = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    d = d[seg < 50]  # saltos de GPS
    d["s"] = np.r_[0, np.cumsum(np.hypot(np.diff(d.x.values), np.diff(d.y.values)))]
    bins = (d.s // step_m).astype(int)
    g = d.groupby(bins).agg(lat=("lat", "mean"), lon=("lon", "mean"), x=("x", "mean"), y=("y", "mean"),
                            s=("s", "mean"), depth_m=("depth_m", "median"), n=("depth_m", "size")).reset_index(drop=True)
    w = max(3, int(round(window_m / step_m)) | 1)
    base = g.depth_m.rolling(w, center=True, min_periods=w // 2).median()
    g["relief_m"] = base - g.depth_m  # positivo = mais raso que o envolvente (pedra saliente)
    g["rough_m"] = g.depth_m.rolling(5, center=True, min_periods=3).std()
    g["source"] = Path(path).name
    W, S, E, N = HOLDOUT_BBOX_WGS84
    g["holdout"] = g.lon.between(W, E) & g.lat.between(S, N)
    return g


def main() -> None:
    p = argparse.ArgumentParser(description="Batimetria própria do barco a partir de registos Lowrance SL3")
    p.add_argument("--glob", default="/Volumes/4G_WD_BLACK/Marinho_Windows/Sonar_*.sl3")
    p.add_argument("--step-m", type=float, default=5.0)
    p.add_argument("--window-m", type=float, default=50.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    allg = pd.concat([process(f, a.step_m, a.window_m) for f in sorted(glob.glob(a.glob))], ignore_index=True)
    allg.to_csv(OUT / "sl3_soundings.csv", index=False)
    gpd.GeoDataFrame(allg, geometry=gpd.points_from_xy(allg.lon, allg.lat), crs=4326).to_file(OUT / "sl3_soundings.gpkg", driver="GPKG")
    print(allg.groupby("source").agg(pontos=("depth_m", "size"), prof_med=("depth_m", "median"), prof_max=("depth_m", "max"),
                                     relevo_p95=("relief_m", lambda v: v.quantile(.95)), holdout=("holdout", "sum")).round(2).to_string())


if __name__ == "__main__":
    main()
