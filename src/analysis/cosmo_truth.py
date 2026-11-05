"""
src/analysis/cosmo_truth.py — Verdade-terreno de rocha a partir do COSMO (LiDAR da APA, DEM 0,30 m).

Os rasters ``data/processed/cosmo_truth/*_truth10m.tif`` têm a classe na banda 5 (1 = rocha estável: relevo ≥ 0,3 m
nos dois anos e |Δz| ≤ 0,3 m; 0 = liso; NaN = fora da pegada). Aqui convertem-se os píxeis de rocha em pontos
WGS84 com um ``cume_id`` (fragmentos a ≤ ``merge_px`` píxeis ficam no mesmo cume), a unidade de reamostragem
do bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage as ndi

COLUMNS = ["lon", "lat", "cume_id"]


def cosmo_rock_points(cls: np.ndarray, transform, crs, merge_px: int = 2) -> pd.DataFrame:
    """Centros dos píxeis de rocha (``cls == 1``) em lon/lat, com ``cume_id`` por agrupamento espacial."""
    rock = cls == 1
    if not rock.any():
        return pd.DataFrame(columns=COLUMNS)
    grown, _ = ndi.label(ndi.binary_dilation(rock, structure=np.ones((3, 3), bool), iterations=merge_px), structure=np.ones((3, 3), int))
    rows, cols = np.nonzero(rock)
    x, y = transform * (cols + 0.5, rows + 0.5)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(np.asarray(x), np.asarray(y))
    ids = grown[rows, cols]
    _, dense = np.unique(ids, return_inverse=True)
    return pd.DataFrame({"lon": lon, "lat": lat, "cume_id": dense + 1})
