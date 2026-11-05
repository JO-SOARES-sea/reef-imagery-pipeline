"""Verdade-terreno a partir do COSMO (LiDAR da APA): pontos de rocha estável com identificador de cume — offline."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine

from src.analysis.cosmo_truth import cosmo_rock_points

ROOT = Path(__file__).resolve().parent.parent
T = Affine.translation(580000.0, 4100000.0) @ Affine.scale(10.0, -10.0)


def _cls():
    cls = np.full((100, 100), np.nan, np.float32)
    cls[10:90, 10:90] = 0.0  # pegada com fundo liso
    cls[20:24, 20:26] = 1.0  # cume A
    cls[22:24, 27:29] = 1.0  # fragmento do cume A, a ~10 m
    cls[70:75, 70:75] = 1.0  # cume B, longe
    return cls


def test_one_point_per_rock_pixel_in_wgs84():
    df = cosmo_rock_points(_cls(), T, "EPSG:32629")
    assert len(df) == int((_cls() == 1).sum())
    assert df.lon.between(-9.0, -8.0).all() and df.lat.between(36.8, 37.3).all()


def test_nearby_fragments_share_a_cume_and_distant_ones_do_not():
    df = cosmo_rock_points(_cls(), T, "EPSG:32629", merge_px=2)
    assert df.cume_id.nunique() == 2
    sizes = sorted(df.groupby("cume_id").size())
    assert sizes == [25, 24 + 4]


def test_no_rock_gives_empty_frame_with_columns():
    cls = np.zeros((20, 20), np.float32)
    df = cosmo_rock_points(cls, T, "EPSG:32629")
    assert df.empty and list(df.columns) == ["lon", "lat", "cume_id"]


def test_quarteira_zone_uses_cosmo_truth_not_the_invalid_csv():
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module("scripts.run_reef_pipeline")
    path = mod.ZONE_CONFIGS["quarteira"]["ground_truth_path"]
    assert path.name == "quarteira_cosmo_rocha.csv"
    assert {"lon", "lat", "cume_id"} <= set(pd.read_csv(path, nrows=1).columns)
