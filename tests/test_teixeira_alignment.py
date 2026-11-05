"""Alinhamento dos polígonos de Teixeira (Fig. 20, georreferenciada à mão) pela imagem — offline, sintético."""

import numpy as np
import pytest
from shapely.geometry import box

from src.analysis.teixeira_alignment import depth_detrend, estimate_shift, shift_polygons

RES = 10.0


def _scene(true_dr=-2, true_dc=3, seed=0, noise=0.3, ny=120, nx=140):
    """Rocha escura (resíduo +1) deslocada de (true_dr, true_dc) píxeis em relação à máscara dos polígonos."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((ny, nx), bool)
    mask[40:60, 50:90] = True
    mask[75:90, 30:45] = True
    resid = rng.normal(0, noise, (ny, nx))
    resid[np.roll(np.roll(mask, true_dr, axis=0), true_dc, axis=1)] += 1.0
    return mask, resid, np.ones((ny, nx), bool)


def test_estimate_shift_recovers_known_offset():
    mask, resid, valid = _scene(true_dr=-2, true_dc=3)
    out = estimate_shift(mask, resid, valid, max_shift_px=8)
    assert (out["dr"], out["dc"]) == (-2, 3)
    assert out["score_at_peak"] > out["score_at_zero"]


def test_estimate_shift_is_zero_when_already_aligned():
    mask, resid, valid = _scene(true_dr=0, true_dc=0)
    out = estimate_shift(mask, resid, valid, max_shift_px=8)
    assert (out["dr"], out["dc"]) == (0, 0)


def test_estimate_shift_ignores_invalid_pixels():
    mask, resid, valid = _scene(true_dr=1, true_dc=-2)
    resid = resid.copy()
    valid = valid.copy()
    valid[:, :20] = False
    resid[:, :20] = 99.0  # lixo em píxeis inválidos não pode arrastar o pico
    out = estimate_shift(mask, resid, valid, max_shift_px=8)
    assert (out["dr"], out["dc"]) == (1, -2)


def test_depth_detrend_removes_pure_depth_dependence():
    rng = np.random.default_rng(1)
    depth = rng.uniform(3, 20, (80, 80)).astype(np.float32)
    score = 0.7 * depth  # sinal só de profundidade
    valid = np.ones_like(depth, bool)
    resid = depth_detrend(score, depth, valid, bin_m=0.5)
    assert np.nanstd(resid) < 0.05 * np.std(score)


def test_depth_detrend_keeps_local_anomaly():
    rng = np.random.default_rng(2)
    depth = rng.uniform(3, 20, (80, 80)).astype(np.float32)
    score = 0.7 * depth
    score[10:14, 10:14] += 5.0
    resid = depth_detrend(score, depth, np.ones_like(depth, bool))
    assert np.nanmean(resid[10:14, 10:14]) > 4.0


def test_shift_polygons_translates_in_metres_and_keeps_area():
    polys = [box(0, 0, 100, 50)]
    moved = shift_polygons(polys, dx_m=30.0, dy_m=-20.0)
    assert moved[0].bounds == (30.0, -20.0, 130.0, 30.0)
    assert moved[0].area == pytest.approx(polys[0].area)


def test_pixel_shift_to_metres_convention():
    """Linhas crescem para sul: dr=-2 (acima) = +20 m a norte; dc=3 = +30 m a este."""
    from src.analysis.teixeira_alignment import pixel_shift_to_metres

    assert pixel_shift_to_metres(dr=-2, dc=3, res_m=RES) == (30.0, 20.0)
