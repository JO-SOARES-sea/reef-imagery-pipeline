"""Validação GT vs fundo aleatório (evaluate_against_background) — offline, rasters sintéticos."""

import numpy as np
import pytest

from src.analysis.reef_segmenter import evaluate_against_background

GT_ROWS = np.array([20, 50, 80])
GT_COLS = np.array([20, 50, 80])


def test_gt_on_hotspots_gives_auc_near_one():
    proba = np.full((100, 100), 0.2, dtype=np.float32)
    proba[GT_ROWS, GT_COLS] = 0.9
    res = evaluate_against_background(proba, np.ones_like(proba, dtype=bool), GT_ROWS, GT_COLS)
    assert res["auc"] == pytest.approx(1.0)
    assert res["gt_detection_rate"] == 1.0
    assert res["bg_detection_rate"] == 0.0


def test_uniform_noise_gives_auc_near_chance():
    # Ruído sem relação com os recifes: o recall pode ser alto, mas a AUC denuncia o acaso
    rng = np.random.default_rng(1)
    proba = rng.uniform(0.3, 0.7, (300, 300)).astype(np.float32)
    gt_r = rng.integers(10, 290, 200)
    gt_c = rng.integers(10, 290, 200)
    res = evaluate_against_background(proba, np.ones_like(proba, dtype=bool), gt_r, gt_c)
    assert res["auc"] == pytest.approx(0.5, abs=0.07)
    assert res["gt_detection_rate"] == pytest.approx(res["bg_detection_rate"], abs=0.08)


def test_background_excludes_area_around_gt_and_outside_mask():
    proba = np.zeros((100, 100), dtype=np.float32)
    mask = np.zeros_like(proba, dtype=bool)
    mask[:, :50] = True
    # Zonas quentes afastadas >1 px das fronteiras, para não entrarem na janela 3x3 do fundo
    proba[:, 55:] = 1.0  # fora da máscara: nunca pode ser amostrado como fundo
    proba[17:24, 17:24] = 1.0  # vizinhança do GT (exclusão de 5 px): excluída do fundo
    res = evaluate_against_background(proba, mask, np.array([20]), np.array([20]))
    assert res["bg_detection_rate"] == 0.0
    assert res["auc"] == pytest.approx(1.0)


def test_nan_pixels_do_not_break_auc():
    proba = np.full((100, 100), 0.2, dtype=np.float32)
    proba[40:60, 40:60] = np.nan
    proba[GT_ROWS[0], GT_COLS[0]] = 0.9
    res = evaluate_against_background(proba, np.ones_like(proba, dtype=bool), GT_ROWS, GT_COLS)
    assert res["auc"] is not None and np.isfinite(res["auc"])
    # Só (20, 20) está quente; (50, 50) cai no bloco NaN → 0, não detetado
    assert res["gt_detection_rate"] == pytest.approx(1 / 3)


def test_no_ground_truth_returns_none():
    proba = np.zeros((10, 10), dtype=np.float32)
    res = evaluate_against_background(proba, np.ones_like(proba, dtype=bool), np.array([]), np.array([]))
    assert res["auc"] is None
