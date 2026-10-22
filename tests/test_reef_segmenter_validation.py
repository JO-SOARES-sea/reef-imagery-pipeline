"""
tests/test_reef_segmenter_validation.py — Testes unitários para Validação Exact-Pixel ROC-AUC e Classificador.
"""

from pathlib import Path
import numpy as np
import pytest
from src.analysis.reef_segmenter import (
    ReefClassifier,
    compute_reef_probability_absolute,
    evaluate_against_background,
    extract_candidate_reefs,
)


def test_evaluate_against_background_exact_pixel():
    """Verifica que a avaliação contra o fundo calcula ROC-AUC exato sem inflação 3x3."""
    h, w = 100, 100
    proba = np.full((h, w), 0.10, dtype=np.float32)
    analysis_mask = np.ones((h, w), dtype=bool)

    # 10 pontos de Ground Truth com valores elevados (0.80)
    gt_rows = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 15])
    gt_cols = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 85])
    proba[gt_rows, gt_cols] = 0.80

    eval_res = evaluate_against_background(
        proba,
        analysis_mask,
        gt_rows,
        gt_cols,
        n_background=500,
        exclusion_px=5,
        seed=42,
    )

    assert eval_res["auc_exact"] is not None
    # Separação perfeita esperada: todos GT=0.80, todos BG=0.10
    assert eval_res["auc_exact"] == 1.0
    assert eval_res["p_value"] < 1e-4
    assert eval_res["gt_median"] == pytest.approx(0.80, abs=1e-3)
    assert eval_res["bg_median"] == pytest.approx(0.10, abs=1e-3)
    assert eval_res["delta_contrast"] == pytest.approx(0.70, abs=1e-3)
    assert eval_res["recall_fpr5"] == 1.0


def test_evaluate_against_background_random_null():
    """Verifica que um mapa de ruído puro resulta em ROC-AUC próximo de 0.50 (acaso)."""
    h, w = 100, 100
    rng = np.random.default_rng(123)
    proba = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
    analysis_mask = np.ones((h, w), dtype=bool)

    gt_rows = rng.integers(10, 90, size=20)
    gt_cols = rng.integers(10, 90, size=20)

    eval_res = evaluate_against_background(
        proba,
        analysis_mask,
        gt_rows,
        gt_cols,
        n_background=1000,
        exclusion_px=5,
        seed=42,
    )

    # Em ruído aleatório, a AUC deve situar-se perto de 0.50 (intervalo 0.35 - 0.65)
    assert 0.35 <= eval_res["auc_exact"] <= 0.65
    assert eval_res["delta_contrast"] == pytest.approx(0.0, abs=0.20)


def test_reef_classifier_fit_predict():
    """Verifica o treino e previsão matricial do ReefClassifier."""
    from sklearn.ensemble import RandomForestClassifier
    custom_rf = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_leaf=1, random_state=42)
    clf = ReefClassifier(model=custom_rf, feature_names=["f1", "f2"])
    X = np.array([
        [1.0, 2.0],
        [1.2, 1.8],
        [0.0, 0.1],
        [0.1, 0.2],
    ], dtype=np.float32)
    y = np.array([1, 1, 0, 0])

    clf.fit(X, y)
    assert clf.is_fitted

    preds = clf.predict_proba(X)
    assert len(preds) == 4
    assert preds[0] > preds[2]

    # Previsão raster
    h, w = 10, 10
    features = {
        "f1": np.full((h, w), 1.0, dtype=np.float32),
        "f2": np.full((h, w), 2.0, dtype=np.float32),
    }
    mask = np.zeros((h, w), dtype=bool)
    mask[2:5, 2:5] = True

    proba_map = clf.predict_proba_map(features, mask)
    assert proba_map.shape == (h, w)
    assert np.all(proba_map[~mask] == 0.0)
    assert np.all(proba_map[mask] > 0.5)


def test_compute_reef_probability_absolute():
    """Verifica que o modo absoluto não depende de estiramento percentílico de cena."""
    h, w = 20, 20
    # Fundo plano com rugosidade nula
    features = {
        "sdb_vrm50": np.zeros((h, w), dtype=np.float32),
        "sdb_bpi100": np.zeros((h, w), dtype=np.float32),
        "sdb_slope": np.zeros((h, w), dtype=np.float32),
        "B03_std30": np.zeros((h, w), dtype=np.float32),
    }
    water_mask = np.ones((h, w), dtype=bool)
    depth_mask = np.ones((h, w), dtype=bool)

    proba = compute_reef_probability_absolute(features, water_mask, depth_mask)
    # Areia plana tem probabilidade exatamente 0.0
    assert np.all(proba == 0.0)

    # Anomalia rochosa com VRM e BPI elevados
    features["sdb_vrm50"][10, 10] = 0.007
    features["sdb_bpi100"][10, 10] = 1.5
    features["sdb_slope"][10, 10] = 0.15
    features["B03_std30"][10, 10] = 0.0025

    proba_reef = compute_reef_probability_absolute(features, water_mask, depth_mask)
    assert proba_reef[10, 10] > 0.80
    assert proba_reef[0, 0] == 0.0
