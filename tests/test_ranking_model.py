import os
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from src.ranking_model import (
    predict_score,
    schema_fingerprint,
    terrain_exposure_modifier,
    validate_schema,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset all module-level model cache before AND after every test.

    Without teardown a test that triggers real model loading (when model files
    happen to exist) would poison the cache for the next test, making results
    order-dependent.
    """
    def _reset():
        import src.ranking_model as rm
        rm._RANKER_MODEL = None
        rm._FEATURE_SCHEMA = None
        rm._DISABLED_FEATURES = set()
        rm._SCHEMA_FINGERPRINT = None
        rm._IS_FALLBACK = False
        rm._LOAD_ATTEMPTED = False

    _reset()
    yield
    _reset()

def test_missing_model_fallback():
    """Test that if the model is missing, the system gracefully reverts to fallback."""
    with patch('os.path.exists', return_value=False):
        features = {
            "benthic_contrast": 0.2,
            "snr": 100.0,
            "fft_clean": 10000,
            "edge_entropy": 6.0,
            "dyn_range": 0.008,
            "signal": 0.12,
        }
        res = predict_score(features)

        assert res["mode"] == "Fallback"
        assert res["score"] > 0
        assert "ML Model unavailable" in res["reason"]
        assert res["features_used"]["benthic_contrast"] == 0.2

def test_extreme_cloud_cover():
    """Test hard filter for clouds > 80%."""
    features = {
        'cloud_cover': 85.0
    }
    res = predict_score(features)
    assert res["score"] == 0.0
    assert res["mode"] == "HardFilter"
    assert "Cloud cover exceeded 80%" in res["reason"]

def test_successful_ml_inference():
    """Test that ML inference works when schema matches."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.95])

    mock_meta = '{"features": ["benthic_contrast", "snr"]}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        features = {
            "benthic_contrast": 0.2,
            "snr": 100.0,
            "fft_clean": 10000,
            "edge_entropy": 6.0,
            "dyn_range": 0.008,
            "signal": 0.12,
            "cloud_cover": 5.0,
        }
        res = predict_score(features)

        assert res["mode"] == "ML"
        assert res["score"] == 0.95
        assert mock_model.predict.called

        called_args = mock_model.predict.call_args[0][0]
        assert called_args.iloc[0, 0] == 0.2
        assert called_args.iloc[0, 1] == 100.0

def test_mismatched_schema():
    """Test that a missing required feature gracefully reverts to fallback."""
    mock_model = MagicMock()

    mock_meta = '{"features": ["benthic_contrast", "non_existent_feature"]}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        features = {
            "benthic_contrast": 0.2,
        }
        res = predict_score(features)

        assert res["mode"] == "Fallback"
        assert not mock_model.predict.called

def test_bathymetry_features_present():
    """Test inference works when extra bathymetry features are present in input."""
    features = {
        "benthic_contrast": 0.2,
        "snr": 100.0,
        "fft_clean": 10000,
        "edge_entropy": 6.0,
        "dyn_range": 0.008,
        "signal": 0.12,
        "cloud_cover": 5.0,
        "nearest_isobath_distance_m": 150.0,
        "bathymetry_zone_class": "shallow_reef",
        "bathy_slope_proxy": 2.5,
    }

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.88])
    mock_meta = '{"features": ["benthic_contrast", "snr", "fft_clean", "edge_entropy", "dyn_range", "signal"]}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        res = predict_score(features)

        assert res["mode"] == "ML"
        assert res["score"] == 0.88

        called_args = mock_model.predict.call_args[0][0]
        assert called_args.iloc[0, 0] == 0.2
        assert called_args.iloc[0, 1] == 100.0
        assert called_args.iloc[0, 2] == 10000
        assert called_args.iloc[0, 3] == 6.0
        assert called_args.iloc[0, 4] == 0.008
        assert called_args.iloc[0, 5] == 0.12

def test_bathymetry_features_missing():
    """Test inference works when bathymetry features are missing or inf in input."""
    features = {
        "benthic_contrast": 0.2,
        "snr": 100.0,
        "fft_clean": 10000,
        "edge_entropy": 6.0,
        "dyn_range": 0.008,
        "signal": 0.12,
        "cloud_cover": 5.0,
        "nearest_isobath_distance_m": np.inf,
    }

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.77])
    mock_meta = '{"features": ["benthic_contrast", "snr", "fft_clean", "edge_entropy", "dyn_range", "signal"]}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        res = predict_score(features)

        assert res["mode"] == "ML"
        assert res["score"] == 0.77

def test_contrast_ignored_in_ml_schema():
    """Test that deprecated contrast is safely ignored when not in model schema."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.72])

    mock_meta = '{"features": ["benthic_contrast", "snr", "fft_clean", "edge_entropy"], "disabled_features": {"contrast": "non-discriminative"}}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        features = {
            "benthic_contrast": 0.2,
            "snr": 100.0,
            "fft_clean": 10000,
            "edge_entropy": 6.0,
            "dyn_range": 0.008,
            "signal": 0.12,
            "contrast": 0.8,
            "cloud_cover": 5.0,
        }
        res = predict_score(features)

        assert res["mode"] == "ML"
        assert res["score"] == 0.72

        called_args = mock_model.predict.call_args[0][0]
        assert list(called_args.columns) == ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]
        assert called_args.shape[1] == 4


def test_legacy_caller_without_contrast():
    """Test that callers who do NOT send deprecated contrast still work fine."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.65])

    mock_meta = '{"features": ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        features = {
            "benthic_contrast": 0.2,
            "snr": 100.0,
            "fft_clean": 10000,
            "edge_entropy": 6.0,
            "dyn_range": 0.008,
            "signal": 0.12,
        }
        res = predict_score(features)

        assert res["mode"] == "ML"
        assert res["score"] == 0.65
        assert 'benthic_contrast' in res["features_used"]


def test_predict_score_output_contract():
    """Test that predict_score always returns the same stable contract."""
    with patch('os.path.exists', return_value=False):
        features = {
            "benthic_contrast": 0.2,
            "snr": 100.0,
            "fft_clean": 10000,
            "edge_entropy": 6.0,
            "dyn_range": 0.008,
            "signal": 0.12,
        }
        res = predict_score(features)

        assert "score" in res
        assert "mode" in res
        assert "features_used" in res
        assert isinstance(res["score"], float)
        assert res["mode"] in ("ML", "Fallback", "HardFilter")


def test_deterministic_training():
    """Test that training the ML model with the same fixed seed yields identical feature importances."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    model1 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model1.fit(X, y)
    imp1 = model1.feature_importances_

    model2 = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model2.fit(X, y)
    imp2 = model2.feature_importances_

    np.testing.assert_array_equal(imp1, imp2)


def test_training_without_contrast():
    """Test that train_feature_ranker FEATURE_COLS uses B02-only features."""
    train_ranker = pytest.importorskip(
        "train_feature_ranker",
        reason="scripts/train_feature_ranker.py not installed"
    )
    FEATURE_COLS = train_ranker.FEATURE_COLS
    DISABLED_FEATURES = train_ranker.DISABLED_FEATURES

    assert 'contrast' not in FEATURE_COLS
    assert 'contrast' in DISABLED_FEATURES
    assert len(FEATURE_COLS) == 6


def test_canonical_model_path_loaded():
    """Test that ranking_model.py only loads the canonical model path."""
    import src.ranking_model as rm

    assert rm.MODEL_PATH.endswith("feature_ranker_model.pkl")
    assert rm.METADATA_PATH.endswith("feature_ranker_metadata.json")
    assert "visibility_rf_bathy" not in rm.MODEL_PATH


def test_metadata_marks_canonical_and_deprecated():
    """Test that metadata explicitly marks canonical status and deprecated models."""
    import json
    meta_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_ranker_metadata.json')

    if not os.path.exists(meta_path):
        pytest.skip("models/feature_ranker_metadata.json not present")

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta.get("canonical") is True
    assert "schema_version" in meta

    assert "deprecated_models" in meta
    assert "visibility_rf_bathy.pkl" in meta["deprecated_models"]

    assert "contrast" not in meta["features"]
    assert "contrast" in meta.get("disabled_features", {})


def test_legacy_model_not_loaded_by_ranking():
    """Test that predict_score never loads the legacy visibility_rf_bathy model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.60])

    mock_meta = '{"features": ["benthic_contrast", "snr", "fft_clean", "edge_entropy", "dyn_range", "signal"], "canonical": true, "schema_version": "2.0"}'

    with patch('os.path.exists', return_value=True) as mock_exists, \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        res = predict_score({"benthic_contrast": 0.2, "snr": 100.0, "fft_clean": 10000, "edge_entropy": 6.0, "dyn_range": 0.008, "signal": 0.12})

        assert res["mode"] == "ML"
        for call in mock_exists.call_args_list:
            path_arg = call[0][0] if call[0] else ""
            assert "visibility_rf_bathy" not in path_arg


# =============================================================================
# Schema Drift Detection Tests
# =============================================================================

def test_validate_schema_exact_match():
    """Test that exact schema match returns ok=True with no drift."""
    incoming = {"benthic_contrast": 0.2, "snr": 100.0, "fft_clean": 10000, "edge_entropy": 6.0}
    schema = ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]

    result = validate_schema(incoming, schema)

    assert result["ok"] is True
    assert result["missing"] == set()
    assert result["extra"] == set()
    assert result["deprecated"] == set()
    assert result["type_errors"] == []


def test_validate_schema_missing_required():
    """Test that missing required features are detected."""
    incoming = {"benthic_contrast": 0.2, "snr": 100.0}
    schema = ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]

    result = validate_schema(incoming, schema)

    assert result["ok"] is False
    assert result["missing"] == {"fft_clean", "edge_entropy"}


def test_validate_schema_extra_features():
    """Test that extra features are detected but don't block."""
    incoming = {"benthic_contrast": 0.2, "snr": 100.0, "fft_clean": 10000,
                "edge_entropy": 6.0, "some_new_feature": 1.0}
    schema = ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]

    result = validate_schema(incoming, schema)

    assert result["ok"] is True
    assert 'some_new_feature' in result["extra"]


def test_validate_schema_deprecated_features():
    """Test that deprecated features are classified separately from extras."""
    incoming = {"benthic_contrast": 0.2, "snr": 100.0, "fft_clean": 10000,
                "edge_entropy": 6.0, "contrast": 0.8}
    schema = ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]
    disabled = {"contrast"}

    result = validate_schema(incoming, schema, disabled=disabled)

    assert result["ok"] is True
    assert 'contrast' in result["deprecated"]
    assert 'contrast' not in result["extra"]


def test_validate_schema_type_mismatch():
    """Test that non-numeric feature values are caught."""
    incoming = {"benthic_contrast": 0.2, "snr": "bad_string",
                "fft_clean": 10000, "edge_entropy": 6.0}
    schema = ["benthic_contrast", "snr", "fft_clean", "edge_entropy"]

    result = validate_schema(incoming, schema)

    assert result["ok"] is False
    assert any("snr" in e for e in result["type_errors"])


def test_schema_fingerprint_stable():
    """Test that fingerprint is stable and order-independent."""
    fp1 = schema_fingerprint(["benthic_contrast", "snr", "fft_clean", "edge_entropy"])
    fp2 = schema_fingerprint(["edge_entropy", "fft_clean", "snr", "benthic_contrast"])
    fp3 = schema_fingerprint(["benthic_contrast", "snr", "fft_clean"])

    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 12


def test_drift_missing_feature_triggers_fallback():
    """Test that missing required feature in ML path triggers fallback."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.70])

    mock_meta = '{"features": ["benthic_contrast", "snr", "nonexistent_required"], "disabled_features": {}}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        res = predict_score({"benthic_contrast": 0.2, "snr": 100.0, "fft_clean": 10000, "edge_entropy": 6.0, "dyn_range": 0.008, "signal": 0.12})

        assert res["mode"] == "Fallback"
        assert not mock_model.predict.called


def test_drift_extra_features_do_not_block():
    """Test that extra features in caller do not block ML inference."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.85])

    mock_meta = '{"features": ["benthic_contrast", "snr"], "disabled_features": {"contrast": "disabled"}}'

    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_meta)), \
         patch('pickle.load', return_value=mock_model):

        res = predict_score({
            "benthic_contrast": 0.2, "snr": 100.0, "fft_clean": 10000,
            "edge_entropy": 6.0, "dyn_range": 0.008, "signal": 0.12,
            "contrast": 0.8,
            "nearest_isobath_distance_m": 150.0,
        })

        assert res["mode"] == "ML"
        assert res["score"] == 0.85


# =============================================================================
# Phase 4: Terrain Exposure Modifier Tests
# =============================================================================

class TestTerrainExposureModifier:
    def test_returns_float_in_range(self):
        m = terrain_exposure_modifier(1.0, 180.0)
        assert isinstance(m, float)
        assert 0.5 <= m <= 1.0

    def test_sheltered_coast_higher_than_exposed(self):
        exposed  = terrain_exposure_modifier(0.0, 180.0, swell_direction=225.0)  # S coast, SW swell
        sheltered = terrain_exposure_modifier(0.0,   0.0, swell_direction=225.0)  # N coast, SW swell
        assert sheltered > exposed

    def test_steep_slope_penalises(self):
        flat  = terrain_exposure_modifier(0.0,  180.0)
        steep = terrain_exposure_modifier(15.0, 180.0)
        assert steep < flat

    def test_floor_at_0_5(self):
        m = terrain_exposure_modifier(100.0, 225.0, swell_direction=225.0)
        assert m >= 0.5

    def test_ceil_at_1_0(self):
        m = terrain_exposure_modifier(0.0, 45.0, swell_direction=225.0)
        assert m <= 1.0

    def test_pedra_eulalia_known_value(self):
        m = terrain_exposure_modifier(1.076, 179.94, swell_direction=225.0)
        assert 0.75 <= m <= 0.80

    def test_zero_slope_zero_aspect(self):
        m = terrain_exposure_modifier(0.0, 0.0)
        assert 0.5 <= m <= 1.0


class TestPredictScoreWithTerrain:
    def test_terrain_modifier_applied(self):
        feats = dict(benthic_contrast=0.3, snr=110.0, fft_clean=12000,
                     edge_entropy=6.8, dyn_range=0.009, signal=0.13)
        base    = predict_score(feats)
        terrain = predict_score(feats, terrain_features=dict(slope_mean=2.0, aspect_mean=180.0))
        assert terrain["score"] <= base["score"]
        assert "terrain_modifier" in terrain
        assert 0.5 <= terrain["terrain_modifier"] <= 1.0

    def test_terrain_modifier_in_result_keys(self):
        feats = dict(benthic_contrast=0.3, snr=110.0, fft_clean=12000,
                     edge_entropy=6.8, dyn_range=0.009, signal=0.13)
        r = predict_score(feats, terrain_features=dict(slope_mean=1.0, aspect_mean=180.0))
        assert "terrain_modifier" in r
        assert "terrain_features" in r

    def test_no_terrain_preserves_contract(self):
        feats = dict(benthic_contrast=0.3, snr=110.0, fft_clean=12000,
                     edge_entropy=6.8, dyn_range=0.009, signal=0.13)
        r = predict_score(feats)
        assert "terrain_modifier" not in r
        assert "score" in r and "mode" in r and "features_used" in r

    def test_terrain_none_slopes_handled(self):
        feats = dict(benthic_contrast=0.3, snr=110.0, fft_clean=12000,
                     edge_entropy=6.8, dyn_range=0.009, signal=0.13)
        r = predict_score(feats, terrain_features=dict(slope_mean=None, aspect_mean=None))
        assert 0.0 <= r["score"] <= 1.0

    def test_cloud_filter_ignores_terrain(self):
        r = predict_score({"cloud_cover": 90.0},
                          terrain_features=dict(slope_mean=5.0, aspect_mean=180.0))
        assert r["score"] == 0.0
        assert r["mode"] == "HardFilter"
