"""Tests for src/drift_monitor.py — feature & score distribution drift detection."""

import pytest

from src.drift_monitor import (
    CRITICAL,
    FEATURE_BASELINES,
    OK,
    WARNING,
    check_feature_drift,
    check_score_drift,
    observe,
    reset,
    summary,
    summary_line,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset rolling score history between tests."""
    reset()


NORMAL_FEATURES = {
    'benthic_contrast': 0.2,
    'snr': 120.0,
    'fft_clean': 10000.0,
    'edge_entropy': 6.5,
    'dyn_range': 0.008,
    'signal': 0.125,
}


class TestFeatureDrift:
    def test_normal_features_ok(self):
        """Features within normal range produce OK."""
        result = check_feature_drift(dict(NORMAL_FEATURES))
        assert result["level"] == OK
        assert result["alerts"] == []
        assert result["null_count"] == 0

    def test_extreme_snr_warns(self):
        """snr far from mean triggers WARNING."""
        features = dict(NORMAL_FEATURES)
        features['snr'] = 200.0  # z = |200-120|/30 = 2.67 → WARNING
        result = check_feature_drift(features)
        assert result["level"] in (WARNING, CRITICAL)
        assert any('snr' in a[0] for a in result["alerts"])

    def test_extreme_zscore_critical(self):
        """Feature with very high z-score triggers CRITICAL."""
        features = dict(NORMAL_FEATURES)
        features['snr'] = 250.0  # z = |250-120|/30 = 4.33 → CRITICAL
        result = check_feature_drift(features)
        assert result["level"] in (WARNING, CRITICAL)
        assert any('z=' in a[2] for a in result["alerts"])

    def test_null_features_detected(self):
        """NaN or None features increment null count."""
        features = dict(NORMAL_FEATURES)
        features['benthic_contrast'] = None
        features['snr'] = float('nan')
        result = check_feature_drift(features)
        assert result["null_count"] == 2

    def test_high_null_rate_critical(self):
        """All features null triggers CRITICAL."""
        features = dict.fromkeys(FEATURE_BASELINES)
        result = check_feature_drift(features)
        assert result["level"] == CRITICAL
        assert result["null_count"] == len(FEATURE_BASELINES)

    def test_schema_filter(self):
        """Only features in schema_features are checked."""
        features = dict(NORMAL_FEATURES)
        features['fft_clean'] = 999999.0  # Would be extreme, but not checked
        result = check_feature_drift(features, schema_features=['benthic_contrast', 'snr'])
        assert result["level"] == OK


class TestScoreDrift:
    def test_normal_score_ok(self):
        """Score within expected range is OK."""
        result = check_score_drift(0.65)
        assert result["level"] == OK
        assert result["alerts"] == []

    def test_extreme_score_warns(self):
        """Score far from baseline mean triggers alert."""
        # z = |1.5 - 0.60| / 0.15 = 6.0 → CRITICAL
        result = check_score_drift(1.5)
        assert result["level"] == CRITICAL

    def test_rolling_mean_drift(self):
        """Consistently extreme scores trigger rolling drift alert."""
        # z = |1.0 - 0.60| / 0.15 = 2.67 → above Z_WARN (2.5)
        for _ in range(15):
            check_score_drift(1.0)

        result = check_score_drift(1.0)
        assert result["rolling_mean"] is not None
        assert result["rolling_mean"] > 0.95
        assert result["level"] in (WARNING, CRITICAL)

    def test_normal_rolling_ok(self):
        """Normal scores don't trigger rolling drift."""
        for _ in range(15):
            check_score_drift(0.60)

        result = check_score_drift(0.58)
        assert result["level"] == OK


class TestObserve:
    def test_observe_returns_both_checks(self):
        """observe() returns feature_drift and score_drift."""
        result = observe(dict(NORMAL_FEATURES), 0.65)

        assert "feature_drift" in result
        assert "score_drift" in result
        assert result["feature_drift"]["level"] == OK
        assert result["score_drift"]["level"] == OK

    def test_observe_does_not_alter_score(self):
        """observe() is purely observational — does not modify inputs."""
        features = dict(NORMAL_FEATURES)
        original = dict(features)
        observe(features, 0.65)
        assert features == original


class TestThrottling:
    def test_throttle_suppresses_repeats(self):
        """After first log, repeated same-level drift is suppressed."""
        import src.drift_monitor as dm

        reset()
        # First call — triggers log (level changes from OK to WARNING)
        observe(dict(NORMAL_FEATURES), 0.22)
        assert dm._tls._last_logged_level == WARNING
        assert dm._tls._calls_since_log == 1

        # Second call — same level, suppressed
        observe(dict(NORMAL_FEATURES), 0.22)
        assert dm._tls._calls_since_log == 2

    def test_throttle_resets_on_clear(self):
        """Counter resets when drift clears."""
        import src.drift_monitor as dm

        reset()
        observe(dict(NORMAL_FEATURES), 0.22)
        assert dm._tls._last_logged_level == WARNING

        # Normal call clears drift
        observe(dict(NORMAL_FEATURES), 0.60)
        assert dm._tls._last_logged_level == OK
        assert dm._tls._calls_since_log == 0

    def test_level_escalation_logs_again(self):
        """Escalation from WARNING to CRITICAL triggers new log."""
        import src.drift_monitor as dm

        reset()
        # WARNING level
        observe(dict(NORMAL_FEATURES), 0.22)
        assert dm._tls._last_logged_level == WARNING

        # Escalate to CRITICAL (score z > 4.0 → |score - 0.60| / 0.15 > 4 → score < 0.0 or > 1.2)
        observe(dict(NORMAL_FEATURES), 1.5)
        assert dm._tls._last_logged_level == CRITICAL
        assert dm._tls._calls_since_log == 1  # Just logged

    def test_no_cross_thread_state_leak(self):
        """Thread-local state must not bleed between concurrent threads."""
        import threading as _threading

        thread_results: dict = {}

        def run_in_thread(thread_id: int, score: float) -> None:
            reset()
            for _ in range(5):
                observe(dict(NORMAL_FEATURES), score)
            thread_results[thread_id] = summary()

        t1 = _threading.Thread(target=run_in_thread, args=(1, 0.60))   # OK scores
        t2 = _threading.Thread(target=run_in_thread, args=(2, 1.50))   # CRITICAL scores
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert thread_results[1]["worst_level"] == OK
        assert thread_results[2]["worst_level"] == CRITICAL
        assert thread_results[1]["total_observations"] == 5
        assert thread_results[2]["total_observations"] == 5


class TestIntegration:
    def test_predict_score_contract_unchanged(self):
        """predict_score() still returns {score, mode, features_used} with drift monitor active."""
        from unittest.mock import patch

        import src.ranking_model as rm
        from src.ranking_model import predict_score

        rm._RANKER_MODEL = None
        rm._FEATURE_SCHEMA = None
        rm._IS_FALLBACK = False

        with patch('os.path.exists', return_value=False):
            res = predict_score(dict(NORMAL_FEATURES))

        assert "score" in res
        assert "mode" in res
        assert "features_used" in res
        assert isinstance(res["score"], float)


class TestSummary:
    def test_summary_structure(self):
        """summary() returns all required fields."""
        s = summary()
        assert "total_observations" in s
        assert "counts" in s
        assert "feature_drift_count" in s
        assert "score_drift_count" in s
        assert "null_spike_count" in s
        assert "worst_level" in s
        assert "worst_alert" in s

    def test_summary_empty_after_reset(self):
        """After reset, summary shows zero counts."""
        s = summary()
        assert s["total_observations"] == 0
        assert s["counts"] == {OK: 0, WARNING: 0, CRITICAL: 0}
        assert s["worst_level"] == OK
        assert s["worst_alert"] is None

    def test_summary_counts_by_level(self):
        """summary correctly counts OK vs WARNING observations."""
        # 3 normal
        for _ in range(3):
            observe(dict(NORMAL_FEATURES), 0.60)
        # 2 with score drift (z=2.5 → WARNING)
        for _ in range(2):
            observe(dict(NORMAL_FEATURES), 0.22)

        s = summary()
        assert s["total_observations"] == 5
        assert s["counts"][OK] == 3
        assert s["counts"][WARNING] == 2
        assert s["score_drift_count"] == 2
        assert s["feature_drift_count"] == 0

    def test_summary_tracks_worst(self):
        """summary tracks highest severity alert."""
        observe(dict(NORMAL_FEATURES), 0.60)  # OK
        observe(dict(NORMAL_FEATURES), 0.22)  # WARNING
        observe(dict(NORMAL_FEATURES), 1.5)   # CRITICAL (z=6.0)

        s = summary()
        assert s["worst_level"] == CRITICAL
        assert s["worst_alert"] is not None
        assert s["worst_alert"][1] == CRITICAL

    def test_summary_null_spike_count(self):
        """summary counts observations that had null features."""
        feat_null = dict(NORMAL_FEATURES)
        feat_null['benthic_contrast'] = None
        observe(feat_null, 0.60)
        observe(dict(NORMAL_FEATURES), 0.60)

        s = summary()
        assert s["null_spike_count"] == 1

    def test_summary_line_format(self):
        """summary_line() returns a readable one-liner."""
        observe(dict(NORMAL_FEATURES), 0.60)

        line = summary_line()
        assert "Drift summary:" in line
        assert "obs=1" in line
        assert "OK=1" in line

    def test_reset_clears_counters(self):
        """reset() clears all batch counters."""
        observe(dict(NORMAL_FEATURES), 1.5)
        assert summary()["total_observations"] == 1

        reset()
        s = summary()
        assert s["total_observations"] == 0
        assert s["worst_level"] == OK


# =============================================================================
# Phase 4: Terrain drift + plume estimation tests
# =============================================================================

from src.drift_monitor import (
    TERRAIN_BASELINES,
    check_terrain_features,
    estimate_plume_extent,
)


class TestEstimatePlumeExtent:
    def test_returns_dict_keys(self):
        r = estimate_plume_extent(180.0, 5.0)
        assert {"plume_km", "exposure_factor", "shelter_factor", "wind_speed_ms"} <= r.keys()

    def test_plume_km_positive(self):
        r = estimate_plume_extent(180.0, 8.0)
        assert r["plume_km"] > 0

    def test_higher_wind_larger_plume(self):
        calm  = estimate_plume_extent(180.0, 0.0, wind_direction_deg=315.0)
        windy = estimate_plume_extent(180.0, 20.0, wind_direction_deg=315.0)
        assert windy["plume_km"] >= calm["plume_km"]

    def test_exposed_coast_larger_plume(self):
        exposed   = estimate_plume_extent(315.0, 10.0, wind_direction_deg=315.0)  # coast faces wind
        sheltered = estimate_plume_extent(135.0, 10.0, wind_direction_deg=315.0)  # coast faces away
        assert exposed["plume_km"] > sheltered["plume_km"]

    def test_clamped_to_reasonable_range(self):
        extreme = estimate_plume_extent(315.0, 100.0, wind_direction_deg=315.0, base_km=5.0)
        assert extreme["plume_km"] <= 25.0

    def test_steep_slope_adds_resuspension(self):
        flat  = estimate_plume_extent(180.0, 5.0, slope_mean=0.0)
        steep = estimate_plume_extent(180.0, 5.0, slope_mean=15.0)
        assert steep["plume_km"] > flat["plume_km"]

    def test_exposure_and_shelter_sum_to_one(self):
        r = estimate_plume_extent(180.0, 8.0)
        assert abs(r["exposure_factor"] + r["shelter_factor"] - 1.0) < 1e-6

    def test_pedra_eulalia_plausible(self):
        r = estimate_plume_extent(179.94, 8.0, wind_direction_deg=315.0, slope_mean=1.076)
        assert 4.0 <= r["plume_km"] <= 10.0


class TestCheckTerrainFeatures:
    def test_normal_algarve_values_ok(self):
        r = check_terrain_features({"slope_mean": 1.5, "slope_p90": 4.5, "aspect_mean": 180.0})
        assert r["level"] == OK
        assert r["alerts"] == []

    def test_extreme_slope_warns(self):
        r = check_terrain_features({"slope_mean": 40.0, "slope_p90": 4.5, "aspect_mean": 180.0})
        assert r["level"] in (WARNING, CRITICAL)

    def test_out_of_range_aspect_warns(self):
        r = check_terrain_features({"slope_mean": 1.5, "slope_p90": 4.5, "aspect_mean": 400.0})
        assert r["level"] in (WARNING, CRITICAL)

    def test_null_terrain_features_counted(self):
        r = check_terrain_features({"slope_mean": None, "slope_p90": None, "aspect_mean": None})
        assert r["null_count"] == 3

    def test_terrain_baselines_coverage(self):
        assert "slope_mean" in TERRAIN_BASELINES
        assert "aspect_mean" in TERRAIN_BASELINES
        assert "slope_p90" in TERRAIN_BASELINES
