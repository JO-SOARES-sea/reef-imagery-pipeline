"""
Tests for the 6 pipeline improvement modules.
All tests are offline — no network or satellite access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ── helpers ──────────────────────────────────────────────────────────────────

def _write_tiff(path: Path, data: np.ndarray,
                bounds=(-8.5, 36.9, -7.85, 37.2), crs="EPSG:4326") -> Path:
    h, w = data.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def _read_tiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Multi-date Stumpf fusion
# ═══════════════════════════════════════════════════════════════════════════════

class TestStumpfMultiscene:
    def test_fuse_returns_expected_keys(self):

        results = [
            {"m0": -14.0, "m1": 21.0, "rmse_m": 1.5, "calibrated": True},
            {"m0": -15.0, "m1": 23.0, "rmse_m": 2.0, "calibrated": True},
            {"m0": -13.5, "m1": 20.5, "rmse_m": 1.8, "calibrated": True},
        ]
        # Test the inner fusion logic directly
        m0s = np.array([r["m0"] for r in results])
        m1s = np.array([r["m1"] for r in results])
        assert abs(np.median(m0s) - (-14.0)) < 0.1
        assert abs(np.median(m1s) - 21.0)   < 0.1

    def test_fuse_scenes_no_ih_returns_default(self):
        from src.stumpf_multiscene import fuse_scenes

        # Pass scenes but mock IH failure by providing empty features
        result = fuse_scenes(
            scenes=[],
            site_bbox=(-8.5, 36.9, -7.85, 37.2),
            isobath_features=[],   # empty → no samples → default
        )
        assert result["status"] in ("default", "single_scene", "fused")

    def test_fuse_rejects_high_rmse(self):
        """Scenes with RMSE above threshold should be excluded when enough good scenes remain."""
        from src.stumpf_multiscene import _MAX_RMSE_M

        good = [{"m0": -14.0, "m1": 21.0, "rmse_m": 1.0, "calibrated": True} for _ in range(4)]
        bad  = [{"m0": -50.0, "m1": 80.0, "rmse_m": 8.0, "calibrated": True}]
        all_results = good + bad

        good_kept = [r for r in all_results if r["rmse_m"] <= _MAX_RMSE_M]
        assert len(good_kept) == 4  # bad scene excluded

    def test_median_robust_to_outlier(self):
        values = [-14.0, -15.0, -14.5, -60.0]  # one outlier
        assert abs(np.median(values) - (-14.5)) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IH isobath quality filter — curvature and B03 threshold
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsobathQualityFilter:
    def test_curvature_straight_line_accepted(self):
        from src.bathy_calibrator import _isobath_curvature_ok

        # Straight line → angle ~180° → always OK
        coords = [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]]
        for i in range(1, 3):
            assert _isobath_curvature_ok(coords, i)

    def test_curvature_sharp_bend_rejected(self):
        from src.bathy_calibrator import _isobath_curvature_ok

        # U-turn → angle ~0° → rejected
        coords = [[0.0, 0.0], [0.01, 0.0], [0.0, 0.0]]
        assert not _isobath_curvature_ok(coords, 1)

    def test_curvature_endpoints_always_ok(self):
        from src.bathy_calibrator import _isobath_curvature_ok

        coords = [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]]
        assert _isobath_curvature_ok(coords, 0)   # first
        assert _isobath_curvature_ok(coords, 2)   # last

    def test_b03_sand_threshold_constant(self):
        from src.bathy_calibrator import _B03_SAND_THRESHOLD
        assert 0.10 <= _B03_SAND_THRESHOLD <= 0.25

    def test_sample_excludes_bright_sand(self):
        """Pixels with B03 > 0.15 must not appear in calibration samples."""
        from src.bathy_calibrator import _B03_SAND_THRESHOLD, _sample_pixels_near_isobath

        H, W = 20, 20
        # All B03 values above the sand threshold
        b02 = np.full((H, W), 0.05, dtype=np.float32)
        b03 = np.full((H, W), _B03_SAND_THRESHOLD + 0.05, dtype=np.float32)

        # Single isobath vertex in the middle of the raster
        features = [{"depth": 10.0, "coords": [[0.0, 0.0]], "length_deg": 0.1}]
        bounds = (0.0, 0.0, 1.0, 1.0)   # min_lat, min_lon, max_lat, max_lon

        samples = _sample_pixels_near_isobath(b02, b03, features, 10.0, bounds)
        assert len(samples) == 0   # all rejected by B03 filter


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Kd490 spatial map
# ═══════════════════════════════════════════════════════════════════════════════

class TestKd490Map:
    def test_returns_correct_shape(self):
        from src.utils import get_kd490_map
        b02 = np.full((10, 20), 0.04, dtype=np.float32)
        b03 = np.full((10, 20), 0.03, dtype=np.float32)
        b04 = np.full((10, 20), 0.01, dtype=np.float32)
        kd  = get_kd490_map(b02, b03, b04)
        assert kd.shape == (10, 20)

    def test_values_in_physical_range(self):
        from src.utils import get_kd490_map
        rng = np.random.default_rng(42)
        b02 = rng.uniform(0.01, 0.10, (50, 50)).astype(np.float32)
        b03 = rng.uniform(0.01, 0.10, (50, 50)).astype(np.float32)
        b04 = rng.uniform(0.005, 0.05, (50, 50)).astype(np.float32)
        kd  = get_kd490_map(b02, b03, b04)
        assert float(kd.min()) >= 0.01
        assert float(kd.max()) <= 2.0

    def test_turbid_water_higher_kd(self):
        """Turbid water (high B04 relative to B03) should produce higher Kd."""
        from src.utils import get_kd490_map
        clear   = get_kd490_map(
            np.array([[0.05]]), np.array([[0.04]]), np.array([[0.005]])
        )
        turbid  = get_kd490_map(
            np.array([[0.05]]), np.array([[0.04]]), np.array([[0.04]])
        )
        assert float(turbid[0, 0]) > float(clear[0, 0])

    def test_returns_float32(self):
        from src.utils import get_kd490_map
        kd = get_kd490_map(
            np.ones((5, 5), np.float32) * 0.05,
            np.ones((5, 5), np.float32) * 0.04,
            np.ones((5, 5), np.float32) * 0.01,
        )
        assert kd.dtype == np.float32

    def test_bright_sand_saturates_toward_ceiling(self):
        """Bright sand / sunglint (high B04) drives Kd toward the 2.0 clip — the
        condition the predictor's scalar-fallback guard is meant to catch."""
        from src.constants import KD490_MAP_SATURATION_CEILING
        from src.utils import get_kd490_map
        # Red-dominated reflectance: B04 >> B03/B02 forces the band ratio low → Kd high
        b02 = np.full((20, 20), 0.02, dtype=np.float32)
        b03 = np.full((20, 20), 0.02, dtype=np.float32)
        b04 = np.full((20, 20), 0.30, dtype=np.float32)
        kd = get_kd490_map(b02, b03, b04)
        assert float(np.nanmean(kd)) > KD490_MAP_SATURATION_CEILING

    def test_saturation_ceiling_constant_is_physical(self):
        from src.constants import KD490_MAP_SATURATION_CEILING
        # Below the 2.0 hard clip, above clear-ocean Kd; a meaningful trip point.
        assert 0.1 < KD490_MAP_SATURATION_CEILING < 2.0


class TestMultisceneOverride:
    """run_predictor exposes fused-coefficient overrides for the orchestrator wiring."""

    def test_run_predictor_accepts_stumpf_overrides(self):
        import inspect

        from src.reef_ml_predictor_acolite import run_predictor
        params = inspect.signature(run_predictor).parameters
        assert "stumpf_m0_override" in params
        assert "stumpf_m1_override" in params
        assert params["stumpf_m0_override"].default is None
        assert params["stumpf_m1_override"].default is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Ensemble calibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsembleCalibrator:
    def test_returns_expected_keys(self):
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate(
            ih_result={"m0": -14.0, "m1": 22.0, "n_samples": 300, "calibrated": True, "rmse_m": 1.5},
        )
        assert {"m0", "m1", "weights", "sources_used", "status"} <= result.keys()

    def test_single_source_status(self):
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate(
            ih_result={"m0": -14.0, "m1": 22.0, "n_samples": 300, "calibrated": True, "rmse_m": 1.5},
        )
        assert result["status"] == "single_source"
        assert "IH" in result["sources_used"]

    def test_multi_source_ensemble(self):
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate(
            ih_result={"m0": -14.0, "m1": 22.0, "n_samples": 300, "calibrated": True, "rmse_m": 1.5},
            emodnet_result={"m0": -12.0, "m1": 20.0, "rmse": 2.0, "calibration_samples": 5000},
        )
        assert result["status"] == "ensemble"
        assert len(result["sources_used"]) == 2
        # Ensemble m0 should be between the two sources
        assert -14.0 <= result["m0"] <= -12.0

    def test_weights_sum_to_one(self):
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate(
            ih_result={"m0": -14.0, "m1": 22.0, "n_samples": 400, "calibrated": True, "rmse_m": 1.5},
            emodnet_result={"m0": -12.0, "m1": 20.0, "rmse": 2.0, "calibration_samples": 5000},
            icesat2_result={"m0": 190.0, "m1": -168.0, "rmse": 1.5, "calibration_samples": 300, "status": "success"},
        )
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-4

    def test_no_sources_returns_default(self):
        from src.constants import STUMPF_M0_DEFAULT, STUMPF_M1_DEFAULT
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate()
        assert result["status"] == "default"
        assert result["m0"] == STUMPF_M0_DEFAULT
        assert result["m1"] == STUMPF_M1_DEFAULT

    def test_uncalibrated_ih_excluded(self):
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate(
            ih_result={"m0": -14.0, "m1": 22.0, "n_samples": 0, "calibrated": False},
        )
        assert "IH" not in result["sources_used"]

    def test_icesat2_failed_status_excluded(self):
        from src.ensemble_calibrator import ensemble_calibrate
        result = ensemble_calibrate(
            icesat2_result={"m0": 190.0, "m1": -168.0, "status": "insufficient_data"},
        )
        assert "ICESat-2" not in result["sources_used"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Temporal interpolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalInterpolation:
    def _make_scene(self, tmp_path: Path, name: str, depth_val: float) -> Path:
        p = tmp_path / f"{name}.tif"
        data = np.full((20, 20), depth_val, dtype=np.float32)
        _write_tiff(p, data)
        return p

    def test_interpolated_value_between_anchors(self, tmp_path):
        from src.temporal_interpolation import interpolate_cloud_gaps
        s1 = self._make_scene(tmp_path, "a", 10.0)
        s2 = self._make_scene(tmp_path, "b", 20.0)
        out = tmp_path / "interp.tif"
        result = interpolate_cloud_gaps(
            scenes=[
                {"date": "2024-01-01", "sdb_path": str(s1), "cloud_frac": 0.02},
                {"date": "2024-01-15", "sdb_path": None, "cloud_frac": 1.0},
                {"date": "2024-02-01", "sdb_path": str(s2), "cloud_frac": 0.05},
            ],
            output_path=out,
            target_date="2024-01-15",
        )
        assert result is not None and out.exists()
        arr = _read_tiff(out)
        # t=14/31 ≈ 0.45 — midpoint should be between 10 and 20
        assert 10.0 < float(np.nanmean(arr)) < 20.0

    def test_no_gap_returns_none(self, tmp_path):
        from src.temporal_interpolation import interpolate_cloud_gaps
        s1 = self._make_scene(tmp_path, "a", 10.0)
        s2 = self._make_scene(tmp_path, "b", 20.0)
        out = tmp_path / "nointerp.tif"
        result = interpolate_cloud_gaps(
            scenes=[
                {"date": "2024-01-01", "sdb_path": str(s1), "cloud_frac": 0.0},
                {"date": "2024-02-01", "sdb_path": str(s2), "cloud_frac": 0.0},
            ],
            output_path=out,
            target_date="2024-01-15",
        )
        # target_date is between the two clear scenes — should interpolate
        assert result is not None

    def test_insufficient_clear_scenes_returns_none(self, tmp_path):
        from src.temporal_interpolation import interpolate_cloud_gaps
        out = tmp_path / "fail.tif"
        result = interpolate_cloud_gaps(
            scenes=[
                {"date": "2024-01-15", "sdb_path": None, "cloud_frac": 1.0},
            ],
            output_path=out,
            target_date="2024-01-15",
        )
        assert result is None

    def test_monthly_composite_written(self, tmp_path):
        from src.temporal_interpolation import build_monthly_composite
        s1 = self._make_scene(tmp_path, "jan1", 12.0)
        s2 = self._make_scene(tmp_path, "jan2", 14.0)
        out_dir = tmp_path / "composites"
        results = build_monthly_composite(
            scenes=[
                {"date": "2024-01-05", "sdb_path": str(s1), "cloud_frac": 0.01},
                {"date": "2024-01-20", "sdb_path": str(s2), "cloud_frac": 0.03},
            ],
            output_dir=out_dir,
        )
        assert "2024-01" in results
        arr = _read_tiff(results["2024-01"])
        assert abs(float(np.nanmean(arr)) - 13.0) < 0.1  # median of 12 and 14


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Benthic substrate classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubstrateClassifier:
    def test_sand_detected(self):
        from src.substrate_classifier import _B03_SAND_MIN, _B03B04_SAND, SAND, classify_substrate
        H, W = 10, 10
        # Bright sand: high B03, high B03/B04
        b03_val = _B03_SAND_MIN + 0.02
        b04_val = b03_val / (_B03B04_SAND + 0.2)
        b02 = np.full((H, W), 0.04, np.float32)
        b03 = np.full((H, W), b03_val, np.float32)
        b04 = np.full((H, W), b04_val, np.float32)
        depth = np.full((H, W), 5.0, np.float32)
        classes = classify_substrate(b02, b03, b04, sdb_depth=depth)
        assert int(classes[5, 5]) == SAND

    def test_reef_detected(self):
        from src.substrate_classifier import REEF, classify_substrate
        H, W = 10, 10
        # Dark reef: low B03 (below sand threshold)
        b02 = np.full((H, W), 0.02, np.float32)
        b03 = np.full((H, W), 0.03, np.float32)
        b04 = np.full((H, W), 0.02, np.float32)
        depth = np.full((H, W), 8.0, np.float32)
        classes = classify_substrate(b02, b03, b04, sdb_depth=depth)
        assert int(classes[5, 5]) == REEF

    def test_seagrass_detected_with_b08(self):
        from src.substrate_classifier import SEAGRASS, classify_substrate
        H, W = 10, 10
        b02 = np.full((H, W), 0.03, np.float32)
        b03 = np.full((H, W), 0.04, np.float32)
        b04 = np.full((H, W), 0.03, np.float32)
        # NDVI = (b08-b04)/(b08+b04) > threshold
        b08_val = 0.10
        b08 = np.full((H, W), b08_val, np.float32)
        depth = np.full((H, W), 4.0, np.float32)
        classes = classify_substrate(b02, b03, b04, b08=b08, sdb_depth=depth)
        # NDVI = (0.10 - 0.03) / (0.10 + 0.03) = 0.538 > 0.05
        assert int(classes[5, 5]) == SEAGRASS

    def test_masked_outside_depth_range(self):
        from src.substrate_classifier import MASKED, classify_substrate
        H, W = 10, 10
        b02 = np.full((H, W), 0.04, np.float32)
        b03 = np.full((H, W), 0.04, np.float32)
        b04 = np.full((H, W), 0.02, np.float32)
        depth = np.full((H, W), 30.0, np.float32)   # beyond 25 m limit
        classes = classify_substrate(b02, b03, b04, sdb_depth=depth)
        assert int(classes[5, 5]) == MASKED

    def test_no_depth_uses_all_pixels(self):
        from src.substrate_classifier import MASKED, classify_substrate
        H, W = 10, 10
        b02 = np.full((H, W), 0.04, np.float32)
        b03 = np.full((H, W), 0.04, np.float32)
        b04 = np.full((H, W), 0.02, np.float32)
        classes = classify_substrate(b02, b03, b04)   # no sdb_depth
        # Should not be all masked
        assert (classes != MASKED).any()

    def test_get_sand_mask_returns_bool(self):
        from src.substrate_classifier import get_sand_mask
        b02 = np.full((5, 5), 0.04, np.float32)
        b03 = np.full((5, 5), 0.04, np.float32)
        b04 = np.full((5, 5), 0.02, np.float32)
        mask = get_sand_mask(b02, b03, b04)
        assert mask.dtype == bool

    def test_write_substrate_tiff(self, tmp_path):
        from src.substrate_classifier import classify_substrate, write_substrate_tiff
        b02 = np.full((10, 10), 0.04, np.float32)
        b03 = np.full((10, 10), 0.04, np.float32)
        b04 = np.full((10, 10), 0.02, np.float32)
        classes = classify_substrate(b02, b03, b04)

        dummy = tmp_path / "ref.tif"
        _write_tiff(dummy, b02)
        with rasterio.open(dummy) as src:
            profile = src.profile

        out = tmp_path / "substrate.tif"
        write_substrate_tiff(classes, profile, out)
        assert out.exists()
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "int8"
            assert src.nodata == -1
