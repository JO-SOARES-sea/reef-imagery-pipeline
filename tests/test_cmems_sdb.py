"""
Tests for src/cmems_sdb.py and the updated stumpf_emodnet_calibration.py.

All tests are offline (no network). CMEMS fetch is mocked via tmp GeoTIFFs.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

# ─── helpers ────────────────────────────────────────────────────────────────

def _write_tiff(path: Path, data: np.ndarray, bounds=(-8.5, 36.9, -7.85, 37.2), crs="EPSG:4326"):
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


# ─── blend_depth_priors ─────────────────────────────────────────────────────

class TestBlendDepthPriors:
    from src.cmems_sdb import blend_depth_priors

    def test_no_cmems_returns_emodnet_copy(self):
        from src.cmems_sdb import blend_depth_priors
        e = np.array([[-10., -15.], [np.nan, -5.]], dtype=np.float32)
        result = blend_depth_priors(e, None)
        np.testing.assert_array_equal(result, e)
        assert result is not e  # must be a copy

    def test_both_valid_weighted_blend(self):
        from src.cmems_sdb import blend_depth_priors
        e = np.array([[-10.]], dtype=np.float32)
        c = np.array([[-20.]], dtype=np.float32)
        result = blend_depth_priors(e, c, cmems_weight=0.4)
        expected = -10.0 * 0.6 + -20.0 * 0.4  # -14.0
        assert abs(result[0, 0] - expected) < 1e-4

    def test_emodnet_only_where_cmems_nan(self):
        from src.cmems_sdb import blend_depth_priors
        e = np.array([[-10., -15.]], dtype=np.float32)
        c = np.array([[np.nan, -20.]], dtype=np.float32)
        result = blend_depth_priors(e, c, cmems_weight=0.4)
        assert result[0, 0] == -10.0         # EMODnet only
        assert abs(result[0, 1] - (-15.*0.6 + -20.*0.4)) < 1e-4   # blended

    def test_cmems_only_where_emodnet_nan(self):
        from src.cmems_sdb import blend_depth_priors
        e = np.array([[np.nan, -15.]], dtype=np.float32)
        c = np.array([[-8., np.nan]], dtype=np.float32)
        result = blend_depth_priors(e, c)
        assert result[0, 0] == -8.0    # CMEMS fills gap
        assert result[0, 1] == -15.0   # EMODnet only

    def test_all_nan_stays_nan(self):
        from src.cmems_sdb import blend_depth_priors
        e = np.full((3, 3), np.nan, dtype=np.float32)
        c = np.full((3, 3), np.nan, dtype=np.float32)
        result = blend_depth_priors(e, c)
        assert np.all(np.isnan(result))

    def test_custom_weight_enforced(self):
        from src.cmems_sdb import blend_depth_priors
        e = np.array([[-10.]], dtype=np.float32)
        c = np.array([[-10.]], dtype=np.float32)
        # Same value, any weight → same result
        result = blend_depth_priors(e, c, cmems_weight=0.7)
        assert abs(result[0, 0] - (-10.0)) < 1e-4


# ─── reproject_cmems_to_s2 ──────────────────────────────────────────────────

class TestReprojectCmemsToS2:
    def test_output_matches_s2_grid(self, tmp_path):
        from src.cmems_sdb import reproject_cmems_to_s2
        bounds = (-8.5, 36.9, -7.85, 37.2)

        # S2 reference: 50×100 grid (10m-like)
        s2_ref = tmp_path / "s2_ref.tif"
        _write_tiff(s2_ref, np.ones((50, 100), dtype=np.float32), bounds)

        # CMEMS 100m-like: 5×10 grid
        cmems_raw = tmp_path / "cmems.tif"
        data = np.full((5, 10), -12.0, dtype=np.float32)
        data[0, 0] = np.nan
        _write_tiff(cmems_raw, data, bounds)

        out = tmp_path / "cmems_10m.tif"
        result_path = reproject_cmems_to_s2(cmems_raw, s2_ref, out)

        assert result_path == out
        arr = _read_tiff(out)
        assert arr.shape == (50, 100)

    def test_output_crs_matches_reference(self, tmp_path):
        from src.cmems_sdb import reproject_cmems_to_s2
        bounds = (-8.5, 36.9, -7.85, 37.2)
        s2_ref = tmp_path / "s2.tif"
        _write_tiff(s2_ref, np.ones((20, 40)), bounds)
        cmems = tmp_path / "c.tif"
        _write_tiff(cmems, np.full((5, 10), -8.0), bounds)

        out = tmp_path / "out.tif"
        reproject_cmems_to_s2(cmems, s2_ref, out)

        with rasterio.open(out) as dst, rasterio.open(s2_ref) as ref:
            assert dst.crs == ref.crs
            assert dst.transform == ref.transform


# ─── calibrate_stumpf_vs_emodnet (updated signature) ────────────────────────

class TestCalibrateStumpfUpdated:
    """Offline tests using synthetic rasters."""

    def _make_scene(self, tmp_path, depth_min=-20., depth_max=-5., nx=200, ny=200):
        bounds = (-8.5, 36.9, -7.85, 37.2)
        rng = np.random.default_rng(42)

        # Synthetic depth prior (negative = below sea)
        depth = rng.uniform(depth_min, depth_max, (ny, nx)).astype(np.float32)
        emod_path = tmp_path / "emodnet.tif"
        _write_tiff(emod_path, depth, bounds)

        # Synthetic S2 bands: B02 ∝ depth-dependent reflectance + noise
        b02 = (0.05 + 0.01 * (-depth) + rng.normal(0, 0.002, (ny, nx))).clip(0.001).astype(np.float32)
        b03 = (0.04 + 0.008 * (-depth) + rng.normal(0, 0.002, (ny, nx))).clip(0.001).astype(np.float32)
        b02_path = tmp_path / "B02.tif"
        b03_path = tmp_path / "B03.tif"
        _write_tiff(b02_path, b02, bounds)
        _write_tiff(b03_path, b03, bounds)

        return b02_path, b03_path, emod_path, depth

    def test_emodnet_only_returns_expected_keys(self, tmp_path):
        from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet
        b02, b03, emod, _ = self._make_scene(tmp_path)
        out = tmp_path / "sdb.tif"
        result = calibrate_stumpf_vs_emodnet(str(b02), str(b03), str(emod), str(out))
        assert {"m0", "m1", "rmse", "calibration_samples", "depth_prior_source"} <= result.keys()
        assert result["depth_prior_source"] == "EMODnet"
        assert result["calibration_samples"] >= 100

    def test_cmems_path_changes_source_label(self, tmp_path):
        from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet
        b02, b03, emod, depth = self._make_scene(tmp_path)

        # CMEMS prior: slight offset from EMODnet
        cmems_data = (depth + np.random.default_rng(7).normal(0, 0.5, depth.shape)).astype(np.float32)
        cmems_path = tmp_path / "cmems.tif"
        _write_tiff(cmems_path, cmems_data, (-8.5, 36.9, -7.85, 37.2))

        out = tmp_path / "sdb_cmems.tif"
        result = calibrate_stumpf_vs_emodnet(
            str(b02), str(b03), str(emod), str(out), cmems_10m_path=str(cmems_path)
        )
        assert result["depth_prior_source"] == "EMODnet+CMEMS"

    def test_output_tiff_written(self, tmp_path):
        from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet
        b02, b03, emod, _ = self._make_scene(tmp_path)
        out = tmp_path / "sdb_out.tif"
        calibrate_stumpf_vs_emodnet(str(b02), str(b03), str(emod), str(out))
        assert out.exists()
        arr = _read_tiff(out)
        assert arr.shape == (200, 200)

    def test_rmse_finite_and_positive(self, tmp_path):
        from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet
        b02, b03, emod, _ = self._make_scene(tmp_path)
        out = tmp_path / "sdb.tif"
        result = calibrate_stumpf_vs_emodnet(str(b02), str(b03), str(emod), str(out))
        assert np.isfinite(result["rmse"])
        assert result["rmse"] >= 0.0

    def test_extended_depth_window(self, tmp_path):
        from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet
        b02, b03, emod, _ = self._make_scene(tmp_path, depth_min=-34., depth_max=-5.)
        out = tmp_path / "sdb_deep.tif"
        result = calibrate_stumpf_vs_emodnet(
            str(b02), str(b03), str(emod), str(out), depth_max_m=34.0
        )
        assert result["depth_training_window_m"] == [5.0, 34.0]
        assert result["calibration_samples"] >= 100


# ─── ICESat-2 GeoJSON output ────────────────────────────────────────────────

class TestIcesat2GeoJSON:
    GEOJSON = Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson")

    @pytest.mark.skipif(
        not Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson").exists(),
        reason="ICESat-2 GeoJSON not yet generated",
    )
    def test_valid_geojson_structure(self):
        fc = json.loads(self.GEOJSON.read_text())
        assert fc["type"] == "FeatureCollection"
        assert isinstance(fc["features"], list)
        assert len(fc["features"]) > 0

    @pytest.mark.skipif(
        not Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson").exists(),
        reason="ICESat-2 GeoJSON not yet generated",
    )
    def test_required_properties_present(self):
        fc = json.loads(self.GEOJSON.read_text())
        required = {"confidence_score", "validation_class", "depth_mean_m",
                    "depth_std_m", "photon_count", "source", "depth_zone"}
        for feat in fc["features"][:10]:
            props = feat["properties"]
            missing = required - props.keys()
            assert not missing, f"Missing properties: {missing}"

    @pytest.mark.skipif(
        not Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson").exists(),
        reason="ICESat-2 GeoJSON not yet generated",
    )
    def test_scores_in_range(self):
        fc = json.loads(self.GEOJSON.read_text())
        for feat in fc["features"]:
            s = feat["properties"]["confidence_score"]
            assert 0 <= s <= 100, f"Score out of range: {s}"

    @pytest.mark.skipif(
        not Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson").exists(),
        reason="ICESat-2 GeoJSON not yet generated",
    )
    def test_coordinates_in_algarve_bbox(self):
        fc = json.loads(self.GEOJSON.read_text())
        # Cluster centroids are snapped to 0.015° grid, so allow one cell margin
        MARGIN = 0.02
        W, S, E, N = -8.5 - MARGIN, 36.9 - MARGIN, -7.85 + MARGIN, 37.2 + MARGIN
        for feat in fc["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            assert W <= lon <= E, f"lon {lon} outside Algarve bbox (with margin)"
            assert S <= lat <= N, f"lat {lat} outside Algarve bbox (with margin)"

    @pytest.mark.skipif(
        not Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson").exists(),
        reason="ICESat-2 GeoJSON not yet generated",
    )
    def test_depth_in_target_band(self):
        fc = json.loads(self.GEOJSON.read_text())
        for feat in fc["features"]:
            d = feat["properties"]["depth_mean_m"]
            # Allow ±2m margin for cluster centroid edge effects
            assert 13.0 <= d <= 32.0, f"depth {d}m outside expected 15–30m band"

    @pytest.mark.skipif(
        not Path("outputs/icesat2_deep_survey/candidates_icesat2_deep.geojson").exists(),
        reason="ICESat-2 GeoJSON not yet generated",
    )
    def test_high_confidence_candidates_exist(self):
        fc = json.loads(self.GEOJSON.read_text())
        high = [f for f in fc["features"] if f["properties"]["confidence_score"] >= 70]
        assert len(high) >= 10, f"Expected ≥10 HIGH candidates, got {len(high)}"


# ─── cmems_sdb module imports cleanly ───────────────────────────────────────

def test_cmems_sdb_module_imports():
    from src import cmems_sdb
    assert hasattr(cmems_sdb, "fetch_cmems_sdb")
    assert hasattr(cmems_sdb, "reproject_cmems_to_s2")
    assert hasattr(cmems_sdb, "blend_depth_priors")
    assert hasattr(cmems_sdb, "_DATASET_IDS")
    assert "composite" in cmems_sdb._DATASET_IDS
    assert "wk" in cmems_sdb._DATASET_IDS


def test_fetch_cmems_sdb_returns_none_gracefully(tmp_path, monkeypatch):
    """fetch_cmems_sdb() should return None (not raise) if copernicusmarine fails."""
    import src.cmems_sdb as mod

    # Patch open_dataset to raise so the except-branch is exercised
    def _bad_open_dataset(**kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("copernicusmarine.open_dataset", _bad_open_dataset, raising=False)

    result = mod.fetch_cmems_sdb(
        bbox=(-8.5, 36.9, -7.85, 37.2),
        output_path=tmp_path / "out.tif",
    )
    assert result is None
