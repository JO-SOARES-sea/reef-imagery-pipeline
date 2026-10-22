"""Tests for src/emodnet_bathymetry.py — TRI computation and habitat masking.

All tests are offline (compute_tri and build_habitat_mask are pure rasterio/numpy).
fetch_depth hits the EMODnet WCS endpoint and is marked @network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_depth_tif(path: Path, data: np.ndarray,
                     bbox=(-8.4, 36.9, -8.1, 37.1)) -> Path:
    """Write a synthetic EMODnet-style depth GeoTIFF (negative = below MSL)."""
    h, w = data.shape
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=h, width=w, count=1, dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=from_bounds(*bbox, w, h),
        nodata=-9999.0,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# compute_tri
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTri:
    def test_flat_grid_gives_zero_tri(self, tmp_path):
        """Perfectly flat seafloor → every neighbour difference is 0 → TRI = 0."""
        from src.emodnet_bathymetry import compute_tri
        depth = np.full((20, 20), -10.0, dtype=np.float32)
        depth_p = _write_depth_tif(tmp_path / "depth.tif", depth)
        tri_p = compute_tri(depth_p, tmp_path / "tri.tif")
        with rasterio.open(tri_p) as src:
            tri = src.read(1)
        valid = tri[tri != -9999.0]
        assert np.allclose(valid, 0.0, atol=1e-6)

    def test_sloped_grid_gives_positive_tri(self, tmp_path):
        """Linearly sloped bathymetry → constant positive TRI."""
        from src.emodnet_bathymetry import compute_tri
        x = np.linspace(-5, -25, 30)
        depth = np.tile(x, (30, 1)).astype(np.float32)  # 30×30 slope
        depth_p = _write_depth_tif(tmp_path / "depth.tif", depth)
        tri_p = compute_tri(depth_p, tmp_path / "tri.tif")
        with rasterio.open(tri_p) as src:
            tri = src.read(1)
        valid = tri[tri != -9999.0]
        assert valid.min() > 0.0

    def test_output_file_written(self, tmp_path):
        from src.emodnet_bathymetry import compute_tri
        depth = np.full((10, 10), -8.0)
        depth_p = _write_depth_tif(tmp_path / "d.tif", depth)
        out = tmp_path / "tri_out.tif"
        compute_tri(depth_p, out)
        assert out.exists()

    def test_nodata_propagated(self, tmp_path):
        """Nodata cells must produce nodata in TRI output."""
        from src.emodnet_bathymetry import compute_tri
        depth = np.full((10, 10), -10.0, dtype=np.float32)
        depth[5, 5] = -9999.0   # nodata pixel
        depth_p = _write_depth_tif(tmp_path / "depth.tif", depth)
        tri_p = compute_tri(depth_p, tmp_path / "tri.tif")
        with rasterio.open(tri_p) as src:
            tri = src.read(1)
        assert tri[5, 5] == pytest.approx(-9999.0)

    def test_rough_grid_has_higher_tri_than_smooth(self, tmp_path):
        """Rougher bathymetry must produce higher mean TRI."""
        from src.emodnet_bathymetry import compute_tri
        smooth = np.linspace(-5, -10, 400).reshape(20, 20).astype(np.float32)
        rough  = smooth + np.random.default_rng(0).uniform(-2, 2, (20, 20)).astype(np.float32)

        p_smooth = _write_depth_tif(tmp_path / "smooth.tif", smooth)
        p_rough  = _write_depth_tif(tmp_path / "rough.tif",  rough)
        t_smooth = compute_tri(p_smooth, tmp_path / "tri_s.tif")
        t_rough  = compute_tri(p_rough,  tmp_path / "tri_r.tif")

        def _mean_tri(p):
            with rasterio.open(p) as s:
                d = s.read(1)
            v = d[d != -9999.0]
            return v.mean()

        assert _mean_tri(t_rough) > _mean_tri(t_smooth)


# ─────────────────────────────────────────────────────────────────────────────
# build_habitat_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHabitatMask:
    def _build_pair(self, tmp_path, depth_val: float, tri_val: float):
        """Create matched depth+TRI rasters with a single uniform value each."""
        depth = np.full((10, 10), depth_val, dtype=np.float32)
        tri   = np.full((10, 10), tri_val,   dtype=np.float32)
        dp = _write_depth_tif(tmp_path / "depth.tif", depth)
        tp = _write_depth_tif(tmp_path / "tri.tif",   tri)
        return dp, tp

    def test_shallow_high_tri_classified_reef(self, tmp_path):
        from src.emodnet_bathymetry import DEFAULT_TRI_REEF, build_habitat_mask
        dp, tp = self._build_pair(tmp_path, depth_val=-5.0, tri_val=DEFAULT_TRI_REEF + 0.2)
        result = build_habitat_mask(dp, tp)
        assert result["n_reef_px"] > 0
        assert result["n_sand_px"] == 0

    def test_shallow_low_tri_classified_sand(self, tmp_path):
        from src.emodnet_bathymetry import DEFAULT_TRI_REEF, build_habitat_mask
        dp, tp = self._build_pair(tmp_path, depth_val=-5.0, tri_val=DEFAULT_TRI_REEF - 0.1)
        result = build_habitat_mask(dp, tp)
        assert result["n_sand_px"] > 0
        assert result["n_reef_px"] == 0

    def test_deep_pixel_outside_optical_window(self, tmp_path):
        """Depth > max_depth_m (too deep for Sentinel-2) → classified as 0, not reef/sand."""
        from src.emodnet_bathymetry import build_habitat_mask
        dp, tp = self._build_pair(tmp_path, depth_val=-30.0, tri_val=1.0)
        result = build_habitat_mask(dp, tp, max_depth_m=25.0)
        assert result["n_reef_px"] == 0
        assert result["n_sand_px"] == 0
        assert (result["mask"] == 0).any()

    def test_reef_frac_sums_with_sand_frac(self, tmp_path):
        """reef_frac + sand_frac must equal 1.0 when all pixels are in the optical window."""
        from src.emodnet_bathymetry import DEFAULT_TRI_REEF, build_habitat_mask
        depth = np.where(
            np.arange(100).reshape(10, 10) < 50,
            -5.0,    # shallow (reef or sand)
            -5.0,
        ).astype(np.float32)
        tri = np.where(
            np.arange(100).reshape(10, 10) < 50,
            DEFAULT_TRI_REEF + 0.5,   # reef
            DEFAULT_TRI_REEF - 0.1,   # sand
        ).astype(np.float32)
        dp = _write_depth_tif(tmp_path / "d.tif", depth)
        tp = _write_depth_tif(tmp_path / "t.tif", tri)
        result = build_habitat_mask(dp, tp)
        assert result["reef_frac"] + result["sand_frac"] == pytest.approx(1.0, abs=1e-5)

    def test_mask_shape_matches_input(self, tmp_path):
        from src.emodnet_bathymetry import build_habitat_mask
        dp, tp = self._build_pair(tmp_path, depth_val=-10.0, tri_val=0.5)
        result = build_habitat_mask(dp, tp)
        assert result["mask"].shape == (10, 10)

    def test_mask_values_are_valid_classes(self, tmp_path):
        """All mask values must be in {-1, 0, 1, 2}."""
        from src.emodnet_bathymetry import build_habitat_mask
        depth = np.random.uniform(-30, 5, (15, 15)).astype(np.float32)
        tri   = np.random.uniform(0, 2, (15, 15)).astype(np.float32)
        dp = _write_depth_tif(tmp_path / "d.tif", depth)
        tp = _write_depth_tif(tmp_path / "t.tif", tri)
        result = build_habitat_mask(dp, tp)
        unique = set(np.unique(result["mask"]).tolist())
        assert unique.issubset({-1, 0, 1, 2})


# ─────────────────────────────────────────────────────────────────────────────
# sample_depth_at_point
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleDepthAtPoint:
    def test_returns_correct_depth(self, tmp_path):
        from src.emodnet_bathymetry import sample_depth_at_point
        depth = np.full((50, 50), -12.0, dtype=np.float32)
        dp = _write_depth_tif(tmp_path / "depth.tif", depth,
                              bbox=(-8.4, 36.9, -8.1, 37.1))
        # Centre of the raster
        val = sample_depth_at_point(dp, lon=-8.25, lat=37.0)
        assert val == pytest.approx(-12.0, abs=0.1)

    def test_returns_none_for_nodata(self, tmp_path):
        from src.emodnet_bathymetry import sample_depth_at_point
        depth = np.full((50, 50), -9999.0, dtype=np.float32)  # all nodata
        dp = _write_depth_tif(tmp_path / "depth.tif", depth,
                              bbox=(-8.4, 36.9, -8.1, 37.1))
        val = sample_depth_at_point(dp, lon=-8.25, lat=37.0)
        assert val is None

    def test_returns_none_outside_bbox(self, tmp_path):
        from src.emodnet_bathymetry import sample_depth_at_point
        depth = np.full((50, 50), -5.0, dtype=np.float32)
        dp = _write_depth_tif(tmp_path / "depth.tif", depth,
                              bbox=(-8.4, 36.9, -8.1, 37.1))
        val = sample_depth_at_point(dp, lon=0.0, lat=0.0)  # far outside
        assert val is None
