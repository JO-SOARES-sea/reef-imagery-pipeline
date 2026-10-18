"""Tests for scripts/reef_detector.py — offline physics and signal-processing functions.

All tests use synthetic numpy arrays; no network, no Sentinel-2 download.
Functions that require live STAC / COG access (load_scene, process_scene,
_resolve_assets) are excluded — mark them @pytest.mark.network when added.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_bounds

# reef_detector lives in scripts/, not src/, so we need the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.reef_detector import (
    _attenuation_ratio,
    _robust_minmax,
    local_stats,
    water_mask,
    measure_clarity,
    GLINT_NIR_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bands(h: int = 40, w: int = 40, *,
           blue: float = 0.05, green: float = 0.04,
           red: float = 0.03, nir: float = 0.005) -> dict:
    """Return synthetic band arrays (constant, clear-water values)."""
    return {
        "blue":  np.full((h, w), blue,  dtype=np.float32),
        "green": np.full((h, w), green, dtype=np.float32),
        "red":   np.full((h, w), red,   dtype=np.float32),
        "nir":   np.full((h, w), nir,   dtype=np.float32),
    }


def _scene(h: int = 40, w: int = 40,
           bbox=(-8.4, 36.9, -8.1, 37.1), **band_kwargs) -> dict:
    """Minimal scene dict with synthetic bands + georeference.

    The transform is built from a lon/lat bbox, so the CRS must be EPSG:4326.
    (Before 2026-10-09 this fixture declared EPSG:32629 with a degree-based transform —
    an inconsistent georeference that the old detector never checked because it
    indexed lon/lat straight into the transform.)
    """
    import rasterio.crs
    return {
        "bands":     _bands(h, w, **band_kwargs),
        "transform": from_bounds(*bbox, w, h),
        "crs":       rasterio.crs.CRS.from_epsg(4326),
        "shape":     (h, w),
    }


# ─────────────────────────────────────────────────────────────────────────────
# _attenuation_ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestAttenuationRatio:
    def test_identical_bands_returns_1(self):
        """Equal variance and covariance → a=0 → ratio=1.0."""
        xi = np.array([1.0, 2.0, 3.0, 4.0])
        assert _attenuation_ratio(xi, xi.copy()) == pytest.approx(1.0)

    def test_zero_covariance_returns_1(self):
        """Orthogonal signals → cov≈0 → fall back to 1.0."""
        xi = np.array([1.0, -1.0, 1.0, -1.0])
        xj = np.array([1.0,  1.0, -1.0, -1.0])
        result = _attenuation_ratio(xi, xj)
        assert result == pytest.approx(1.0)

    def test_ratio_is_always_positive(self):
        """k_i/k_j = a + sqrt(a²+1) — always > 0 regardless of a."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            xi = rng.normal(size=100)
            xj = rng.normal(size=100)
            r = _attenuation_ratio(xi, xj)
            assert r > 0.0, f"Got non-positive ratio {r}"

    def test_returns_float(self):
        xi = np.linspace(1, 10, 50)
        xj = xi + np.random.default_rng(0).normal(0, 0.1, 50)
        assert isinstance(_attenuation_ratio(xi, xj), float)

    def test_strongly_correlated_bands_near_1(self):
        """Blue and green are highly correlated in clear water — ratio ≈ 1."""
        rng = np.random.default_rng(42)
        xi = rng.uniform(1.5, 3.5, 500)
        xj = xi + rng.normal(0, 0.05, 500)   # nearly identical
        ratio = _attenuation_ratio(xi, xj)
        assert 0.8 < ratio < 1.5


# ─────────────────────────────────────────────────────────────────────────────
# _robust_minmax
# ─────────────────────────────────────────────────────────────────────────────

class TestRobustMinmax:
    def test_output_in_0_1(self):
        arr = np.random.default_rng(0).uniform(-10, 10, (50, 50)).astype(np.float32)
        out = _robust_minmax(arr)
        assert out[np.isfinite(out)].min() >= 0.0
        assert out[np.isfinite(out)].max() <= 1.0

    def test_constant_array_returns_nan(self):
        """No contrast → normalised feature undefined (NaN), not a fake zero likelihood (policy 2026-10-09)."""
        arr = np.full((20, 20), 5.0, dtype=np.float32)
        out = _robust_minmax(arr)
        assert np.all(np.isnan(out))

    def test_nan_stays_nan(self):
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = _robust_minmax(arr)
        assert np.isnan(out[2])

    def test_all_nan_returns_all_nan(self):
        arr = np.full((5, 5), np.nan)
        out = _robust_minmax(arr)
        assert np.all(np.isnan(out))

    def test_median_maps_near_half(self):
        """The 50th percentile should land around 0.5."""
        arr = np.arange(100, dtype=np.float32)
        out = _robust_minmax(arr)
        mid = out[50]
        assert 0.4 < mid < 0.6

    def test_monotone_preserving(self):
        """Rank order must be preserved."""
        arr = np.sort(np.random.default_rng(1).uniform(0, 100, 200))
        out = _robust_minmax(arr)
        diffs = np.diff(out[np.isfinite(out)])
        assert np.all(diffs >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# local_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalStats:
    def test_constant_field_zero_std(self):
        """Uniform value → std = 0 everywhere inside the mask."""
        arr  = np.full((30, 30), 3.0, dtype=np.float32)
        mask = np.ones((30, 30), dtype=bool)
        mean, std = local_stats(arr, mask, win=5)
        interior = std[5:-5, 5:-5]  # avoid edge effects
        assert np.allclose(interior[np.isfinite(interior)], 0.0, atol=1e-5)

    def test_constant_field_correct_mean(self):
        arr  = np.full((30, 30), 7.0, dtype=np.float32)
        mask = np.ones((30, 30), dtype=bool)
        mean, _ = local_stats(arr, mask, win=5)
        interior = mean[5:-5, 5:-5]
        assert np.allclose(interior[np.isfinite(interior)], 7.0, atol=1e-4)

    def test_all_masked_out_gives_nan(self):
        arr  = np.ones((20, 20), dtype=np.float32)
        mask = np.zeros((20, 20), dtype=bool)   # nothing valid
        mean, std = local_stats(arr, mask, win=5)
        assert np.all(np.isnan(mean))
        assert np.all(np.isnan(std))

    def test_output_shape_matches_input(self):
        arr  = np.random.rand(25, 35).astype(np.float32)
        mask = arr > 0.3
        mean, std = local_stats(arr, mask, win=7)
        assert mean.shape == arr.shape
        assert std.shape  == arr.shape

    def test_std_non_negative(self):
        arr  = np.random.default_rng(3).uniform(0, 1, (40, 40)).astype(np.float32)
        mask = arr > 0.1
        _, std = local_stats(arr, mask, win=9)
        valid = std[np.isfinite(std)]
        assert np.all(valid >= 0.0)

    def test_heterogeneous_field_positive_std(self):
        """Alternating 0/1 checkerboard → std > 0 in the interior."""
        arr = np.indices((30, 30)).sum(axis=0) % 2 * 1.0
        arr = arr.astype(np.float32)
        mask = np.ones((30, 30), dtype=bool)
        _, std = local_stats(arr, mask, win=11)
        interior = std[11:-11, 11:-11]
        assert np.nanmean(interior) > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# water_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestWaterMask:
    def test_low_nir_clean_water_passes(self):
        bands = _bands(nir=0.005, blue=0.05, green=0.04)
        m = water_mask(bands)
        assert m.all()

    def test_high_nir_rejected(self):
        """NIR above glint threshold → not water."""
        bands = _bands(nir=GLINT_NIR_MAX + 0.01)
        m = water_mask(bands)
        assert not m.any()

    def test_negative_blue_rejected(self):
        bands = _bands(blue=-0.001)
        m = water_mask(bands)
        assert not m.any()

    def test_zero_green_rejected(self):
        bands = _bands(green=0.0)
        m = water_mask(bands)
        assert not m.any()

    def test_depth_mask_filters_deep(self):
        """Pixels deeper than DEPTH_DEEP should be excluded."""
        from scripts.reef_detector import DEPTH_DEEP
        bands = _bands(nir=0.005, blue=0.05, green=0.04)
        depth = np.full((40, 40), DEPTH_DEEP - 5.0, dtype=np.float32)  # too deep
        m = water_mask(bands, depth=depth)
        assert not m.any()

    def test_depth_mask_passes_shallow(self):
        """Pixels within the optical window should pass the depth filter."""
        from scripts.reef_detector import DEPTH_DEEP, DEPTH_SHALLOW
        bands = _bands(nir=0.005, blue=0.05, green=0.04)
        mid_depth = (DEPTH_DEEP + DEPTH_SHALLOW) / 2.0
        depth = np.full((40, 40), mid_depth, dtype=np.float32)
        m = water_mask(bands, depth=depth)
        assert m.all()

    def test_output_is_boolean(self):
        bands = _bands()
        m = water_mask(bands)
        assert m.dtype == bool

    def test_output_shape_matches_bands(self):
        bands = _bands(h=15, w=25)
        m = water_mask(bands)
        assert m.shape == (15, 25)


# ─────────────────────────────────────────────────────────────────────────────
# measure_clarity (synthetic scene with known values)
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureClarity:
    def test_valid_point_inside_scene(self):
        scene = _scene(blue=0.05, green=0.04, nir=0.006)
        # Centre of the bbox
        result = measure_clarity(scene, lon=-8.25, lat=37.0, win=3)
        assert result["valid"] is True
        assert result["blue"]  == pytest.approx(0.05, abs=1e-4)
        assert result["green"] == pytest.approx(0.04, abs=1e-4)
        assert result["nir_glint"] == pytest.approx(0.006, abs=1e-4)

    def test_bottom_contrast_is_blue_minus_green(self):
        scene = _scene(blue=0.06, green=0.04)
        r = measure_clarity(scene, lon=-8.25, lat=37.0)
        assert r["bottom_contrast"] == pytest.approx(0.06 - 0.04, abs=1e-4)

    def test_point_outside_bbox_returns_invalid(self):
        scene = _scene()
        result = measure_clarity(scene, lon=0.0, lat=0.0)
        assert result["valid"] is False
        assert len(result) == 1   # only {"valid": False}
