"""Tests for src/s2_patch_extractor.py — offline-only utilities.

Network-dependent functions (_find_best_scene, extract_patch, build_dataset)
are excluded; they require live STAC access and are marked @network when added.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.s2_patch_extractor import _CLIP_HIGH, _CLIP_LOW, _band_url, _normalise_band

# ─────────────────────────────────────────────────────────────────────────────
# _normalise_band
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseBand:
    def test_output_in_0_1_for_all_bands(self):
        for band in ("B02", "B03", "B04", "B08"):
            lo, hi = _CLIP_LOW[band], _CLIP_HIGH[band]
            data = np.linspace(lo - 100, hi + 100, 200).astype(np.float32)
            out = _normalise_band(data, band)
            assert out.min() >= 0.0, f"{band}: min={out.min()}"
            assert out.max() <= 1.0, f"{band}: max={out.max()}"

    def test_clip_low_maps_to_0(self):
        for band in ("B02", "B03", "B04", "B08"):
            lo = _CLIP_LOW[band]
            out = _normalise_band(np.array([float(lo)]), band)
            assert out[0] == pytest.approx(0.0)

    def test_clip_high_maps_to_1(self):
        for band in ("B02", "B03", "B04", "B08"):
            hi = _CLIP_HIGH[band]
            out = _normalise_band(np.array([float(hi)]), band)
            assert out[0] == pytest.approx(1.0)

    def test_mid_value_maps_to_half(self):
        for band in ("B02", "B03", "B04", "B08"):
            lo, hi = _CLIP_LOW[band], _CLIP_HIGH[band]
            mid = (lo + hi) / 2.0
            out = _normalise_band(np.array([mid]), band)
            assert out[0] == pytest.approx(0.5, abs=1e-5)

    def test_values_below_clip_clamped_to_0(self):
        out = _normalise_band(np.array([-99999.0]), "B02")
        assert out[0] == pytest.approx(0.0)

    def test_values_above_clip_clamped_to_1(self):
        out = _normalise_band(np.array([99999.0]), "B03")
        assert out[0] == pytest.approx(1.0)

    def test_shape_preserved(self):
        data = np.random.randint(0, 3000, (4, 128, 128)).astype(np.float32)
        out = _normalise_band(data, "B02")
        assert out.shape == data.shape

    def test_monotonic_within_range(self):
        """Higher DN must map to higher normalised value."""
        lo, hi = _CLIP_LOW["B02"], _CLIP_HIGH["B02"]
        data = np.linspace(lo, hi, 50)
        out = _normalise_band(data, "B02")
        assert np.all(np.diff(out) >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# _band_url
# ─────────────────────────────────────────────────────────────────────────────

class TestBandUrl:
    def test_earth_search_naming(self):
        """Earth Search uses 'blue', 'green', 'red', 'nir' asset keys."""
        assets = {
            "blue":  "https://example.com/B02.tif",
            "green": "https://example.com/B03.tif",
            "red":   "https://example.com/B04.tif",
            "nir":   "https://example.com/B08.tif",
        }
        assert _band_url(assets, "B02") == "https://example.com/B02.tif"
        assert _band_url(assets, "B03") == "https://example.com/B03.tif"
        assert _band_url(assets, "B04") == "https://example.com/B04.tif"
        assert _band_url(assets, "B08") == "https://example.com/B08.tif"

    def test_direct_band_key(self):
        """Planetary Computer stores assets under the canonical band name."""
        assets = {"B02": "https://pc.com/b02.tif"}
        assert _band_url(assets, "B02") == "https://pc.com/b02.tif"

    def test_case_insensitive_fallback(self):
        assets = {"b04": "https://example.com/b04.tif"}
        assert _band_url(assets, "B04") == "https://example.com/b04.tif"

    def test_missing_band_returns_none(self):
        assets = {"blue": "https://example.com/B02.tif"}
        assert _band_url(assets, "B08") is None

    def test_empty_assets_returns_none(self):
        assert _band_url({}, "B02") is None
