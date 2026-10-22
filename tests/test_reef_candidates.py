"""Tests for scripts/reef_candidates.py — offline-only.

extract_candidates() and write_geojson() work on local GeoTIFFs; no network required.
write_figure() uses matplotlib and is excluded (optional dep, display-less CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.reef_candidates import extract_candidates, write_geojson

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_likelihood(path: Path, data: np.ndarray,
                      bbox=(-8.4, 36.9, -8.1, 37.1)) -> Path:
    """Write a synthetic reef_likelihood GeoTIFF (float32, nodata=-9999)."""
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


def _two_blob_map(h: int = 60, w: int = 80, seed: int = 42) -> np.ndarray:
    """Random [0, 0.5] background + two square blobs at high likelihood.
    Using a varied background ensures np.percentile(90) ≈ 0.45, cleanly
    separating the blobs (0.85/0.90) from the background.
    """
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0, 0.5, (h, w)).astype(np.float32)
    arr[10:20, 10:20] = 0.9   # blob A (10×10 = 100 px)
    arr[35:40, 50:55] = 0.85  # blob B (5×5 = 25 px)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# extract_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractCandidates:
    def test_detects_both_blobs(self, tmp_path):
        """Two distinct high-likelihood blobs above threshold → both extracted.
        P90 of uniform [0,0.5] background ≈ 0.45; blobs at 0.85/0.90 pass cleanly.
        """
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=90, min_size=4, smooth_sigma=0)
        assert result["n_kept"] >= 2

    def test_minimum_size_filter(self, tmp_path):
        """A blob smaller than min_size is dropped."""
        rng = np.random.default_rng(7)
        arr = rng.uniform(0, 0.5, (40, 40)).astype(np.float32)  # varied background
        arr[5:7, 5:7] = 0.9   # 4 px blob (below strict min_size=10)
        arr[20:25, 20:25] = 0.9  # 25 px blob (above both thresholds)
        p = _write_likelihood(tmp_path / "like.tif", arr)
        result_strict = extract_candidates(p, percentile=90, min_size=10, smooth_sigma=0)
        result_loose  = extract_candidates(p, percentile=90, min_size=2,  smooth_sigma=0)
        assert result_strict["n_kept"] < result_loose["n_kept"]

    def test_candidates_sorted_by_score_descending(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        scores = [c["score"] for c in result["candidates"]]
        assert scores == sorted(scores, reverse=True)

    def test_rank_starts_at_1_and_is_consecutive(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        ranks = [c["rank"] for c in result["candidates"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_centroid_is_lon_lat(self, tmp_path):
        """Centroid of a blob in bbox (-8.4→-8.1, 36.9→37.1) must be inside those bounds."""
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        for c in result["candidates"]:
            lon, lat = c["centroid"]
            assert -8.4 <= lon <= -8.1, f"lon={lon} outside bbox"
            assert 36.9 <= lat <= 37.1, f"lat={lat} outside bbox"

    def test_area_ha_positive(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        for c in result["candidates"]:
            assert c["area_ha"] > 0.0

    def test_no_candidates_from_scattered_noise(self, tmp_path):
        """Scattered individual pixels above threshold (noise) are smaller than
        min_size and should all be dropped — producing zero kept candidates."""
        rng = np.random.default_rng(99)
        arr = rng.uniform(0, 0.6, (50, 50)).astype(np.float32)
        # No large contiguous blobs — only scattered single-pixel peaks
        p = _write_likelihood(tmp_path / "like.tif", arr)
        # min_size=20 drops any connected region smaller than 20 px; scattered
        # noise pixels are isolated (size=1), so nothing survives.
        result = extract_candidates(p, percentile=99, min_size=20, smooth_sigma=0)
        assert result["n_kept"] == 0

    def test_nodata_pixels_excluded(self, tmp_path):
        """Nodata (-9999) pixels must not generate candidates."""
        arr = np.full((30, 30), -9999.0, dtype=np.float32)
        arr[10:15, 10:15] = 0.9   # a real blob
        p = _write_likelihood(tmp_path / "like.tif", arr)
        result = extract_candidates(p, percentile=50, min_size=4, smooth_sigma=0)
        # The blob is above nodata-excluded finite pixels' 50th percentile
        for c in result["candidates"]:
            lon, lat = c["centroid"]
            assert -8.4 <= lon <= -8.1

    def test_result_has_required_keys(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        for key in ("candidates", "threshold", "n_total", "n_kept", "gps_labels"):
            assert key in result, f"Missing result key: {key}"

    def test_threshold_respects_percentile(self, tmp_path):
        """Higher percentile → higher threshold → fewer candidates."""
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        r_low  = extract_candidates(p, percentile=70, min_size=1, smooth_sigma=0)
        r_high = extract_candidates(p, percentile=95, min_size=1, smooth_sigma=0)
        assert r_high["threshold"] >= r_low["threshold"]
        assert r_high["n_kept"] <= r_low["n_kept"]


# ─────────────────────────────────────────────────────────────────────────────
# write_geojson
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteGeojson:
    def _run(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        out = tmp_path / "candidates.geojson"
        write_geojson(result, out)
        with open(out) as f:
            return json.load(f)

    def test_output_file_created(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        out = tmp_path / "out.geojson"
        write_geojson(result, out)
        assert out.exists()

    def test_is_feature_collection(self, tmp_path):
        fc = self._run(tmp_path)
        assert fc["type"] == "FeatureCollection"

    def test_feature_count_matches_n_kept(self, tmp_path):
        p = _write_likelihood(tmp_path / "like.tif", _two_blob_map())
        result = extract_candidates(p, percentile=80, min_size=4, smooth_sigma=0)
        out = tmp_path / "out.geojson"
        write_geojson(result, out)
        with open(out) as f:
            fc = json.load(f)
        assert len(fc["features"]) == result["n_kept"]

    def test_geometry_is_polygon(self, tmp_path):
        fc = self._run(tmp_path)
        for feat in fc["features"]:
            assert feat["geometry"]["type"] == "Polygon"

    def test_polygon_is_closed(self, tmp_path):
        """First and last coordinate of each ring must be identical."""
        fc = self._run(tmp_path)
        for feat in fc["features"]:
            ring = feat["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1], "Ring is not closed"

    def test_properties_contain_score(self, tmp_path):
        fc = self._run(tmp_path)
        for feat in fc["features"]:
            assert "score" in feat["properties"]

    def test_bbox_not_in_properties(self, tmp_path):
        """bbox is converted to polygon coords; it must not appear raw in properties."""
        fc = self._run(tmp_path)
        for feat in fc["features"]:
            assert "bbox" not in feat["properties"]
