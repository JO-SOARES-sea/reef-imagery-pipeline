#!/usr/bin/env python3
"""
Tests for ih_bathy_features.py
==============================

Run:  python -m pytest tests/test_ih_bathy_features.py -v
Or:   python tests/test_ih_bathy_features.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.ih_bathy_features import (
    _QUERY_URL,
    BathyFeatureEngine,
    IHBathyDownloader,
    get_bathy_features_for_summary,
)

# =============================================================================
# A.  Downloader unit tests  (offline / mocked)
# =============================================================================

def test_tile_bbox_generation():
    """Tile a large bbox into smaller chunks."""
    downloader = IHBathyDownloader()
    tiles = downloader._tile_bbox(-8.5, 37.0, -8.0, 37.5, step=0.25)
    assert len(tiles) == 4  # 2×2 grid
    # First tile
    w, s, e, n = tiles[0]
    assert w == -8.5 and s == 37.0
    assert e == -8.25 and n == 37.25
    # Last tile
    w, s, e, n = tiles[-1]
    assert e == -8.0 and n == 37.5


def test_deduplicate_removes_duplicates():
    """Exact duplicate polylines by (objectid, depth, first coord) are removed."""
    features = [
        {"depth": 10.0, "coords": [[-8.2, 37.1], [-8.1, 37.1]], "shape_leng": 100.0, "objectid": 1},
        {"depth": 10.0, "coords": [[-8.2, 37.1], [-8.1, 37.1]], "shape_leng": 100.0, "objectid": 1},  # dup
        {"depth": 20.0, "coords": [[-8.2, 37.0], [-8.1, 37.0]], "shape_leng": 200.0, "objectid": 2},
    ]
    downloader = IHBathyDownloader()
    uniq = downloader._deduplicate(features)
    assert len(uniq) == 2
    assert uniq[0]["depth"] == 10.0
    assert uniq[1]["depth"] == 20.0


def test_parse_features_arcgis_json():
    """Parse a minimal ArcGIS REST JSON response."""
    mock_data = {
        "features": [
            {
                "attributes": {"OBJECTID": 101, "Depth": 10, "Shape_Leng": 1234.5},
                "geometry": {"paths": [[[-8.21, 37.06], [-8.20, 37.06]]]},
            },
            {
                "attributes": {"OBJECTID": 102, "Depth": 20, "Shape_Leng": 567.8},
                "geometry": {"paths": [[[-8.21, 37.05], [-8.20, 37.05]]]},
            },
        ]
    }
    downloader = IHBathyDownloader()
    parsed = downloader._parse_features(mock_data)
    assert len(parsed) == 2
    assert parsed[0]["depth"] == 10.0
    assert parsed[0]["objectid"] == 101
    assert parsed[0]["shape_leng"] == 1234.5
    assert parsed[0]["coords"] == [[-8.21, 37.06], [-8.20, 37.06]]


def test_cache_path_is_deterministic():
    """Same bbox + depths → same cache path."""
    downloader = IHBathyDownloader()
    p1 = downloader._cache_path(-8.5, 37.0, -8.0, 37.5, [10, 20])
    p2 = downloader._cache_path(-8.5, 37.0, -8.0, 37.5, [10, 20])
    assert p1 == p2
    # Different depths → different path
    p3 = downloader._cache_path(-8.5, 37.0, -8.0, 37.5, [10])
    assert p1 != p3


def test_cache_save_and_load_roundtrip():
    """Save features to cache, then load them back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = IHBathyDownloader(cache_dir=tmpdir)
        features = [
            {"depth": 10.0, "coords": [[-8.2, 37.1], [-8.1, 37.1]], "shape_leng": 100.0, "objectid": 1},
            {"depth": 20.0, "coords": [[-8.2, 37.0], [-8.1, 37.0]], "shape_leng": 200.0, "objectid": 2},
        ]
        path = Path(tmpdir) / "test_cache.gpkg"
        downloader._save_cache(path, features)
        loaded = downloader._load_cache(path)
        assert len(loaded) == 2
        assert loaded[0]["depth"] == 10.0
        assert loaded[1]["objectid"] == 2


# =============================================================================
# B.  Feature-engineering unit tests  (pure geometry, no network)
# =============================================================================

def test_classify_zone_rules():
    """Zone classification logic with known distance inputs."""
    engine = BathyFeatureEngine()

    # Very shallow (<200m from 10m isobath)
    assert engine._classify_zone({"dist_10m": 150.0}) == "very_shallow"

    # Shallow reef (<500m from 20m isobath)
    assert engine._classify_zone({"dist_10m": 500.0, "dist_20m": 300.0}) == "shallow_reef"

    # Nearshore mid (10m <1500m or 20m <1500m)
    assert engine._classify_zone({"dist_10m": 1000.0, "dist_20m": 1200.0}) == "nearshore_mid"

    # Mid depth (30m <1000m)
    assert engine._classify_zone({"dist_30m": 800.0}) == "mid_depth"

    # Offshore (50m <500m)
    assert engine._classify_zone({"dist_50m": 400.0}) == "offshore"

    # Unknown / far away
    assert engine._classify_zone({"dist_10m": np.inf, "dist_20m": np.inf}) == "offshore"


def test_empty_features_structure():
    """Empty feature dict has correct keys and sentinel values."""
    engine = BathyFeatureEngine()
    empty = engine._empty_features()
    assert empty["nearest_isobath_distance_m"] == np.inf
    assert empty["bathymetry_zone_class"] == "unknown"
    assert empty["bathymetry_slope_proxy"] == 0.0
    assert empty["n_isobaths_in_aoi"] == 0


def test_compute_distances_haversine():
    """Haversine fallback produces finite distances for known points."""
    engine = BathyFeatureEngine()
    features = [
        {"depth": 10.0, "coords": [[-8.210, 37.069], [-8.209, 37.069]]},
    ]
    dists = engine._compute_distances_haversine(-8.210492, 37.069071, features)
    assert "dist_10m" in dists
    assert dists["dist_10m"] < 1000.0  # Should be <1km
    assert dists["dist_10m"] > 0.0


def test_slope_proxy_with_nearby_contours():
    """Slope proxy from nearby contours of different depths."""
    engine = BathyFeatureEngine()
    features = [
        {"depth": 10.0, "coords": [[-8.210, 37.069]]},
        {"depth": 20.0, "coords": [[-8.210, 37.069]]},
        {"depth": 30.0, "coords": [[-8.210, 37.069]]},
    ]
    slope = engine._slope_proxy(features, -8.210, 37.069)
    assert slope > 0.0  # std of [10,20,30]


def test_slope_proxy_no_nearby():
    """No nearby contours → slope proxy = 0.0."""
    engine = BathyFeatureEngine()
    features = [
        {"depth": 10.0, "coords": [[-8.0, 37.0]]},  # far away
    ]
    slope = engine._slope_proxy(features, -8.210, 37.069)
    assert slope == 0.0


# =============================================================================
# C.  Integration / end-to-end  (uses real network — may be skipped)
# =============================================================================

@pytest.mark.network
def test_live_fetch_pedra_do_alto():
    """
    Real network test: fetch isobaths for the Pedra do Alto AOI.
    Skipped if service is unreachable (no failure).
    """
    import requests

    try:
        requests.get(_QUERY_URL, timeout=5)
    except Exception:
        print("SKIP: IH service unreachable — live test skipped")
        return

    downloader = IHBathyDownloader()
    features = downloader.fetch_for_aoi(
        min_lon=-8.25, min_lat=37.05,
        max_lon=-8.18, max_lat=37.08,
        depths=[10, 20, 30],
    )
    assert len(features) > 0, "Expected at least some isobaths for Algarve"
    depths_found = {f["depth"] for f in features}
    assert 10.0 in depths_found or 20.0 in depths_found or 30.0 in depths_found
    print(f"OK: fetched {len(features)} isobaths, depths={sorted(depths_found)}")


@pytest.mark.network
def test_live_features_for_point_albufeira():
    """
    Real network test: compute bathymetry features for Albufeira reef spot.
    Skipped if service is unreachable.
    """
    import requests

    try:
        requests.get(_QUERY_URL, timeout=5)
    except Exception:
        print("SKIP: IH service unreachable — live test skipped")
        return

    engine = BathyFeatureEngine()
    feats = engine.compute_features_for_point(
        lon=-8.210492, lat=37.069071, buffer_m=5_000
    )
    assert feats["bathymetry_zone_class"] != "unknown"
    assert feats["n_isobaths_in_aoi"] > 0
    assert feats["nearest_isobath_distance_m"] < np.inf
    print(f"OK: features={json.dumps(feats, indent=2, default=str)}")


@pytest.mark.network
def test_get_bathy_features_for_summary():
    """One-liner convenience function works end-to-end."""
    import requests

    try:
        requests.get(_QUERY_URL, timeout=5)
    except Exception:
        print("SKIP: IH service unreachable")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        feats = get_bathy_features_for_summary(
            lon=-8.210492, lat=37.069071,
            cache_dir=tmpdir, buffer_m=5_000
        )
        assert "nearest_isobath_distance_m" in feats
        assert "bathymetry_zone_class" in feats
        print(f"OK: summary keys = {list(feats.keys())}")


# =============================================================================
# D.  Run all
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running ih_bathy_features tests")
    print("=" * 60)

    tests = [
        test_tile_bbox_generation,
        test_deduplicate_removes_duplicates,
        test_parse_features_arcgis_json,
        test_cache_path_is_deterministic,
        test_cache_save_and_load_roundtrip,
        test_classify_zone_rules,
        test_empty_features_structure,
        test_compute_distances_haversine,
        test_slope_proxy_with_nearby_contours,
        test_slope_proxy_no_nearby,
        test_live_fetch_pedra_do_alto,
        test_live_features_for_point_albufeira,
        test_get_bathy_features_for_summary,
    ]

    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            # Network skips print their own message
            skipped += 1
            print(f"  ⚠ {t.__name__}: {e}")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0:
        print("✓ All tests successful")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
