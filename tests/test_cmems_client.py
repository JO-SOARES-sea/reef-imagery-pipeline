"""Tests for src/cmems_client.py.

Pure-logic tests (stream resolution, variable→product routing, ISO
normalisation, coord detection, dataset/bbox sanity) run offline. The live
point_timeseries pull is marked @pytest.mark.network.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.cmems_client import (
    _HAS_CM,
    ALGARVE_BBOX,
    DATASETS,
    DEFAULT_VARIABLES,
    MY_LAG_DAYS,
    VAR_PRODUCT,
    _coord_name,
    _group_by_product,
    _iso,
    _resolve_stream,
    point_timeseries,
)

_SITES_CSV = Path("outputs/coastal_topography/algarve_coastal_features.csv")


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_stream
# ─────────────────────────────────────────────────────────────────────────────
class TestResolveStream:
    def test_old_window_uses_my(self):
        assert _resolve_stream(("2020-01-01", "2020-01-31"), "auto") == "my"

    def test_recent_window_uses_nrt(self):
        today = datetime.now(UTC).date().isoformat()
        assert _resolve_stream((today, today), "auto") == "nrt"

    def test_just_inside_lag_is_nrt(self):
        end = (datetime.now(UTC) - timedelta(days=MY_LAG_DAYS - 2)).date().isoformat()
        assert _resolve_stream(("2024-01-01", end), "auto") == "nrt"

    def test_explicit_override_my(self):
        # explicit stream wins even for a recent window
        today = datetime.now(UTC).date().isoformat()
        assert _resolve_stream((today, today), "my") == "my"

    def test_explicit_override_nrt(self):
        assert _resolve_stream(("2010-01-01", "2010-01-02"), "nrt") == "nrt"


# ─────────────────────────────────────────────────────────────────────────────
# _group_by_product
# ─────────────────────────────────────────────────────────────────────────────
class TestGroupByProduct:
    def test_transparency_vars_grouped(self):
        groups = _group_by_product(["KD490", "ZSD", "SPM"])
        assert set(groups) == {"transp"}
        assert sorted(groups["transp"]) == ["KD490", "SPM", "ZSD"]

    def test_chl_routed_to_plankton(self):
        groups = _group_by_product(["CHL"])
        assert set(groups) == {"plankton"}

    def test_mixed_splits_across_products(self):
        groups = _group_by_product(["KD490", "CHL"])
        assert set(groups) == {"transp", "plankton"}

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            _group_by_product(["NOT_A_VAR"])


# ─────────────────────────────────────────────────────────────────────────────
# _iso
# ─────────────────────────────────────────────────────────────────────────────
class TestIso:
    def test_date_only_start(self):
        assert _iso("2024-09-01") == "2024-09-01T00:00:00"

    def test_date_only_end(self):
        assert _iso("2024-09-01", end=True) == "2024-09-01T23:59:59"

    def test_full_iso_passthrough(self):
        assert _iso("2024-09-01T12:30:00") == "2024-09-01T12:30:00"


# ─────────────────────────────────────────────────────────────────────────────
# _coord_name
# ─────────────────────────────────────────────────────────────────────────────
class _FakeDS:
    def __init__(self, names):
        self.coords = dict.fromkeys(names)
        self.dims = tuple(names)


class TestCoordName:
    def test_finds_latitude(self):
        assert (
            _coord_name(_FakeDS(["latitude", "longitude", "time"]), "latitude", "lat") == "latitude"
        )

    def test_falls_through_to_alias(self):
        assert _coord_name(_FakeDS(["lat", "lon"]), "latitude", "lat") == "lat"

    def test_missing_raises(self):
        with pytest.raises(KeyError):
            _coord_name(_FakeDS(["time"]), "latitude", "lat", "y")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / bbox structural sanity
# ─────────────────────────────────────────────────────────────────────────────
class TestStructuralSanity:
    def test_every_variable_product_exists_in_both_streams(self):
        for product in set(VAR_PRODUCT.values()):
            for stream in ("my", "nrt"):
                assert product in DATASETS[stream], f"{product} missing from {stream}"

    def test_default_variables_are_all_known(self):
        for var in DEFAULT_VARIABLES:
            assert var in VAR_PRODUCT

    def test_bbox_is_min_max_ordered(self):
        min_lon, min_lat, max_lon, max_lat = ALGARVE_BBOX
        assert min_lon < max_lon and min_lat < max_lat

    @pytest.mark.skipif(not _SITES_CSV.exists(), reason="coastal features CSV not present")
    def test_bbox_covers_all_reef_sites(self):
        min_lon, min_lat, max_lon, max_lat = ALGARVE_BBOX
        with open(_SITES_CSV) as f:
            for row in csv.DictReader(f):
                lon, lat = float(row["longitude"]), float(row["latitude"])
                assert min_lon <= lon <= max_lon, f"{row['site_name']} lon {lon} outside bbox"
                assert min_lat <= lat <= max_lat, f"{row['site_name']} lat {lat} outside bbox"


# ─────────────────────────────────────────────────────────────────────────────
# Live pull (network)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.network
@pytest.mark.skipif(not _HAS_CM, reason="copernicusmarine is not installed")
class TestPointTimeseriesLive:
    def test_pedra_september_returns_finite_kd490(self):
        ts = point_timeseries(
            lon=-8.2102,
            lat=37.0691,
            date_range=("2024-09-01", "2024-09-10"),
            variables=("KD490",),
        )
        assert "KD490" in ts
        assert len(ts["KD490"]["value"]) == len(ts["KD490"]["time"])
        assert all(v == v for v in ts["KD490"]["value"])  # no NaN
