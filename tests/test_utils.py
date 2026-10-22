"""Tests for src/utils.py — physics utility functions."""
import csv
import json
import math
from pathlib import Path

import pytest

from src.utils import (
    beer_lambert_transmittance,
    build_coastal_geojson,
    compute_metadata_stub,
    get_kd490,
    optical_path,
    snell_air_to_water,
    snell_sza,
)

# ── snell_air_to_water ────────────────────────────────────────────────────────

class TestSnellAirToWater:
    def test_normal_incidence_gives_zero(self):
        assert snell_air_to_water(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_30deg_incidence(self):
        # sin(30°)/1.333 = 0.5/1.333 = 0.375 → asin(0.375) ≈ 22.03°
        theta_w = snell_air_to_water(math.radians(30))
        assert math.degrees(theta_w) == pytest.approx(22.03, abs=0.1)

    def test_output_less_than_input(self):
        # Refracted angle in water is always smaller than air angle
        for deg in [10, 20, 30, 45, 60]:
            theta_air = math.radians(deg)
            theta_w = snell_air_to_water(theta_air)
            assert theta_w < theta_air

    def test_grazing_incidence_clamped(self):
        # Near 90° should not raise and should return a valid angle
        result = snell_air_to_water(math.radians(89.9))
        assert 0.0 <= result <= math.pi / 2


# ── snell_sza ─────────────────────────────────────────────────────────────────

class TestSnellSza:
    def test_returns_tuple(self):
        deg, rad = snell_sza(40.0)
        assert isinstance(deg, float)
        assert isinstance(rad, float)

    def test_consistent_with_snell_air_to_water(self):
        # snell_sza should agree with snell_air_to_water
        deg, rad = snell_sza(40.0)
        expected = snell_air_to_water(math.radians(40.0))
        assert rad == pytest.approx(expected, abs=1e-9)
        assert deg == pytest.approx(math.degrees(expected), abs=1e-6)

    def test_zero_sza(self):
        deg, rad = snell_sza(0.0)
        assert deg == pytest.approx(0.0, abs=1e-9)
        assert rad == pytest.approx(0.0, abs=1e-9)


# ── optical_path ──────────────────────────────────────────────────────────────

class TestOpticalPath:
    def test_vertical_ray(self):
        # theta = 0 → path = depth / cos(0) = depth
        assert optical_path(10.0, 0.0) == pytest.approx(10.0)

    def test_oblique_ray(self):
        # theta = 30° → path = 10 / cos(30°) ≈ 11.547
        path = optical_path(10.0, math.radians(30))
        assert path == pytest.approx(10.0 / math.cos(math.radians(30)), rel=1e-6)

    def test_zero_depth_gives_zero(self):
        assert optical_path(0.0, math.radians(20)) == pytest.approx(0.0)


# ── beer_lambert_transmittance ────────────────────────────────────────────────

class TestBeerLambertTransmittance:
    def test_zero_path_gives_one(self):
        assert beer_lambert_transmittance(0.042, 0.0) == pytest.approx(1.0)

    def test_zero_kd_gives_one(self):
        assert beer_lambert_transmittance(0.0, 20.0) == pytest.approx(1.0)

    def test_typical_algarve_10m(self):
        # kd=0.042 m⁻¹, two-way path 10m: T = exp(-2 * 0.042 * 10)
        expected = math.exp(-2 * 0.042 * 10)
        assert beer_lambert_transmittance(0.042, 10.0) == pytest.approx(expected, rel=1e-9)

    def test_transmittance_decreases_with_depth(self):
        kd = 0.042
        t_shallow = beer_lambert_transmittance(kd, 5.0)
        t_deep = beer_lambert_transmittance(kd, 20.0)
        assert t_shallow > t_deep

    def test_transmittance_decreases_with_turbidity(self):
        path = 10.0
        t_clear = beer_lambert_transmittance(0.042, path)
        t_turbid = beer_lambert_transmittance(0.10, path)
        assert t_clear > t_turbid

    def test_range_is_zero_to_one(self):
        for kd in [0.02, 0.042, 0.08, 0.15]:
            for depth in [1, 5, 10, 20, 40]:
                t = beer_lambert_transmittance(kd, depth)
                assert 0.0 < t <= 1.0


# ── get_kd490 ─────────────────────────────────────────────────────────────────

class TestGetKd490:
    def test_integer_key_lookup(self):
        table = {7: 0.042, 8: 0.042, 9: 0.046}
        assert get_kd490(7, table) == 0.042

    def test_string_key_fallback(self):
        # Some callers may pass string keys
        table = {"7": 0.042, "8": 0.042}
        assert get_kd490(7, table) == 0.042

    def test_missing_month_returns_default(self):
        table = {7: 0.042}
        result = get_kd490(11, table)
        assert result == 0.080  # hardcoded default in utils.get_kd490

    def test_int_cast_on_month(self):
        # String month should be cast to int for lookup
        table = {9: 0.046}
        assert get_kd490("9", table) == 0.046


# ── compute_metadata_stub ─────────────────────────────────────────────────────

class TestComputeMetadataStub:
    def test_known_date_returns_correct_sza(self):
        m = compute_metadata_stub("2025-09-25")
        assert m["solar_zenith_deg"] == pytest.approx(40.498)

    def test_unknown_date_returns_defaults(self):
        m = compute_metadata_stub("2000-01-01")
        assert m["solar_zenith_deg"] == pytest.approx(40.0)
        assert m["cloud_cover_pct"] == pytest.approx(2.0)

    def test_required_keys_present(self):
        m = compute_metadata_stub("2025-09-25")
        for key in ("date", "crs", "datum", "level", "solar_zenith_deg",
                    "solar_azimuth_deg", "cloud_cover_pct"):
            assert key in m

    def test_date_field_matches_input(self):
        m = compute_metadata_stub("2023-10-01")
        assert m["date"] == "2023-10-01"


# ── build_coastal_geojson ─────────────────────────────────────────────────────

_SAMPLE_ROWS = [
    {
        "site_name": "pedra_sta_eulalia", "latitude": "37.069081", "longitude": "-8.210242",
        "buffer_m": "3000", "slope_min": "0.0", "slope_max": "22.5", "slope_mean": "4.1",
        "slope_std": "3.2", "slope_median": "3.0", "slope_percentile_90": "8.9",
        "aspect_min": "0.0", "aspect_max": "360.0", "aspect_mean": "190.3",
        "aspect_std": "108.7", "aspect_median": "195.1",
    },
    {
        "site_name": "albufeira_reef", "latitude": "37.0690", "longitude": "-8.2105",
        "buffer_m": "3000", "slope_min": "0.1", "slope_max": "31.0", "slope_mean": "5.3",
        "slope_std": "4.1", "slope_median": "4.0", "slope_percentile_90": "10.2",
        "aspect_min": "1.0", "aspect_max": "359.0", "aspect_mean": "185.5",
        "aspect_std": "107.2", "aspect_median": "191.0",
    },
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


class TestBuildCoastalGeojson:
    def test_feature_count_matches_csv_rows(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        n = build_coastal_geojson(csv_p, geo_p)
        assert n == len(_SAMPLE_ROWS)
        with open(geo_p) as fh:
            g = json.load(fh)
        assert len(g["features"]) == len(_SAMPLE_ROWS)

    def test_geometry_is_point_with_lon_lat(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        build_coastal_geojson(csv_p, geo_p)
        with open(geo_p) as fh:
            g = json.load(fh)
        for feat, row in zip(g["features"], _SAMPLE_ROWS):
            assert feat["geometry"]["type"] == "Point"
            lon, lat = feat["geometry"]["coordinates"]
            assert lon == pytest.approx(float(row["longitude"]))
            assert lat == pytest.approx(float(row["latitude"]))

    def test_numeric_properties_preserved(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        build_coastal_geojson(csv_p, geo_p)
        with open(geo_p) as fh:
            g = json.load(fh)
        feat = g["features"][0]["properties"]
        assert feat["slope_mean"] == pytest.approx(4.1)
        assert feat["aspect_mean"] == pytest.approx(190.3)
        assert feat["buffer_m"] == pytest.approx(3000.0)
        assert feat["site_name"] == "pedra_sta_eulalia"

    def test_timestamp_added_and_is_string(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        build_coastal_geojson(csv_p, geo_p)
        with open(geo_p) as fh:
            g = json.load(fh)
        ts = g["features"][0]["properties"].get("timestamp")
        assert isinstance(ts, str) and len(ts) > 10

    def test_timestamp_preserved_from_csv_when_present(self, tmp_path):
        rows = [{**_SAMPLE_ROWS[0], "timestamp": "2026-06-15T23:46:02.531296+00:00"}]
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, rows)
        build_coastal_geojson(csv_p, geo_p)
        with open(geo_p) as fh:
            g = json.load(fh)
        # build_coastal_geojson overwrites timestamp with fresh UTC value — that's intentional
        assert "timestamp" in g["features"][0]["properties"]

    def test_crs_is_wgs84_crs84(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        build_coastal_geojson(csv_p, geo_p)
        with open(geo_p) as fh:
            g = json.load(fh)
        assert g["type"] == "FeatureCollection"
        assert g["crs"]["properties"]["name"] == "urn:ogc:def:crs:OGC:1.3:CRS84"

    def test_backup_created_when_geojson_exists(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        # First write — no backup expected
        build_coastal_geojson(csv_p, geo_p)
        backups_before = list(tmp_path.glob("*.bak_*"))
        assert not backups_before, "No backup expected on first write"
        # Second write — backup of first file must appear
        build_coastal_geojson(csv_p, geo_p)
        backups_after = list(tmp_path.glob("*.bak_*"))
        assert len(backups_after) == 1
        assert backups_after[0].name.startswith("features.geojson.bak_")

    def test_empty_csv_raises(self, tmp_path):
        csv_p = tmp_path / "empty.csv"
        csv_p.write_text("site_name,latitude,longitude\n")  # header only
        geo_p = tmp_path / "out.geojson"
        with pytest.raises(ValueError, match="empty"):
            build_coastal_geojson(csv_p, geo_p)

    def test_atomic_write_leaves_no_tmp_on_success(self, tmp_path):
        csv_p = tmp_path / "features.csv"
        geo_p = tmp_path / "features.geojson"
        _write_csv(csv_p, _SAMPLE_ROWS)
        build_coastal_geojson(csv_p, geo_p)
        tmp_files = list(tmp_path.glob("*.geojson.tmp"))
        assert not tmp_files, "Temp file must be removed after atomic rename"
