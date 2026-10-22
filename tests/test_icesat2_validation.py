"""Tests for src/icesat2_validation.py — ICESat-2 ATL08 SDB validation."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.icesat2_validation import (
    _fetch_atl08_for_date,
    _sample_sdb_at_points,
    _search_atl08,
    run_icesat2_validation,
)

BBOX = (-8.25, 37.04, -8.17, 37.10)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_photon(lat=37.07, lon=-8.21, h=-5.0):
    return {"lat": lat, "lon": lon, "h": h}


def _oa_response(photons: list[dict]):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "product": "ATL08", "trackId": 1,
        "series": [{"beam": "gt1l", "data": photons}],
    }
    return resp


def _oa_empty():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"product": "ATL08", "trackId": 1, "series": []}
    return resp


def _oa_error(code=500):
    resp = MagicMock()
    resp.status_code = code
    return resp


def _make_depth_raster(tmp_path: Path, depths: np.ndarray,
                       bbox=BBOX, crs="EPSG:4326") -> Path:
    """Write a minimal float32 GeoTIFF covering BBOX with given depth values."""
    import rasterio
    from rasterio.transform import from_bounds

    h, w = depths.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    tif = tmp_path / "sdb_depth_map.tif"
    with rasterio.open(
        str(tif), "w", driver="GTiff", dtype="float32",
        width=w, height=h, count=1, crs=crs, transform=transform,
    ) as dst:
        dst.write(depths.astype(np.float32), 1)
    return tif


# ── _fetch_atl08_for_date ─────────────────────────────────────────────────────

class TestFetchAtl08ForDate:
    def test_returns_photons_from_series(self):
        photons = [_make_photon(), _make_photon(lat=37.08, h=-8.0)]
        with patch("requests.get", return_value=_oa_response(photons)):
            result = _fetch_atl08_for_date(BBOX, "2023-07-15", track_id=1)
        assert len(result) == 2
        assert result[0]["lat"] == 37.07

    def test_returns_empty_on_non_200(self):
        with patch("requests.get", return_value=_oa_error(404)):
            result = _fetch_atl08_for_date(BBOX, "2023-07-15", track_id=1)
        assert result == []

    def test_returns_empty_on_network_exception(self):
        import requests as _req
        with patch("requests.get", side_effect=_req.ConnectionError("timeout")):
            result = _fetch_atl08_for_date(BBOX, "2023-07-15", track_id=1)
        assert result == []

    def test_returns_empty_when_series_is_empty(self):
        with patch("requests.get", return_value=_oa_empty()):
            result = _fetch_atl08_for_date(BBOX, "2023-07-15", track_id=1)
        assert result == []

    def test_accumulates_photons_across_multiple_series(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "series": [
                {"beam": "gt1l", "data": [_make_photon()]},
                {"beam": "gt1r", "data": [_make_photon(h=-7.0), _make_photon(h=-3.0)]},
            ]
        }
        with patch("requests.get", return_value=resp):
            result = _fetch_atl08_for_date(BBOX, "2023-07-15", 1)
        assert len(result) == 3


# ── _search_atl08 ─────────────────────────────────────────────────────────────

class TestSearchAtl08:
    def test_returns_empty_on_invalid_date(self):
        result = _search_atl08(BBOX, "not-a-date")
        assert result == []

    def test_returns_empty_when_all_fetches_empty(self):
        with patch("src.icesat2_validation._fetch_atl08_for_date", return_value=[]):
            result = _search_atl08(BBOX, "2023-07-15", search_days=7)
        assert result == []

    def test_stops_when_500_photons_collected(self):
        """Should stop searching once ≥500 photons are accumulated."""
        big_batch = [_make_photon(h=-float(i % 30)) for i in range(500)]
        call_count = 0

        def _side(bbox, date, track_id):
            nonlocal call_count
            call_count += 1
            return big_batch if call_count == 1 else []

        with patch("src.icesat2_validation._fetch_atl08_for_date", side_effect=_side):
            result = _search_atl08(BBOX, "2023-07-15", search_days=90)
        assert len(result) == 500
        # Should have stopped after the first successful date
        assert call_count <= len(range(1, 15))  # ≤ 14 (one date, all tracks)

    def test_searches_weekly_dates(self):
        """Dates tried should be 7 days apart within the search window."""
        dates_tried = []

        def _side(bbox, date, track_id):
            if track_id == 1:  # only record on first track to avoid duplicates
                dates_tried.append(date)
            return []

        with patch("src.icesat2_validation._fetch_atl08_for_date", side_effect=_side):
            _search_atl08(BBOX, "2023-07-15", search_days=14)

        from datetime import date, timedelta
        centre = date.fromisoformat("2023-07-15")
        expected = [
            (centre + timedelta(days=d)).isoformat()
            for d in range(-14, 15, 7)
        ]
        assert dates_tried == expected

    def test_collects_photons_across_multiple_dates(self):
        hits = {"2023-07-08": [_make_photon()], "2023-07-15": [_make_photon(h=-9.0)]}

        def _side(bbox, date, track_id):
            return hits.get(date, []) if track_id == 1 else []

        with patch("src.icesat2_validation._fetch_atl08_for_date", side_effect=_side):
            result = _search_atl08(BBOX, "2023-07-15", search_days=14)
        assert len(result) == 2


# ── _sample_sdb_at_points ─────────────────────────────────────────────────────

class TestSampleSdbAtPoints:
    def test_returns_correct_depth_at_raster_centre(self, tmp_path):
        depths = np.full((10, 10), 7.5, dtype=np.float32)
        tif = _make_depth_raster(tmp_path, depths)
        lons = np.array([-8.21])
        lats = np.array([37.07])
        result = _sample_sdb_at_points(tif, lons, lats)
        assert len(result) == 1
        assert abs(result[0] - 7.5) < 0.01

    def test_returns_nan_outside_raster_extent(self, tmp_path):
        depths = np.full((10, 10), 5.0, dtype=np.float32)
        tif = _make_depth_raster(tmp_path, depths)
        lons = np.array([-10.0])  # far outside bbox
        lats = np.array([37.07])
        result = _sample_sdb_at_points(tif, lons, lats)
        assert np.isnan(result[0])

    def test_handles_nodata_as_nan(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        tif = tmp_path / "nd.tif"
        transform = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], 4, 4)
        arr = np.full((4, 4), -9999.0, dtype=np.float32)
        with rasterio.open(str(tif), "w", driver="GTiff", dtype="float32",
                           width=4, height=4, count=1, crs="EPSG:4326",
                           transform=transform, nodata=-9999.0) as dst:
            dst.write(arr, 1)
        lons = np.array([-8.21])
        lats = np.array([37.07])
        result = _sample_sdb_at_points(tif, lons, lats)
        assert np.isnan(result[0])

    def test_returns_all_nan_on_missing_file(self, tmp_path):
        lons = np.array([-8.21, -8.20])
        lats = np.array([37.07, 37.08])
        result = _sample_sdb_at_points(tmp_path / "nonexistent.tif", lons, lats)
        assert np.all(np.isnan(result))
        assert len(result) == 2


# ── run_icesat2_validation ────────────────────────────────────────────────────

class TestRunIcesat2Validation:
    def test_returns_none_when_depth_map_missing(self, tmp_path):
        result = run_icesat2_validation(
            depth_map_path=tmp_path / "missing.tif",
            output_dir=tmp_path / "val",
            scene_date="2023-07-15", bbox=BBOX,
        )
        assert result is None

    def test_skipped_report_written_when_no_photons(self, tmp_path):
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)
        with patch("src.icesat2_validation._search_atl08", return_value=[]):
            result = run_icesat2_validation(
                depth_map_path=tif,
                output_dir=tmp_path / "val",
                scene_date="2023-07-15", bbox=BBOX, search_days=0,
            )
        assert result is not None
        assert result["status"] == "skipped"
        assert result["n_photons"] == 0
        report_file = tmp_path / "val" / "validation_report.json"
        assert report_file.exists()
        with open(report_file) as f:
            on_disk = json.load(f)
        assert on_disk["status"] == "skipped"

    def test_skipped_when_photons_have_no_coords(self, tmp_path):
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)
        bad_photons = [{"x": 1, "y": 2}]  # no lat/lon/h keys
        with patch("src.icesat2_validation._search_atl08", return_value=bad_photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )
        assert result["status"] == "skipped"
        assert "missing lat/lon/h" in result["reason"]

    def test_skipped_when_too_few_shallow_water_photons(self, tmp_path):
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)
        # All photons above sea surface → no negative elevations
        photons = [{"lat": 37.07, "lon": -8.21, "h": 10.0} for _ in range(20)]
        with patch("src.icesat2_validation._search_atl08", return_value=photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )
        assert result["status"] == "skipped"
        assert "shallow-water photons" in result["reason"]

    def test_ok_report_with_correct_metrics(self, tmp_path):
        """End-to-end with synthetic photons that match the raster exactly → RMSE≈0."""
        # 10×10 raster, all 5.0 m deep
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)

        # 10 photons at depth 5.0 m → elevation = −5.0
        photons = [
            {"lat": 37.04 + i * 0.006, "lon": -8.24 + i * 0.007, "h": -5.0}
            for i in range(10)
        ]
        with patch("src.icesat2_validation._search_atl08", return_value=photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )

        assert result["status"] == "ok"
        assert result["n_photons_shallow"] == 10
        # RMSE and bias should be near zero (SDB = ICESat depth = 5.0 m)
        assert result["rmse_m"] < 0.5
        assert abs(result["bias_m"]) < 0.5
        assert result["mae_m"] >= 0.0
        assert result.get("pearson_r") is None or -1 <= result["pearson_r"] <= 1

    def test_metrics_correct_for_known_residuals(self, tmp_path):
        """SDB = 6 m, ICESat = 4 m → bias = +2.0, RMSE = 2.0."""
        depths = np.full((10, 10), 6.0)  # SDB says 6 m everywhere
        tif = _make_depth_raster(tmp_path, depths)

        # 10 photons at 4 m depth (ICESat elevation = −4)
        photons = [
            {"lat": 37.04 + i * 0.006, "lon": -8.24 + i * 0.007, "h": -4.0}
            for i in range(10)
        ]
        with patch("src.icesat2_validation._search_atl08", return_value=photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )

        assert result["status"] == "ok"
        assert abs(result["rmse_m"] - 2.0) < 0.05
        assert abs(result["bias_m"] - 2.0) < 0.05   # SDB overestimates
        assert "overestimates" in result["interpretation"]

    def test_skipped_when_too_few_colocated_after_raster_sample(self, tmp_path):
        """Photons outside raster extent → NaN → too few co-located points."""
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)
        # All photons far outside the raster → NaN depths
        photons = [
            {"lat": 40.0 + i * 0.01, "lon": -5.0, "h": -5.0}
            for i in range(10)
        ]
        with patch("src.icesat2_validation._search_atl08", return_value=photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )
        assert result["status"] == "skipped"
        assert "co-located points" in result["reason"]

    def test_output_dir_created_if_missing(self, tmp_path):
        tif = _make_depth_raster(tmp_path, np.full((4, 4), 5.0))
        out = tmp_path / "new" / "nested" / "dir"
        with patch("src.icesat2_validation._search_atl08", return_value=[]):
            run_icesat2_validation(tif, out, "2023-07-15", BBOX, search_days=0)
        assert out.is_dir()

    def test_date_extracted_from_filename(self, tmp_path):
        """When scene_date is None, it should be parsed from the raster filename."""
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)
        # Rename to include a date
        dated_tif = tmp_path / "sdb_depth_20230715.tif"
        tif.rename(dated_tif)

        with patch("src.icesat2_validation._search_atl08", return_value=[]) as mock_search:
            run_icesat2_validation(dated_tif, tmp_path / "val",
                                   scene_date=None, bbox=BBOX, search_days=0)
        # The date passed to _search_atl08 should have been extracted from filename
        call_args = mock_search.call_args
        passed_date = call_args[0][1]  # second positional arg
        assert passed_date == "2023-07-15"

    def test_report_json_contains_all_ok_keys(self, tmp_path):
        depths = np.full((10, 10), 8.0)
        tif = _make_depth_raster(tmp_path, depths)
        photons = [
            {"lat": 37.04 + i * 0.006, "lon": -8.24 + i * 0.007, "h": -8.0}
            for i in range(8)
        ]
        with patch("src.icesat2_validation._search_atl08", return_value=photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )
        required = {"status", "rmse_m", "bias_m", "mae_m", "n_colocated",
                    "n_photons_shallow", "icesat2_depth_mean_m", "sdb_depth_mean_m",
                    "interpretation"}
        assert required.issubset(result.keys())

    def test_pearson_r_is_none_when_all_same_depth(self, tmp_path):
        """Constant depth arrays have zero std → Pearson-r undefined → None."""
        depths = np.full((10, 10), 5.0)
        tif = _make_depth_raster(tmp_path, depths)
        photons = [
            {"lat": 37.04 + i * 0.006, "lon": -8.24 + i * 0.007, "h": -5.0}
            for i in range(8)
        ]
        with patch("src.icesat2_validation._search_atl08", return_value=photons):
            result = run_icesat2_validation(
                tif, tmp_path / "val", "2023-07-15", BBOX, search_days=0,
            )
        # pearson_r is None when std=0 for either array
        assert result.get("pearson_r") is None
