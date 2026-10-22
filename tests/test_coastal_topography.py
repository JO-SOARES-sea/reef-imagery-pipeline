"""Tests for src/coastal_topography.py — CoastalTopographyAnalyzer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.coastal_topography import CoastalTopographyAnalyzer

ALGARVE_BBOX = (-8.25, 37.04, -8.17, 37.10)


# ── MinIO /vsis3 streaming (preferred over full download) ──────────────────────

class TestMinioStreaming:
    def test_href_to_vsis3_https_minio(self):
        href = "https://stor-002.a.acnca.pt:9000/lidar/MDT50cm/MDT-50cm-191013-04-2024_v01.tif"
        assert CoastalTopographyAnalyzer._href_to_vsis3(href) == \
            "/vsis3/lidar/MDT50cm/MDT-50cm-191013-04-2024_v01.tif"

    def test_href_to_vsis3_s3_scheme(self):
        assert CoastalTopographyAnalyzer._href_to_vsis3("s3://lidar/MDS50cm/x.tif") == \
            "/vsis3/lidar/MDS50cm/x.tif"

    def test_href_to_vsis3_passthrough_and_none(self):
        assert CoastalTopographyAnalyzer._href_to_vsis3("/vsis3/lidar/x.tif") == "/vsis3/lidar/x.tif"
        assert CoastalTopographyAnalyzer._href_to_vsis3(None) is None

    def test_no_credentials_disables_streaming(self, tmp_path, monkeypatch):
        for v in ("AWS_ENDPOINT_URL2", "AWS_ACCESS_KEY_ID2", "AWS_SECRET_ACCESS_KEY2"):
            monkeypatch.delenv(v, raising=False)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), dem_source="dgt")
        assert a._minio_stream_env() is None
        # With no env, the crop helper signals fallback (False) without touching network.
        assert a._stream_crop_to_local("https://h/lidar/a.tif", tmp_path / "a.tif", None) is False


# ── Tile naming ────────────────────────────────────────────────────────────────

class TestGlo30TileName:
    def test_northern_western(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(37, -9)
        assert name == "Copernicus_DSM_COG_10_N37_00_W009_00_DEM"

    def test_southern_eastern(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(-5, 12)
        assert name == "Copernicus_DSM_COG_10_S05_00_E012_00_DEM"

    def test_equator_prime_meridian(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(0, 0)
        assert name == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"

    def test_algarve_tile(self):
        name = CoastalTopographyAnalyzer._glo30_tile_name(37, -8)
        assert "N37" in name and "W008" in name


class TestGlo30TilesForBbox:
    def test_single_tile(self):
        # bbox fully within one 1°×1° tile
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox((-8.25, 37.04, -8.17, 37.10))
        assert tiles == [(37, -9)]

    def test_two_lon_tiles(self):
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox((-8.5, 37.0, -7.5, 37.9))
        lons = {lon for _, lon in tiles}
        assert -9 in lons and -8 in lons

    def test_four_tiles_crossing_boundaries(self):
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox((-8.5, 36.5, -7.5, 37.5))
        assert len(tiles) == 4

    def test_output_is_list_of_tuples(self):
        tiles = CoastalTopographyAnalyzer._glo30_tiles_for_bbox(ALGARVE_BBOX)
        assert isinstance(tiles, list)
        for t in tiles:
            assert isinstance(t, tuple) and len(t) == 2


# ── Initialisation ─────────────────────────────────────────────────────────────

class TestInit:
    def test_default_dem_source(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        assert a.dem_source == "auto"

    def test_explicit_dem_source(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), dem_source="srtm")
        assert a.dem_source == "srtm"

    def test_output_dirs_created(self, tmp_path):
        out = tmp_path / "costal_out"
        CoastalTopographyAnalyzer(ALGARVE_BBOX, str(out))
        assert out.exists()
        assert (out / "mdt_tiles").exists()

    def test_bbox_stored(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        assert a.bbox == ALGARVE_BBOX


# ── STAC fetch (mocked) ────────────────────────────────────────────────────────

class TestFetchStacItems:
    def test_returns_list(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "features": [{"id": "tile-1", "assets": {"Data": {"href": "http://x/t.tif"}},
                          "geometry": {"type": "Polygon", "coordinates": [[[-8.2,37.0],[-8.1,37.0],[-8.1,37.1],[-8.2,37.1],[-8.2,37.0]]]}}],
            "context": {"returned": 1},
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
            items = a.fetch_stac_items(limit=1)
        assert len(items) == 1
        assert items[0]["id"] == "tile-1"

    def test_network_error_returns_empty(self, tmp_path):
        import requests as _req
        with patch("requests.get", side_effect=_req.RequestException("timeout")):
            a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
            items = a.fetch_stac_items()
        assert items == []


# ── Credentials (mocked) ───────────────────────────────────────────────────────

class TestReadCdseCredentials:
    def test_reads_username_password(self, tmp_path):
        import base64
        cred_content = base64.b64encode(b"username = test@example.com\npassword = secret123\n").decode()
        cred_file = tmp_path / ".copernicusmarine-credentials"
        cred_file.write_text(cred_content)
        with patch.object(CoastalTopographyAnalyzer, "CDSE_CRED_FILE", cred_file):
            user, pwd = CoastalTopographyAnalyzer._read_cdse_credentials()
        assert user == "test@example.com"
        assert pwd == "secret123"

    def test_missing_file_raises(self, tmp_path):
        with patch.object(CoastalTopographyAnalyzer, "CDSE_CRED_FILE", tmp_path / "no_file"):
            with pytest.raises(FileNotFoundError):
                CoastalTopographyAnalyzer._read_cdse_credentials()


# ── Production artefact check ──────────────────────────────────────────────────

class TestProductionOutputs:
    """Verify the 15-site production run produced valid artefacts."""

    FEATURES_CSV = Path("outputs/coastal_topography/algarve_coastal_features.csv")
    FEATURES_GJ  = Path("outputs/coastal_topography/algarve_coastal_features.geojson")
    DEM_TIF      = Path("outputs/coastal_topography/dem_mosaic_50cm.tif")

    def test_csv_exists(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        assert self.FEATURES_CSV.stat().st_size > 0

    def test_csv_has_15_sites(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        import pandas as pd
        df = pd.read_csv(self.FEATURES_CSV)
        assert len(df) == 15

    def test_csv_slope_values_in_range(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        import pandas as pd
        df = pd.read_csv(self.FEATURES_CSV)
        assert (df["slope_mean"] >= 0.0).all()
        assert (df["slope_mean"] <= 45.0).all()

    def test_csv_aspect_values_in_range(self):
        if not self.FEATURES_CSV.exists():
            pytest.skip("Production run not yet executed")
        import pandas as pd
        df = pd.read_csv(self.FEATURES_CSV)
        assert (df["aspect_mean"] >= 0.0).all()
        assert (df["aspect_mean"] <= 360.0).all()

    def test_geojson_exists_and_valid(self):
        if not self.FEATURES_GJ.exists():
            pytest.skip("Production run not yet executed")
        import json
        with open(self.FEATURES_GJ) as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 15

    def test_dem_mosaic_exists(self):
        if not self.DEM_TIF.exists():
            pytest.skip("Production run not yet executed")
        import rasterio
        with rasterio.open(str(self.DEM_TIF)) as src:
            assert src.crs.to_epsg() == 3763
            assert src.count == 1


# =============================================================================
# Synthetic DEM helpers + coverage tests (no network, no real tiles)
# =============================================================================

import rasterio
from rasterio.crs import CRS as _RioCRS
from rasterio.transform import from_bounds as _tfrom_bounds


def _make_dem(tmp_path, crs_epsg=3763, width=120, height=100, slope=True,
              nodata=-999.0, filename="synth_dem.tif"):
    """Tiny synthetic DEM in EPSG:3763 near Algarve (for offline tests)."""
    out = tmp_path / filename
    west, south, east, north = -8000.0, -291000.0, -6800.0, -290000.0
    transform = _tfrom_bounds(west, south, east, north, width, height)
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    data = (xx * 80 + yy * 40).astype(np.float32) if slope else np.ones((height, width), np.float32) * 10.0
    with rasterio.open(str(out), "w", driver="GTiff",
                       height=height, width=width, count=1,
                       dtype="float32", crs=_RioCRS.from_epsg(crs_epsg),
                       transform=transform, nodata=nodata) as dst:
        dst.write(data, 1)
    return out


# ── Nodata normalisation tests ─────────────────────────────────────────────────

class TestNodataNormalisation:
    """Verify that -3.4e38 (float32-min) tiles are recoded to -999.0 in the mosaic."""

    def test_alt_nodata_recoded_in_mosaic(self, tmp_path):
        # Create a tile that declares nodata = float32-min (-3.4e28)
        alt_nodata = -3.4028235e38
        dem_alt = _make_dem(tmp_path, nodata=alt_nodata, filename="dem_alt.tif")
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        result = a.build_dem_mosaic([dem_alt])
        assert result is not None
        with rasterio.open(str(result)) as src:
            data = src.read(1)
            # No pixel should carry the old alt nodata value
            assert not np.any(data <= alt_nodata * 0.5), \
                "float32-min nodata survived into mosaic"
            # The declared nodata in the file should be -999.0
            assert src.nodata == pytest.approx(-999.0)

    def test_standard_nodata_unchanged(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        result = a.build_dem_mosaic([dem])
        assert result is not None
        with rasterio.open(str(result)) as src:
            assert src.nodata == pytest.approx(-999.0)


# ── CHM (Canopy Height Model) tests ───────────────────────────────────────────

class TestComputeCanopyHeight:
    def test_chm_non_negative(self, tmp_path):
        # MDT and MDS same grid; MDS = MDT + 2 m everywhere → CHM = 2 m
        mdt = _make_dem(tmp_path, slope=False, filename="mdt.tif")
        # Build MDS with 2 m added
        mds_out = tmp_path / "mds.tif"
        with rasterio.open(str(mdt)) as src:
            data = src.read(1).astype(np.float32) + 2.0
            meta = src.meta.copy()
        with rasterio.open(str(mds_out), "w", **meta) as dst:
            dst.write(data, 1)

        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        chm_path = a.compute_canopy_height(mdt_path=mdt, mds_path=mds_out)
        assert chm_path is not None and chm_path.exists()
        with rasterio.open(str(chm_path)) as src:
            chm = src.read(1)
            assert np.nanmin(chm) >= 0.0
            assert np.nanmean(chm) == pytest.approx(2.0, abs=0.1)

    def test_chm_clamped_at_50m(self, tmp_path):
        mdt = _make_dem(tmp_path, slope=False, filename="mdt.tif")
        mds_out = tmp_path / "mds.tif"
        with rasterio.open(str(mdt)) as src:
            data = src.read(1).astype(np.float32) + 100.0  # exceeds 50 m cap
            meta = src.meta.copy()
        with rasterio.open(str(mds_out), "w", **meta) as dst:
            dst.write(data, 1)

        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        chm_path = a.compute_canopy_height(mdt_path=mdt, mds_path=mds_out)
        assert chm_path is not None
        with rasterio.open(str(chm_path)) as src:
            assert np.nanmax(src.read(1)) <= 50.0

    def test_chm_missing_mds_returns_none(self, tmp_path):
        mdt = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        result = a.compute_canopy_height(mdt_path=mdt, mds_path=tmp_path / "no_mds.tif")
        assert result is None

    def test_chm_stats_in_extracted_features(self, tmp_path):
        mdt = _make_dem(tmp_path, slope=False, filename="mdt.tif")
        mds_out = tmp_path / "mds.tif"
        with rasterio.open(str(mdt)) as src:
            data = src.read(1).astype(np.float32) + 3.0
            meta = src.meta.copy()
        with rasterio.open(str(mds_out), "w", **meta) as dst:
            dst.write(data, 1)

        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        a.dem_mosaic_path = mdt
        sp, ap = a.derive_slope_aspect(mdt)
        a.slope_path = sp
        a.aspect_path = ap
        a.chm_path = a.compute_canopy_height(mdt_path=mdt, mds_path=mds_out)

        sites = [("synth_site", 37.069, -8.210)]
        df = a.extract_features_for_sites(sites, buffer_m=500, include_chm=True)
        assert df is not None
        chm_cols = [c for c in df.columns if c.startswith("chm_")]
        assert len(chm_cols) > 0, "No chm_* columns in extracted features"


class TestDerivesSlopeAspect:
    def test_slope_non_negative(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        sp, ap = a.derive_slope_aspect(dem)
        assert sp is not None and sp.exists()
        assert ap is not None and ap.exists()
        with rasterio.open(str(sp)) as src:
            s = src.read(1)
            assert np.nanmin(s) >= 0.0

    def test_aspect_in_0_360(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        _, ap = a.derive_slope_aspect(dem)
        with rasterio.open(str(ap)) as src:
            asp = src.read(1)
            assert np.nanmin(asp) >= 0.0
            assert np.nanmax(asp) <= 360.0

    def test_flat_dem_zero_slope(self, tmp_path):
        dem = _make_dem(tmp_path, slope=False)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        sp, _ = a.derive_slope_aspect(dem)
        with rasterio.open(str(sp)) as src:
            s = src.read(1)
            assert np.nanmean(s) < 1.0  # essentially flat

    def test_missing_dem_returns_none(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        sp, ap = a.derive_slope_aspect(tmp_path / "nonexistent.tif")
        assert sp is None and ap is None


class TestBuildDemMosaic:
    def test_mosaic_from_local_tiles(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        result = a.build_dem_mosaic([dem])
        assert result is not None
        assert result.exists()

    def test_mosaic_uses_cache(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=True)
        r1 = a.build_dem_mosaic([dem])
        r2 = a.build_dem_mosaic([dem])  # should return cached
        assert r1 == r2

    def test_empty_tiles_returns_none(self, tmp_path):
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        result = a.build_dem_mosaic([])
        assert result is None


class TestExtractFeaturesForSites:
    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        # Pre-populate slope/aspect so no download needed
        a.dem_mosaic_path = dem
        sp, ap = a.derive_slope_aspect(dem)
        a.slope_path = sp
        a.aspect_path = ap
        sites = [("synth_site", 37.069, -8.210)]
        df = a.extract_features_for_sites(sites, buffer_m=500)
        assert df is not None
        assert len(df) == 1
        assert "slope_mean" in df.columns
        assert "aspect_mean" in df.columns
        assert "site_name" in df.columns

    def test_multiple_sites(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=False)
        a.dem_mosaic_path = dem
        sp, ap = a.derive_slope_aspect(dem)
        a.slope_path = sp
        a.aspect_path = ap
        sites = [("site_a", 37.069, -8.210), ("site_b", 37.072, -8.205)]
        df = a.extract_features_for_sites(sites, buffer_m=200)
        assert df is not None
        assert len(df) == 2


class TestRunAnalysisWithSyntheticData:
    def test_full_pipeline_no_download(self, tmp_path):
        dem = _make_dem(tmp_path)
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path), cache_tiles=True)
        # Pre-populate cache so build_dem_mosaic skips download
        import shutil
        shutil.copy(dem, a.dem_mosaic_path)
        sites = [("synth_site", 37.069, -8.210)]
        result = a.run_analysis(sites, buffer_m=500)
        assert result["status"] == "success"
        assert result["sites_analyzed"] == 1

    def test_save_features_creates_files(self, tmp_path):
        import pandas as pd
        a = CoastalTopographyAnalyzer(ALGARVE_BBOX, str(tmp_path))
        df = pd.DataFrame([{
            "site_name": "test", "latitude": 37.069, "longitude": -8.210,
            "buffer_m": 1000, "slope_mean": 1.5, "aspect_mean": 180.0,
            "timestamp": "2026-01-01",
        }])
        paths = a.save_features(df, output_name="test_features")
        assert Path(paths["csv"]).exists()
        assert Path(paths["json"]).exists()
        assert Path(paths["geojson"]).exists()


# ── --selftest CLI: end-to-end via _run_selftest() ────────────────────────────

class TestRunSelftest:
    """End-to-end tests for the private `_run_selftest` covering the 4 spec
    scenarios + edge cases. MinIO is never actually contacted — the internal
    helpers `_minio_stream_env` and `_stream_crop_to_local` are patched, and
    the output file is redirected via the `COASTAL_SELFTEST_OUT` env var.
    """

    @staticmethod
    def _make_tiny_geotiff(path, w=64, h=48, crs_str="EPSG:3763"):
        """Write a minimal valid GeoTIFF to `path`."""
        data = np.arange(w * h, dtype=np.float32).reshape(h, w)
        transform = _tfrom_bounds(-8.234, 37.074, -8.222, 37.083, w, h)
        with rasterio.open(
            str(path), "w",
            driver="GTiff", height=h, width=w, count=1,
            dtype=np.float32, crs=crs_str, transform=transform,
        ) as dst:
            dst.write(data, 1)
        return path

    # ── Scenario 1: creds OK + stream OK → exit 0, success message, file written
    def test_scenario1_creds_and_stream_succeeds(
        self, monkeypatch, capsys, tmp_path,
    ):
        monkeypatch.setenv("AWS_ENDPOINT_URL2", "https://stub")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID2", "AKIA_STUB")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY2", "secret")
        out = tmp_path / "selftest_crop.tif"
        monkeypatch.setenv("COASTAL_SELFTEST_OUT", str(out))

        # Mock the stream helper to actually write a valid GeoTIFF
        # (so the subsequent rasterio.open in _run_selftest has something to read).
        def fake_stream(href, local_path, env):
            self._make_tiny_geotiff(local_path)
            return True

        sentinel = object()  # any non-None env marker
        with patch.object(CoastalTopographyAnalyzer, "_minio_stream_env",
                          return_value=sentinel), \
             patch.object(CoastalTopographyAnalyzer, "_stream_crop_to_local",
                          side_effect=fake_stream):
            from src.coastal_topography import _run_selftest
            rc = _run_selftest()

        out_text = capsys.readouterr().out
        assert rc == 0
        assert "Streamed crop from MinIO" in out_text
        assert str(out) in out_text
        assert out.exists()

    # ── Scenario 2: no creds → exit 0, no-creds message, no network
    def test_scenario2_no_creds_no_network(self, monkeypatch, capsys):
        for v in ("AWS_ENDPOINT_URL2", "AWS_ACCESS_KEY_ID2", "AWS_SECRET_ACCESS_KEY2"):
            monkeypatch.delenv(v, raising=False)
        from src.coastal_topography import _run_selftest
        rc = _run_selftest()
        out_text = capsys.readouterr().out
        assert rc == 0
        assert "No MinIO credentials found" in out_text
        assert "streaming inactive" in out_text
        # No stream was attempted → no success message
        assert "Streamed crop" not in out_text

    # ── Scenario 3: creds OK + stream fails → exit 0, fallback message
    def test_scenario3_creds_stream_fails_gracefully(
        self, monkeypatch, capsys, tmp_path,
    ):
        monkeypatch.setenv("AWS_ENDPOINT_URL2", "https://stub")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID2", "AKIA_STUB")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY2", "secret")
        out = tmp_path / "selftest_crop.tif"
        monkeypatch.setenv("COASTAL_SELFTEST_OUT", str(out))
        with patch.object(CoastalTopographyAnalyzer, "_minio_stream_env",
                          return_value=object()), \
             patch.object(CoastalTopographyAnalyzer, "_stream_crop_to_local",
                          return_value=False):
            from src.coastal_topography import _run_selftest
            rc = _run_selftest()
        out_text = capsys.readouterr().out
        assert rc == 0
        assert "Streaming failed" in out_text
        assert "fall back to CDD" in out_text
        # No file was created (the mock didn't write anything)
        assert not out.exists()

    # ── Scenario 4: creds OK + unexpected crash → exit 1 (not masked as success)
    def test_scenario4_creds_unexpected_crash_returns_one(
        self, monkeypatch, capsys, tmp_path,
    ):
        monkeypatch.setenv("AWS_ENDPOINT_URL2", "https://stub")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID2", "AKIA_STUB")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY2", "secret")
        out = tmp_path / "selftest_crop.tif"
        monkeypatch.setenv("COASTAL_SELFTEST_OUT", str(out))
        with patch.object(CoastalTopographyAnalyzer, "_minio_stream_env",
                          return_value=object()), \
             patch.object(CoastalTopographyAnalyzer, "_stream_crop_to_local",
                          side_effect=RuntimeError("simulated blow-up")):
            from src.coastal_topography import _run_selftest
            rc = _run_selftest()
        out_text = capsys.readouterr().out
        assert rc == 1
        assert "simulated blow-up" in out_text
        assert "Streaming failed" in out_text

    # ── Edge case: output file already exists → must be overwritten
    def test_edgecase_existing_output_file_is_overwritten(
        self, monkeypatch, capsys, tmp_path,
    ):
        monkeypatch.setenv("AWS_ENDPOINT_URL2", "https://stub")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID2", "AKIA_STUB")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY2", "secret")
        out = tmp_path / "selftest_crop.tif"
        out.write_bytes(b"STALE CONTENT FROM A PREVIOUS RUN")
        monkeypatch.setenv("COASTAL_SELFTEST_OUT", str(out))

        # Mock the stream helper to actually write a valid GeoTIFF
        def fake_stream(href, local_path, env):
            self._make_tiny_geotiff(local_path)
            return True

        with patch.object(CoastalTopographyAnalyzer, "_minio_stream_env",
                          return_value=object()), \
             patch.object(CoastalTopographyAnalyzer, "_stream_crop_to_local",
                          side_effect=fake_stream):
            from src.coastal_topography import _run_selftest
            rc = _run_selftest()

        out_text = capsys.readouterr().out
        assert rc == 0
        # Stale content must be gone
        with open(out, "rb") as f:
            content = f.read()
        assert b"STALE CONTENT" not in content
        # And the file is now a valid GeoTIFF
        with rasterio.open(out) as src:
            assert src.crs.to_string() == "EPSG:3763"

    # ── Edge case: no-overlap bbox (stream returns False internally) → graceful fallback
    # _stream_crop_to_local returns False at line 224 when the AOI window
    # doesn't overlap the tile. From _run_selftest's perspective this is
    # indistinguishable from a network/auth failure — both yield exit 0
    # with a fallback message and no file is created.
    def test_edgecase_no_overlap_bbox_falls_back(
        self, monkeypatch, capsys, tmp_path,
    ):
        monkeypatch.setenv("AWS_ENDPOINT_URL2", "https://stub")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID2", "AKIA_STUB")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY2", "secret")
        out = tmp_path / "selftest_crop.tif"
        monkeypatch.setenv("COASTAL_SELFTEST_OUT", str(out))
        with patch.object(CoastalTopographyAnalyzer, "_minio_stream_env",
                          return_value=object()), \
             patch.object(CoastalTopographyAnalyzer, "_stream_crop_to_local",
                          return_value=False):
            from src.coastal_topography import _run_selftest
            rc = _run_selftest()
        out_text = capsys.readouterr().out
        assert rc == 0
        assert "Streaming failed" in out_text
        assert "fall back" in out_text
        # No file was created
        assert not out.exists()

    # ── Edge case: permission failure on rasterio.open → exit 1 (not masked)
    # PermissionError is an OSError subclass, caught by the generic
    # `except Exception` at line 1209. The selftest must NOT mask this as
    # success — it should return 1 so CI/operators see the failure.
    def test_edgecase_permission_error_on_read_returns_one(
        self, monkeypatch, capsys, tmp_path,
    ):
        monkeypatch.setenv("AWS_ENDPOINT_URL2", "https://stub")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID2", "AKIA_STUB")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY2", "secret")
        out = tmp_path / "selftest_crop.tif"
        monkeypatch.setenv("COASTAL_SELFTEST_OUT", str(out))

        # Mock the stream helper to actually write a valid GeoTIFF, then
        # make rasterio.open raise on the subsequent read.
        def fake_stream(href, local_path, env):
            self._make_tiny_geotiff(local_path)
            return True

        with patch.object(CoastalTopographyAnalyzer, "_minio_stream_env",
                          return_value=object()), \
             patch.object(CoastalTopographyAnalyzer, "_stream_crop_to_local",
                          side_effect=fake_stream), \
             patch.object(rasterio, "open",
                          side_effect=PermissionError("read-only /tmp")):
            from src.coastal_topography import _run_selftest
            rc = _run_selftest()

        out_text = capsys.readouterr().out
        # PermissionError is an unexpected crash → exit 1, not masked as success
        assert rc == 1
        assert "Streaming failed" in out_text
        assert "read-only /tmp" in out_text
