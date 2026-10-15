"""
Tests for the explicit missing-spectral-band / radiometry / georeferencing policy.

ALL DATA HERE IS SYNTHETIC (hand-made GeoTIFFs and arrays). These tests verify software
behaviour only — they are not evidence of scientific performance of the reef detector.
No network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import band_registry  # noqa: E402
from src.s2_radiometry import (  # noqa: E402
    RADIOMETRY_UNKNOWN,
    dn_to_reflectance,
    parse_processing_baseline,
    radiometry_from_product_metadata,
    radiometry_from_stac_asset,
)
import scripts.reef_detector as rd  # noqa: E402

# Synthetic UTM 29N grid near Santa Eulália (values chosen only to be inside zone 29)
X0, Y0, RES = 580_000.0, 4_105_000.0, 10.0
H, W = 120, 100
CRS = "EPSG:32629"
ES_RASTER_BANDS = [{"nodata": 0, "data_type": "uint16", "spatial_resolution": 10, "scale": 0.0001, "offset": -0.1}]
ES_PROPS = {"s2:processing_baseline": "05.11", "earthsearch:boa_offset_applied": True,
            "s2:product_uri": "SYNTHETIC_TEST_PRODUCT.SAFE", "s2:datatake_id": "SYNTHETIC"}


def _write(path: Path, arr: np.ndarray, transform=None, nodata=0, crs=CRS) -> str:
    transform = transform or from_origin(X0, Y0, RES, RES)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=str(arr.dtype), crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)
    return str(path)


def _full_lonlat(row: int, col: int) -> tuple[float, float]:
    """lon/lat of a pixel centre in the FULL synthetic raster."""
    from pyproj import Transformer
    x, y = from_origin(X0, Y0, RES, RES) * (col + 0.5, row + 0.5)
    return Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)


def _bbox_around(rows: tuple[int, int], cols: tuple[int, int], pad_px: float = 0.0) -> tuple:
    """lon/lat bbox (lon_min, lat_min, lon_max, lat_max) enclosing a pixel range of the full raster."""
    from pyproj import Transformer
    tf = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    t = from_origin(X0, Y0, RES, RES)
    xs, ys = [], []
    for r in (rows[0] - pad_px, rows[1] + pad_px):
        for c in (cols[0] - pad_px, cols[1] + pad_px):
            x, y = t * (c, r)
            lon, lat = tf.transform(x, y)
            xs.append(lon)
            ys.append(lat)
    return (min(xs), min(ys), max(xs), max(ys))


def _make_item(tmp_path: Path, dn: dict[str, np.ndarray], raster_bands=ES_RASTER_BANDS,
               props=ES_PROPS, drop: tuple = ()) -> dict:
    assets = {}
    for key, arr in dn.items():
        if key in drop:
            continue
        href = _write(tmp_path / f"{key}.tif", arr)
        a = {"href": href}
        if raster_bands is not None:
            a["raster:bands"] = raster_bands
        assets[key] = a
    return {"id": "SYNTHETIC_ITEM", "datetime": "2099-01-01", "cloud_cover": 0.0, "platform": "synthetic",
            "provider": "synthetic", "properties": dict(props), "assets": assets}


def _water_dn(h=H, w=W, seed=0) -> dict[str, np.ndarray]:
    """Clear-water-like DN including the +1000 BOA offset (synthetic)."""
    rng = np.random.default_rng(seed)
    blue = (1000 + 500 + rng.integers(-30, 30, (h, w))).astype(np.uint16)   # refl ≈ 0.05
    green = (1000 + 400 + rng.integers(-30, 30, (h, w))).astype(np.uint16)  # ≈ 0.04
    red = (1000 + 200 + rng.integers(-10, 10, (h, w))).astype(np.uint16)    # ≈ 0.02
    nir = (1000 + 50 + rng.integers(-10, 10, (h, w))).astype(np.uint16)     # ≈ 0.005
    return {"blue": blue, "green": green, "red": red, "nir": nir}


# ─────────────────────────────────────────────────────────────────────────────
# Radiometry
# ─────────────────────────────────────────────────────────────────────────────

class TestRadiometry:
    def test_earth_search_scale_offset_applied_once(self):
        rad = radiometry_from_stac_asset({"raster:bands": ES_RASTER_BANDS}, ES_PROPS)
        assert rad.is_known and rad.scale == 1e-4 and rad.offset == -0.1
        refl, ok = dn_to_reflectance(np.array([0, 1000, 1500, 500], dtype=np.uint16), rad)
        assert not ok[0] and np.isnan(refl[0])                     # DN 0 = nodata, masked first
        assert refl[1] == pytest.approx(0.0, abs=1e-7)
        assert refl[2] == pytest.approx(0.05, abs=1e-6)
        assert ok[3] and refl[3] == pytest.approx(-0.05, abs=1e-6)  # valid negative preserved

    def test_old_divide_by_10000_differs_by_offset(self):
        """Documents the bug fixed: DN/10000 ignores the -0.1 offset."""
        rad = radiometry_from_stac_asset({"raster:bands": ES_RASTER_BANDS}, ES_PROPS)
        dn = np.array([1050], dtype=np.uint16)
        refl, _ = dn_to_reflectance(dn, rad)
        assert dn[0] / 10000.0 - refl[0] == pytest.approx(0.1, abs=1e-6)

    def test_planetary_computer_asset_without_raster_bands_is_unknown(self):
        rad = radiometry_from_stac_asset({"href": "x"}, {"s2:processing_baseline": "05.11"})
        assert rad.status == RADIOMETRY_UNKNOWN
        with pytest.raises(ValueError, match="radiometry_unknown"):
            dn_to_reflectance(np.array([1500], dtype=np.uint16), rad)

    def test_scale_without_offset_after_baseline_04_is_unknown(self):
        rb = [{"nodata": 0, "data_type": "uint16", "scale": 0.0001}]
        assert not radiometry_from_stac_asset({"raster:bands": rb}, {"s2:processing_baseline": "04.00"}).is_known

    def test_scale_without_offset_before_baseline_04_is_known_zero_offset(self):
        rb = [{"nodata": 0, "data_type": "uint16", "scale": 0.0001}]
        rad = radiometry_from_stac_asset({"raster:bands": rb}, {"s2:processing_baseline": "03.01"})
        assert rad.is_known and rad.offset == 0.0

    def test_float_asset_refused_as_already_scaled(self):
        rb = [{"data_type": "float32", "scale": 1.0, "offset": 0.0}]
        assert not radiometry_from_stac_asset({"raster:bands": rb}, ES_PROPS).is_known

    def test_float_input_refused_double_conversion(self):
        rad = radiometry_from_stac_asset({"raster:bands": ES_RASTER_BANDS}, ES_PROPS)
        with pytest.raises(ValueError, match="double-conversion"):
            dn_to_reflectance(np.array([0.05], dtype=np.float32), rad)

    def test_product_metadata_boa_add_offset(self):
        xml = ("<x><PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>"
               "<BOA_QUANTIFICATION_VALUE unit=\"none\">10000</BOA_QUANTIFICATION_VALUE>"
               "<BOA_ADD_OFFSET band_id=\"7\">-1000</BOA_ADD_OFFSET></x>")
        rad = radiometry_from_product_metadata(xml, 7)
        assert rad.is_known and rad.scale == pytest.approx(1e-4) and rad.offset == pytest.approx(-0.1)

    def test_product_metadata_missing_offset_for_new_baseline_is_unknown(self):
        xml = ("<x><PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>"
               "<BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE></x>")
        assert not radiometry_from_product_metadata(xml, 7).is_known

    @pytest.mark.parametrize("raw,expected", [("05.11", 5.11), ("N0511", 5.11), ("04.00", 4.0), (None, None), ("x", None)])
    def test_parse_processing_baseline(self, raw, expected):
        assert parse_processing_baseline(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Band dependency registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_bands_everything_available(self):
        av = band_registry.evaluate(["B02", "B03", "B04", "B08"])
        assert all(av.available.values()) and av.gating_failures() == []

    def test_missing_b08_blocks_mask_and_everything_downstream(self):
        av = band_registry.evaluate(["B02", "B03", "B04"])
        assert av.missing_bands["water_mask"] == ["B08"]
        for f in ("water_mask", "deep_water_background", "lyzenga_dii", "reef_likelihood", "clarity_nir_glint"):
            assert not av.ok(f)
        assert av.blocked_by["lyzenga_dii"]  # transitive
        assert av.ok("clarity_red") and av.ok("true_colour_overlay")
        assert set(av.gating_failures()) == {"water_mask", "deep_water_background", "lyzenga_dii", "reef_likelihood"}

    def test_missing_b04_does_not_gate_detector(self):
        av = band_registry.evaluate(["B02", "B03", "B08"])
        assert av.gating_failures() == []
        assert not av.ok("clarity_red") and not av.ok("true_colour_overlay")


# ─────────────────────────────────────────────────────────────────────────────
# Georeferencing on the native grid
# ─────────────────────────────────────────────────────────────────────────────

class TestGeoreference:
    def test_window_transform_and_crs_are_native(self, tmp_path):
        dn = _water_dn()
        item = _make_item(tmp_path, dn)
        bbox = _bbox_around((20, 60), (30, 70))
        sc = rd.load_scene_from_item(item, bbox)
        assert str(sc["crs"]) == CRS
        # Window origin must sit exactly on a native pixel corner
        a = sc["transform"]
        assert a.a == RES and a.e == -RES
        assert ((a.c - X0) / RES) == pytest.approx(round((a.c - X0) / RES), abs=1e-9)
        assert ((Y0 - a.f) / RES) == pytest.approx(round((Y0 - a.f) / RES), abs=1e-9)

    def test_marker_pixel_found_at_its_coordinates(self, tmp_path):
        dn = _water_dn()
        dn["green"][37, 52] = 9999  # unique marker
        item = _make_item(tmp_path, dn)
        sc = rd.load_scene_from_item(item, _bbox_around((20, 60), (30, 70)))
        lon, lat = _full_lonlat(37, 52)
        r, c = rd.lonlat_to_rowcol(sc, lon, lat)
        assert sc["bands"]["green"][r, c] == pytest.approx(9999 * 1e-4 - 0.1, abs=1e-6)

    def test_coordinate_pixel_coordinate_roundtrip(self, tmp_path):
        item = _make_item(tmp_path, _water_dn())
        sc = rd.load_scene_from_item(item, _bbox_around((10, 80), (10, 80)))
        h, w = sc["shape"]
        for r, c in [(0, 0), (h - 1, w - 1), (h // 2, w // 3), (5, w - 2)]:
            lon, lat = rd.rowcol_to_lonlat(sc, r, c)
            assert rd.lonlat_to_rowcol(sc, lon, lat) == (r, c)

    def test_edge_window_clamped_and_transform_updated(self, tmp_path):
        dn = _water_dn()
        dn["blue"][50, W - 1] = 7777  # marker in the LAST column of the tile
        item = _make_item(tmp_path, dn)
        bbox = _bbox_around((40, 60), (80, W + 40))  # extends 40 px past the east edge
        sc = rd.load_scene_from_item(item, bbox)
        h, w = sc["shape"]
        assert sc["transform"].c + w * RES == pytest.approx(X0 + W * RES)  # clamped to tile edge
        lon, lat = _full_lonlat(50, W - 1)
        r, c = rd.lonlat_to_rowcol(sc, lon, lat)
        assert c == w - 1
        assert sc["bands"]["blue"][r, c] == pytest.approx(7777 * 1e-4 - 0.1, abs=1e-6)

    def test_window_covers_requested_bbox(self, tmp_path):
        """Every corner of the requested lon/lat bbox must fall inside the window read."""
        item = _make_item(tmp_path, _water_dn())
        bbox = _bbox_around((20.3, 60.7), (30.2, 70.6))
        sc = rd.load_scene_from_item(item, bbox)
        h, w = sc["shape"]
        for lon in (bbox[0], bbox[2]):
            for lat in (bbox[1], bbox[3]):
                r, c = rd.lonlat_to_rowcol(sc, lon, lat)
                assert 0 <= r < h + 1 and 0 <= c < w + 1, (lon, lat, r, c, h, w)
        # and the window is not larger than needed (+1 px snapping tolerance per side)
        assert w <= (70.6 - 30.2) + 3 and h <= (60.7 - 20.3) + 3

    def test_gating_reason_comes_from_registry(self, tmp_path):
        item = _make_item(tmp_path, _water_dn(), drop=("nir",))
        res = rd.process_scene_loaded(rd.load_scene_from_item(item, _bbox_around((0, H), (0, W))), tmp_path / "o")
        assert res["availability"]["missing_bands"]["water_mask"] == ["B08"]
        assert res["reason"].startswith("gated features unavailable")

    def test_bbox_outside_tile_is_unavailable(self, tmp_path):
        item = _make_item(tmp_path, _water_dn())
        with pytest.raises(rd.DetectorUnavailable):
            rd.load_scene_from_item(item, (10.0, 50.0, 10.1, 50.1))


# ─────────────────────────────────────────────────────────────────────────────
# Nodata handling and resampling
# ─────────────────────────────────────────────────────────────────────────────

class TestNodata:
    def test_nodata_masked_and_excluded_from_clarity(self, tmp_path):
        dn = _water_dn()
        dn["blue"][:, :] = 1500
        dn["blue"][30:33, 40:42] = 0  # nodata inside the clarity window
        item = _make_item(tmp_path, dn)
        sc = rd.load_scene_from_item(item, _bbox_around((0, H), (0, W)))
        lon, lat = _full_lonlat(31, 41)
        cl = rd.measure_clarity(sc, lon, lat, win=3)
        assert cl["blue"] == pytest.approx(0.05, abs=1e-6)  # zeros NOT averaged in
        assert cl["n_valid_px"]["blue"] < 9

    def test_all_nodata_window_gives_none_not_zero(self, tmp_path):
        dn = _water_dn()
        dn["nir"][28:35, 38:45] = 0
        item = _make_item(tmp_path, dn)
        sc = rd.load_scene_from_item(item, _bbox_around((0, H), (0, W)))
        cl = rd.measure_clarity(sc, *_full_lonlat(31, 41), win=3)
        assert cl["nir_glint"] is None and cl["valid"] is True

    def test_water_mask_excludes_invalid(self, tmp_path):
        dn = _water_dn()
        dn["green"][10:20, 10:20] = 0
        item = _make_item(tmp_path, dn)
        sc = rd.load_scene_from_item(item, _bbox_around((0, H), (0, W)))
        m = rd.water_mask(sc["bands"], valid=sc["valid"])
        r, c = rd.lonlat_to_rowcol(sc, *_full_lonlat(15, 15))
        assert not m[r, c] and m.sum() > 0

    def test_valid_negative_nir_kept_in_mask(self):
        bands = {"blue": np.full((4, 4), 0.05, np.float32), "green": np.full((4, 4), 0.04, np.float32),
                 "nir": np.full((4, 4), -0.003, np.float32)}
        assert rd.water_mask(bands).all()

    def test_depth_reprojection_nodata_does_not_contaminate(self, tmp_path):
        depth = np.full((H // 2, W // 2), -10.0, dtype=np.float32)
        depth[:, : W // 4] = np.nan
        dpath = tmp_path / "depth.tif"
        with rasterio.open(dpath, "w", driver="GTiff", height=depth.shape[0], width=depth.shape[1], count=1,
                           dtype="float32", crs=CRS, transform=from_origin(X0, Y0, 2 * RES, 2 * RES),
                           nodata=np.nan) as dst:
            dst.write(depth, 1)
        item = _make_item(tmp_path, _water_dn())
        sc = rd.load_scene_from_item(item, _bbox_around((0, H), (0, W)))
        grid = rd.load_depth_on_grid(sc, emodnet_path=dpath)
        fin = grid[np.isfinite(grid)]
        assert fin.size > 0 and np.allclose(fin, -10.0)   # no blending with nodata
        assert np.isnan(grid[:, : W // 2 - 2]).all()

    def test_downloader_resampling_excludes_nodata_and_uses_nearest_for_categories(self):
        import scripts.download_multiband_s2 as dl
        from rasterio.warp import Resampling
        src = np.full((30, 25), 1500, dtype=np.uint16)
        src[:, :5] = 0  # nodata strip
        out = dl.resample_to_grid(src, from_origin(X0, Y0, 20, 20), CRS, 0,
                                  from_origin(X0, Y0, 10, 10), CRS, (60, 50), Resampling.bilinear)
        valid = out[out != 0]
        assert valid.size > 0 and np.allclose(valid, 1500)  # never pulled toward 0
        assert (out[:, :9] == 0).all()
        assert dl.resampling_for("SCL") == Resampling.nearest
        assert dl.resampling_for("B05") == Resampling.bilinear


# ─────────────────────────────────────────────────────────────────────────────
# status="unavailable" instead of fabricated bands / zero probability
# ─────────────────────────────────────────────────────────────────────────────

class TestUnavailable:
    def _bbox(self):
        return _bbox_around((0, H), (0, W))

    def test_missing_b08_asset_returns_unavailable(self, tmp_path):
        item = _make_item(tmp_path, _water_dn(), drop=("nir",))
        sc = rd.load_scene_from_item(item, self._bbox())
        assert sc["band_status"]["nir"] == "missing_asset" and "nir" not in sc["bands"]
        out = tmp_path / "out"
        res = rd.process_scene_loaded(sc, out)
        assert res["status"] == "unavailable" and "B08" in res["reason"]
        assert res["validation"] is None and res["outputs"] == {}
        assert not out.exists()  # nothing written, no zero map

    def test_b08_radiometry_unknown_blocks(self, tmp_path):
        item = _make_item(tmp_path, _water_dn())
        item["assets"]["nir"].pop("raster:bands")
        sc = rd.load_scene_from_item(item, self._bbox())
        assert sc["band_status"]["nir"] == "radiometry_unknown"
        assert rd.process_scene_loaded(sc, tmp_path / "o")["status"] == "unavailable"

    def test_b08_all_nodata_blocks(self, tmp_path):
        dn = _water_dn()
        dn["nir"][:] = 0
        sc = rd.load_scene_from_item(_make_item(tmp_path, dn), self._bbox())
        assert sc["band_status"]["nir"] == "no_valid_pixels"
        assert rd.process_scene_loaded(sc, tmp_path / "o")["status"] == "unavailable"

    def test_water_mask_without_nir_raises_not_fabricates(self):
        bands = {"blue": np.ones((3, 3), np.float32), "green": np.ones((3, 3), np.float32)}
        with pytest.raises(rd.DetectorUnavailable):
            rd.water_mask(bands)

    def test_missing_b04_still_runs_and_reports_red_none(self, tmp_path):
        dn = _water_dn(seed=3)
        sc = rd.load_scene_from_item(_make_item(tmp_path, dn, drop=("red",)), self._bbox())
        res = rd.process_scene_loaded(sc, tmp_path / "o")
        assert res["status"] == "ok"
        assert all(c.get("red") is None for c in res["clarity"].values() if c.get("valid"))
        assert not Path(res["outputs"]["overlay"]).exists()  # no fake true colour

    def test_ok_run_refuses_to_overwrite_outputs(self, tmp_path):
        sc = rd.load_scene_from_item(_make_item(tmp_path, _water_dn(seed=5)), self._bbox())
        out = tmp_path / "o"
        assert rd.process_scene_loaded(sc, out)["status"] == "ok"
        with pytest.raises(FileExistsError):
            rd.process_scene_loaded(sc, out)

    def test_degenerate_normalisation_is_nan_not_zero(self):
        assert np.isnan(rd._robust_minmax(np.full((5, 5), 3.0))).all()


# ─────────────────────────────────────────────────────────────────────────────
# Kd(665) is not invented
# ─────────────────────────────────────────────────────────────────────────────

class TestKdB04:
    def test_no_b04_gives_none(self):
        from src.reef_ml_predictor_acolite import gordon_kd_inversion
        rng = np.random.default_rng(1)
        b02 = rng.uniform(0.02, 0.06, (40, 40)).astype(np.float32)
        b03 = rng.uniform(0.02, 0.05, (40, 40)).astype(np.float32)
        kd02, kd03, kd04 = gordon_kd_inversion(b02, b03, None, 0.045)
        assert kd04 is None and np.isfinite(kd02) and np.isfinite(kd03)

    def test_genuine_b04_gives_value(self):
        from src.reef_ml_predictor_acolite import gordon_kd_inversion
        rng = np.random.default_rng(2)
        b = [rng.uniform(0.01, 0.06, (40, 40)).astype(np.float32) for _ in range(3)]
        assert isinstance(gordon_kd_inversion(b[0], b[1], b[2], 0.045)[2], float)

    def test_too_few_pixels_does_not_invent_kd04(self):
        from src.reef_ml_predictor_acolite import gordon_kd_inversion
        z = np.zeros((5, 5), np.float32)
        assert gordon_kd_inversion(z, z, None, 0.045)[2] is None
