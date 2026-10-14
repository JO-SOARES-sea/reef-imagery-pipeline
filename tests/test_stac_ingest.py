from datetime import datetime, timezone

import pytest

from src import stac_ingest


class DummyAsset:
    def __init__(self, href: str):
        self.href = href


class DummyItem:
    def __init__(self, item_id: str, cloud_cover: float, assets: dict):
        self.id = item_id
        self.properties = {"eo:cloud_cover": cloud_cover}
        self.collection_id = "sentinel-2-l2a"
        self.assets = {k: DummyAsset(v) for k, v in assets.items()}
        self.datetime = None


class RichDummyItem:
    """DummyItem with a full set of STAC view/solar/cloud properties."""

    def __init__(self, props: dict, dt=None):
        self.id = "S2A_TEST"
        self.properties = props
        self.collection_id = "sentinel-2-l2a"
        self.assets = {}
        self.datetime = dt


class DummySearch:
    def __init__(self, items):
        self._items = items

    def get_items(self):
        return self._items

    def items(self):
        return self._items


class DummyClient:
    def __init__(self, items):
        self._items = items

    def search(self, *args, **kwargs):
        return DummySearch(self._items)


def test_normalize_datetime_range_string():
    assert stac_ingest.normalize_datetime_range("2025-09-01/2025-09-30") == "2025-09-01/2025-09-30"


def test_normalize_datetime_range_tuple():
    assert stac_ingest.normalize_datetime_range(("2025-09-01", "2025-09-30")) == "2025-09-01/2025-09-30"


def test_choose_least_cloudy():
    items = [
        DummyItem("item-a", 32.0, {}),
        DummyItem("item-b", 10.5, {}),
        DummyItem("item-c", 15.0, {}),
    ]
    best = stac_ingest.choose_least_cloudy(items)
    assert best is not None
    assert best.id == "item-b"


def test_get_asset_hrefs():
    item = DummyItem("item-001", 5.0, {"B02": "https://example.com/B02.tif", "B03": "https://example.com/B03.tif"})
    hrefs = stac_ingest.get_asset_hrefs(item, ["B02", "B03", "B04"])
    assert hrefs["B02"] == "https://example.com/B02.tif"
    assert hrefs["B03"] == "https://example.com/B03.tif"
    assert "B04" not in hrefs


def test_open_stac_catalog(monkeypatch):
    dummy_client = DummyClient([])

    def fake_open(url, modifier=None):
        assert url == stac_ingest.EARTH_SEARCH_STAC_URL
        return dummy_client

    monkeypatch.setattr(stac_ingest.Client, "open", fake_open)
    client = stac_ingest.open_stac_catalog(stac_ingest.EARTH_SEARCH_STAC_URL)
    assert client is dummy_client


def test_search_sentinel2_scenes(monkeypatch):
    items = [DummyItem("item-x", 12.0, {})]
    dummy_client = DummyClient(items)

    def fake_open(url, modifier=None):
        assert url == stac_ingest.EARTH_SEARCH_STAC_URL
        return dummy_client

    monkeypatch.setattr(stac_ingest.Client, "open", fake_open)
    scenes = stac_ingest.search_sentinel2_scenes(
        37.07, -8.21, ("2025-09-01", "2025-09-15"), max_cloud_cover=30.0
    )
    assert scenes == items


def test_search_sentinel2_scenes_fallback(monkeypatch):
    items = [DummyItem("item-y", 11.0, {})]
    dummy_client = DummyClient(items)
    calls = []

    def fake_open(url, modifier=None):
        calls.append(url)
        if url == stac_ingest.EARTH_SEARCH_STAC_URL:
            raise RuntimeError("Earth Search unavailable")
        return dummy_client

    monkeypatch.setattr(stac_ingest.Client, "open", fake_open)
    scenes = stac_ingest.search_sentinel2_scenes(
        37.07, -8.21, ("2025-09-01", "2025-09-15"), max_cloud_cover=30.0
    )
    assert scenes == items
    assert calls == [stac_ingest.EARTH_SEARCH_STAC_URL, stac_ingest.PC_STAC_URL]


# ── metadata_from_stac_item ───────────────────────────────────────────────────

class TestMetadataFromStacItem:
    _FULL_PROPS = {
        "eo:cloud_cover": 1.5,
        "view:sun_elevation": 49.502,   # → zenith = 40.498
        "view:sun_azimuth": 158.883,
        "view:incidence_angle": 7.21,
        "view:azimuth": 100.5,
        "proj:epsg": 32629,
        "s2:product_type": "MSIL2A",
    }

    def test_required_keys_present(self):
        item = RichDummyItem(self._FULL_PROPS)
        m = stac_ingest.metadata_from_stac_item(item)
        for key in ("date", "crs", "datum", "level", "solar_zenith_deg",
                    "solar_azimuth_deg", "satellite_zenith_deg",
                    "satellite_azimuth_deg", "cloud_cover_pct"):
            assert key in m, f"missing key: {key}"

    def test_solar_zenith_derived_from_elevation(self):
        item = RichDummyItem(self._FULL_PROPS)
        m = stac_ingest.metadata_from_stac_item(item)
        assert m["solar_zenith_deg"] == pytest.approx(90.0 - 49.502, abs=1e-3)

    def test_cloud_cover_and_azimuth(self):
        item = RichDummyItem(self._FULL_PROPS)
        m = stac_ingest.metadata_from_stac_item(item)
        assert m["cloud_cover_pct"] == pytest.approx(1.5, abs=1e-3)
        assert m["solar_azimuth_deg"] == pytest.approx(158.883, abs=1e-3)
        assert m["satellite_zenith_deg"] == pytest.approx(7.21, abs=1e-3)
        assert m["satellite_azimuth_deg"] == pytest.approx(100.5, abs=1e-3)

    def test_crs_from_epsg(self):
        item = RichDummyItem(self._FULL_PROPS)
        m = stac_ingest.metadata_from_stac_item(item)
        assert m["crs"] == "EPSG:32629"
        assert m["datum"] == "WGS84"

    def test_crs_from_proj_code(self):
        props = {**self._FULL_PROPS, "proj:epsg": None, "proj:code": "EPSG:32630"}
        assert stac_ingest.metadata_from_stac_item(RichDummyItem(props))["crs"] == "EPSG:32630"

    def test_level_normalised_msil2a(self):
        item = RichDummyItem({**self._FULL_PROPS, "s2:product_type": "MSIL2A"})
        assert stac_ingest.metadata_from_stac_item(item)["level"] == "L2A"

    def test_level_normalised_from_processing_level(self):
        props = {k: v for k, v in self._FULL_PROPS.items() if k != "s2:product_type"}
        props["processing:level"] = "L2A"
        item = RichDummyItem(props)
        assert stac_ingest.metadata_from_stac_item(item)["level"] == "L2A"

    def test_level_l1c_normalised(self):
        item = RichDummyItem({**self._FULL_PROPS, "s2:product_type": "MSIL1C"})
        assert stac_ingest.metadata_from_stac_item(item)["level"] == "L1C"

    def test_date_from_item_datetime(self):
        dt = datetime(2025, 9, 25, 10, 30, 0, tzinfo=timezone.utc)
        item = RichDummyItem(self._FULL_PROPS, dt=dt)
        m = stac_ingest.metadata_from_stac_item(item)
        assert m["date"] == "2025-09-25"

    def test_date_from_property_when_datetime_none(self):
        props = {**self._FULL_PROPS, "datetime": "2023-10-01T09:00:00Z"}
        item = RichDummyItem(props, dt=None)
        m = stac_ingest.metadata_from_stac_item(item)
        assert m["date"] == "2023-10-01"

    def test_defaults_when_optional_props_absent(self):
        """Bare-minimum item (cloud cover only) should not raise."""
        item = RichDummyItem({"eo:cloud_cover": 5.0})
        m = stac_ingest.metadata_from_stac_item(item)
        assert m["solar_zenith_deg"] == pytest.approx(40.0)
        assert m["crs"] == "EPSG:32629"
        assert m["level"] == "L2A"
        assert m["cloud_cover_pct"] == pytest.approx(5.0)

    def test_explicit_null_properties_fall_back_to_defaults(self):
        props = {
            "eo:cloud_cover": None,
            "view:sun_elevation": None,
            "view:sun_azimuth": None,
            "view:incidence_angle": None,
            "view:azimuth": None,
            "proj:epsg": None,
            "s2:product_type": None,
            "datetime": None,
        }
        m = stac_ingest.metadata_from_stac_item(RichDummyItem(props))
        assert m["solar_zenith_deg"] == pytest.approx(40.0)
        assert m["solar_azimuth_deg"] == pytest.approx(150.0)
        assert m["cloud_cover_pct"] == pytest.approx(2.0)
        assert m["crs"] == "EPSG:32629"
        assert m["level"] == "L2A"
        assert m["date"] == ""

    def test_earth_search_v1_product_type_normalised(self):
        item = RichDummyItem({**self._FULL_PROPS, "s2:product_type": "S2MSI2A"})
        assert stac_ingest.metadata_from_stac_item(item)["level"] == "L2A"

    def test_planetary_computer_solar_angle_keys(self):
        props = {"eo:cloud_cover": 3.0, "s2:mean_solar_zenith": 42.413, "s2:mean_solar_azimuth": 160.459}
        m = stac_ingest.metadata_from_stac_item(RichDummyItem(props))
        assert m["solar_zenith_deg"] == pytest.approx(42.413)
        assert m["solar_azimuth_deg"] == pytest.approx(160.459)

    def test_view_sun_elevation_preferred_over_pc_zenith(self):
        props = {"view:sun_elevation": 50.0, "s2:mean_solar_zenith": 10.0}
        m = stac_ingest.metadata_from_stac_item(RichDummyItem(props))
        assert m["solar_zenith_deg"] == pytest.approx(40.0)

    def test_output_values_are_rounded_floats(self):
        item = RichDummyItem(self._FULL_PROPS)
        m = stac_ingest.metadata_from_stac_item(item)
        for key in ("solar_zenith_deg", "solar_azimuth_deg",
                    "satellite_zenith_deg", "satellite_azimuth_deg", "cloud_cover_pct"):
            val = m[key]
            assert isinstance(val, float), f"{key} should be float"
            # Rounded to 3 decimal places — no more than 3 digits after the dot
            assert round(val, 3) == val
