"""Tests for DGTSentinelIntegrator.fetch_sentinel2_copernicus xarray conversion."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Stub optional heavy dependencies not present in the test environment
# ---------------------------------------------------------------------------
def _make_module_stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


if "geopandas" not in sys.modules:
    sys.modules["geopandas"] = _make_module_stub("geopandas", GeoDataFrame=MagicMock)

if "shapely" not in sys.modules:
    sys.modules["shapely"] = _make_module_stub("shapely")
if "shapely.geometry" not in sys.modules:
    sys.modules["shapely.geometry"] = _make_module_stub("shapely.geometry", box=MagicMock())

if "requests" not in sys.modules:
    sys.modules["requests"] = _make_module_stub("requests", get=MagicMock())

from src.dgt_sentinel_integrator import DGTSentinelIntegrator  # noqa: E402

BBOX = (-8.25, 37.04, -8.17, 37.10)


def _make_integrator():
    return DGTSentinelIntegrator(bbox=BBOX, output_dir="/tmp/test_dgt")


def _stub_sentinelhub():
    """Return a minimal sentinelhub stub that passes the import guard."""
    sh = types.ModuleType("sentinelhub")
    for name in ("SentinelHubRequest", "DataCollection", "MimeType", "BBox", "CRS"):
        setattr(sh, name, MagicMock())
    return sh


class TestFetchSentinel2CopernicusConversion:
    def _run_with_fake_image(self, shape=(512, 512, 6)):
        """Patch sentinelhub so get_data() returns a synthetic numpy image."""
        sh = _stub_sentinelhub()
        fake_img = np.random.rand(*shape).astype("float32")
        request_instance = MagicMock()
        request_instance.get_data.return_value = [fake_img]
        sh.SentinelHubRequest.return_value = request_instance

        integrator = _make_integrator()
        with patch.dict(sys.modules, {"sentinelhub": sh}):
            da = integrator.fetch_sentinel2_copernicus()
        return da

    def test_returns_dataarray(self):
        import xarray as xr
        da = self._run_with_fake_image()
        assert isinstance(da, xr.DataArray)

    def test_band_count(self):
        da = self._run_with_fake_image()
        assert da.sizes["band"] == 6

    def test_band_names(self):
        da = self._run_with_fake_image()
        assert list(da.coords["band"].values) == ["B02", "B03", "B04", "B08", "B11", "B12"]

    def test_crs_is_epsg3763(self):
        da = self._run_with_fake_image()
        assert da.rio.crs is not None
        assert "3763" in str(da.rio.crs)

    def test_sentinelhub_import_error_returns_none(self):
        integrator = _make_integrator()
        with patch.dict(sys.modules, {"sentinelhub": None}):
            da = integrator.fetch_sentinel2_copernicus()
        assert da is None

    def test_get_data_exception_returns_none(self):
        sh = _stub_sentinelhub()
        request_instance = MagicMock()
        request_instance.get_data.side_effect = RuntimeError("network error")
        sh.SentinelHubRequest.return_value = request_instance

        integrator = _make_integrator()
        with patch.dict(sys.modules, {"sentinelhub": sh}):
            da = integrator.fetch_sentinel2_copernicus()
        assert da is None
