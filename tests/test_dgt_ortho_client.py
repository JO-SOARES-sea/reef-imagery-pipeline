"""Tests for DGTOrthoClient — offline (mocked) and network variants."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dgt_ortho_client import (
    COLLECTIONS_BY_YEAR,
    ORTHO_COLLECTIONS,
    DGTOrthoClient,
)

# ── helpers ─────────────────────────────────────────────────────────────────

ALGARVE_BBOX = (-8.25, 37.04, -8.17, 37.10)

_FAKE_FEATURE = {
    "id": "ORTOSAT-2023-cog-30cm-605-4",
    "assets": {
        "visual": {
            "href": "https://stor-002.a.acnca.pt:9000/orto2023/tile_605-4.tif"
        }
    },
}

_FAKE_FC = {"features": [_FAKE_FEATURE], "context": {"returned": 1}}


def _mock_stac_ok(url, params=None, timeout=30):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _FAKE_FC
    resp.raise_for_status = MagicMock()
    return resp


def _mock_stac_empty(url, params=None, timeout=30):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"features": [], "context": {"returned": 0}}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_stac_error(url, params=None, timeout=30):
    import requests as _req
    raise _req.RequestException("connection refused")


# ── unit tests ───────────────────────────────────────────────────────────────

class TestDGTOrthoClientInit:
    def test_output_dir_created(self, tmp_path):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path / "out"))
        assert (tmp_path / "out").is_dir()

    def test_collections_catalog(self):
        assert "ORTOSAT-2023" in ORTHO_COLLECTIONS
        assert "ORTOS-1995" in ORTHO_COLLECTIONS
        assert len(COLLECTIONS_BY_YEAR) == len(ORTHO_COLLECTIONS)


class TestListTiles:
    def test_returns_features_on_success(self, tmp_path):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        with patch("src.dgt_ortho_client.requests.get", side_effect=_mock_stac_ok):
            items = client.list_tiles("ORTOSAT-2023")
        assert len(items) == 1
        assert items[0]["id"] == "ORTOSAT-2023-cog-30cm-605-4"

    def test_returns_empty_on_network_error(self, tmp_path):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        with patch("src.dgt_ortho_client.requests.get", side_effect=_mock_stac_error):
            items = client.list_tiles("ORTOSAT-2023")
        assert items == []

    def test_returns_empty_when_no_tiles(self, tmp_path):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        with patch("src.dgt_ortho_client.requests.get", side_effect=_mock_stac_empty):
            items = client.list_tiles("ORTOSAT-2023")
        assert items == []

    def test_unknown_collection_still_queries(self, tmp_path):
        """Unknown collection names fall through to the raw ID — no KeyError."""
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        with patch("src.dgt_ortho_client.requests.get", side_effect=_mock_stac_ok):
            items = client.list_tiles("NONEXISTENT-COLLECTION")
        assert isinstance(items, list)


class TestBestAvailable:
    def test_returns_first_collection_with_tiles(self, tmp_path):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        # First call returns item; rest are empty (should short-circuit)
        responses = [_FAKE_FC] + [{"features": [], "context": {"returned": 0}}] * 20

        def _side(url, params=None, timeout=30):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = responses.pop(0) if responses else {"features": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.dgt_ortho_client.requests.get", side_effect=_side):
            best = client.best_available()
        assert best == COLLECTIONS_BY_YEAR[0]

    def test_returns_none_when_all_empty(self, tmp_path):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        with patch("src.dgt_ortho_client.requests.get", side_effect=_mock_stac_empty):
            best = client.best_available()
        assert best is None


class TestDownloadTiles:
    def test_skips_on_403(self, tmp_path, caplog, monkeypatch):
        """403 from signed download URL → logs error, returns empty list, no exception."""
        import src.dgt_cdd_auth as _auth
        import src.dgt_ortho_client as _mod
        monkeypatch.setattr(_mod, "_DGT_USER", "test_user")
        monkeypatch.setattr(_mod, "_DGT_PASS", "test_pass")
        monkeypatch.setattr(_auth, "_session", None)

        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))

        # Mock authenticated session with connect.sid cookie
        fake_session = MagicMock()
        fake_session.cookies = {"connect.sid": "fake-sid"}

        # Item endpoint returns a signed URL
        signed_item_resp = MagicMock(status_code=200)
        signed_item_resp.json.return_value = {
            "data": {
                "id": "ORTOSAT-2023-cog-30cm-605-4",
                "assets": {"visual": {"href": "/dgt-be/v1/download/abc123"}}
            }
        }
        # Download returns 403
        stream_resp = MagicMock(status_code=403)
        stream_resp.__enter__ = lambda s: s
        stream_resp.__exit__ = MagicMock(return_value=False)

        fake_session.get.side_effect = [
            stream_resp,        # actual download (signed_url now mocked at module level)
        ]

        with patch("src.dgt_ortho_client.requests.get") as mock_get, \
             patch("src.dgt_cdd_auth._login_keycloak_direct", return_value=fake_session), \
             patch("src.dgt_cdd_auth._login_pkce", return_value=fake_session), \
             patch("src.dgt_cdd_auth.get_signed_url",
                   return_value="https://cdd.dgterritorio.gov.pt/dgt-be/v1/download/abc123"):
            mock_get.return_value = MagicMock(**{
                "status_code": 200, "json.return_value": _FAKE_FC,
                "raise_for_status": MagicMock(),
            })
            paths = client.download_tiles("ORTOSAT-2023", limit=1)

        assert paths == []
        assert "403" in caplog.text

    def test_uses_cache(self, tmp_path):
        """If the file already exists, it is returned without a download request."""
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path), cache=True)

        # Pre-create the cached file
        tiles_dir = tmp_path / "ORTOSAT-2023"
        tiles_dir.mkdir()
        cached_file = tiles_dir / "tile_605-4.tif"
        cached_file.write_bytes(b"fake")

        with patch("src.dgt_ortho_client.requests.get") as mock_get:
            # Only the STAC discovery call should be made
            mock_get.return_value = MagicMock(**{
                "status_code": 200,
                "json.return_value": _FAKE_FC,
                "raise_for_status": MagicMock(),
            })
            paths = client.download_tiles("ORTOSAT-2023", limit=1)

        assert len(paths) == 1
        assert paths[0] == cached_file
        # download GET should only have been called once (STAC), not twice
        assert mock_get.call_count == 1


class TestNDVI:
    def test_ndvi_computation(self, tmp_path):
        """NDVI=(NIR-Red)/(NIR+Red) computed correctly from a synthetic 4-band GeoTIFF."""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds

        # Synthetic 4-band tile: R=50, G=80, B=100, NIR=120
        transform = from_bounds(-8.25, 37.04, -8.17, 37.10, 10, 10)
        profile = {
            "driver": "GTiff", "dtype": "uint8", "width": 10, "height": 10,
            "count": 4, "crs": "EPSG:4326", "transform": transform,
        }
        tile_path = tmp_path / "synthetic.tif"
        with rasterio.open(str(tile_path), "w", **profile) as dst:
            dst.write(np.full((10, 10), 50, dtype=np.uint8), 1)   # Red
            dst.write(np.full((10, 10), 80, dtype=np.uint8), 2)   # Green
            dst.write(np.full((10, 10), 100, dtype=np.uint8), 3)  # Blue
            dst.write(np.full((10, 10), 120, dtype=np.uint8), 4)  # NIR

        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        ndvi_path = client.compute_ndvi([tile_path], output_stem="test")

        assert ndvi_path is not None and ndvi_path.exists()
        with rasterio.open(str(ndvi_path)) as src:
            ndvi = src.read(1)
        expected = (120 - 50) / (120 + 50)  # ≈ 0.412
        assert abs(float(np.nanmean(ndvi)) - expected) < 0.01

    def test_ndvi_requires_4_bands(self, tmp_path):
        """Single-band input → returns None, no crash."""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(-8.25, 37.04, -8.17, 37.10, 4, 4)
        profile = {
            "driver": "GTiff", "dtype": "uint8", "width": 4, "height": 4,
            "count": 1, "crs": "EPSG:4326", "transform": transform,
        }
        tile_path = tmp_path / "single_band.tif"
        with rasterio.open(str(tile_path), "w", **profile) as dst:
            dst.write(np.full((4, 4), 100, dtype=np.uint8), 1)

        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir=str(tmp_path))
        result = client.compute_ndvi([tile_path], output_stem="bad")
        assert result is None


# ── network tests (skipped in CI) ────────────────────────────────────────────

@pytest.mark.network
class TestDGTSTACLive:
    """Verify the DGT STAC server is reachable and returns the expected collections."""

    def test_stac_collections_reachable(self):
        import requests
        r = requests.get("https://dgt-be.a.incd.pt:8081/collections", timeout=15)
        assert r.status_code == 200
        cols = {c["id"] for c in r.json().get("collections", [])}
        for expected in ["MDT-50cm", "MDS-50cm", "ORTOSAT-2023", "MosaicoS2-2024"]:
            assert expected in cols, f"Collection {expected} missing from DGT STAC"

    def test_list_tiles_algarve(self):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir="/tmp/dgt_ortho_test")
        items = client.list_tiles("ORTOSAT-2023", limit=5)
        assert len(items) >= 1
        assert "assets" in items[0]

    def test_best_available_returns_ortosat(self):
        client = DGTOrthoClient(bbox=ALGARVE_BBOX, output_dir="/tmp/dgt_ortho_test")
        best = client.best_available()
        assert best is not None
        assert best in COLLECTIONS_BY_YEAR
