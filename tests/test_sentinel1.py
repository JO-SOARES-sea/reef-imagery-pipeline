from unittest.mock import MagicMock, patch

import pytest

from src.sentinel1_roughness import extract_sigma0_at_point, roughness_from_sigma0


def test_roughness_from_sigma0_calm():
    # VV and VH are equal or close -> roughness close to 0 (calm)
    res = roughness_from_sigma0(0.01, 0.009)
    assert res["sea_state"] == "calm"
    assert res["roughness"] is not None
    assert res["vv_linear"] == 0.01
    assert res["vh_linear"] == 0.009

def test_roughness_from_sigma0_moderate():
    # Moderate roughness
    res = roughness_from_sigma0(0.01, 0.007)
    assert res["sea_state"] == "moderate"

def test_roughness_from_sigma0_rough():
    # Rough sea
    res = roughness_from_sigma0(0.01, 0.005)
    assert res["sea_state"] == "rough"

def test_roughness_from_sigma0_invalid():
    # Negative or zero inputs should return None roughness and unknown sea_state
    res = roughness_from_sigma0(-0.01, 0.005)
    assert res["roughness"] is None
    assert res["sea_state"] == "unknown"

@patch("rasterio.open")
@patch("rasterio.warp.reproject")
@patch("rasterio.band")
def test_extract_sigma0_at_point_success(mock_band, mock_reproject, mock_open):
    # Mock rasterio open to return mock src
    mock_src = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_src

    # Mock reproject to fill the destination array with 100.0
    def side_effect(*args, **kwargs):
        dest = kwargs.get("destination")
        if dest is not None:
            dest.fill(100.0)
    mock_reproject.side_effect = side_effect

    mock_item = {
        "assets": {
            "vv": "s3://sentinel-s1/measurement/iw-vv.tiff",
            "vh": "s3://sentinel-s1/measurement/iw-vh.tiff"
        }
    }

    with patch("rasterio.Env"):
        res = extract_sigma0_at_point(mock_item, -8.21, 37.06)

    assert res is not None
    assert "vv" in res
    assert "vh" in res
    # amplitude = 100 -> amplitude^2 / 1e8 = 10000 / 1e8 = 0.0001
    assert pytest.approx(res["vv"], 1e-6) == 0.0001


def test_extract_sigma0_missing_vv_returns_none():
    """Item with no VV asset should return None, not raise."""
    item = {"assets": {"vh": "s3://fake/vh.tiff"}}  # VV missing
    result = extract_sigma0_at_point(item, -8.21, 37.06)
    assert result is None


def test_extract_sigma0_missing_vh_returns_none():
    """Item with no VH asset should return None."""
    item = {"assets": {"vv": "s3://fake/vv.tiff"}}  # VH missing
    result = extract_sigma0_at_point(item, -8.21, 37.06)
    assert result is None


def test_extract_sigma0_empty_assets_returns_none():
    """Item with no assets should return None."""
    result = extract_sigma0_at_point({"assets": {}}, -8.21, 37.06)
    assert result is None


@patch("rasterio.open")
@patch("rasterio.warp.reproject")
@patch("rasterio.band")
def test_extract_sigma0_all_zero_pixels_returns_none(mock_band, mock_reproject, mock_open):
    """When all reprojected pixels are zero (no valid data), vv/vh should be None → return None."""
    mock_src = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_src

    # reproject leaves the destination filled with zeros (default np.zeros)
    # so data_pos = dst_data[dst_data > 0] is empty

    item = {
        "assets": {
            "vv": "s3://fake/vv.tiff",
            "vh": "s3://fake/vh.tiff",
        }
    }
    with patch("rasterio.Env"):
        result = extract_sigma0_at_point(item, -8.21, 37.06)
    assert result is None
