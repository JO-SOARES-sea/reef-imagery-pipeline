import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.reef_bathy_module import compute_s2_depth_inversion
from src.stumpf_emodnet_calibration import (
    calibrate_stumpf_vs_emodnet,
    stumpf_log_ratio,
    validate_reprojected_emodnet,
)


def write_test_tif(path, arr, crs="EPSG:32629", transform=None, nodata=-9999.0):
    if transform is None:
        transform = Affine.translation(500000.0, 4100000.0) * Affine.scale(10.0, -10.0)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# validate_reprojected_emodnet — sign convention and pixel count checks
# ---------------------------------------------------------------------------

def test_validate_reprojected_emodnet_rejects_empty(tmp_path):
    empty_raster = tmp_path / "emodnet_empty.tif"
    arr = np.full((8, 8), np.nan, dtype=np.float32)
    write_test_tif(empty_raster, arr)

    with pytest.raises(ValueError, match="few valid pixels"):
        validate_reprojected_emodnet(str(empty_raster), min_valid_pixels=1)


def test_validate_reprojected_emodnet_rejects_positive_depths(tmp_path):
    """Raster where all marine pixels (< 5 m) are positive must raise ValueError.

    Uses values in [0, 4] so they pass the marine-pixel filter (< 5 m) but are
    all non-negative — the sign is clearly wrong.
    """
    raster = tmp_path / "emodnet_positive.tif"
    arr = np.full((10, 10), 2.0, dtype=np.float32)  # positive, below 5 m → marine filter keeps these
    write_test_tif(raster, arr)
    with pytest.raises(ValueError, match="sign convention"):
        validate_reprojected_emodnet(str(raster), min_valid_pixels=1)


def test_validate_reprojected_emodnet_accepts_mostly_land_with_some_sea(tmp_path):
    """Coastal tile: mostly positive land (>5 m) with a few valid marine pixels.

    The old median check would fail this because the median is positive.
    The new check only looks at pixels < 5 m, so negative sea pixels pass.
    """
    raster = tmp_path / "emodnet_coastal.tif"
    arr = np.full((10, 10), 50.0, dtype=np.float32)  # land
    arr[8:, 8:] = -8.0  # small sea corner — negative, correct sign
    write_test_tif(raster, arr)
    assert validate_reprojected_emodnet(str(raster), min_valid_pixels=1) is True


def test_validate_reprojected_emodnet_accepts_negative_depths(tmp_path):
    """Raster with negative depths (below sea level) must pass validation."""
    raster = tmp_path / "emodnet_valid.tif"
    arr = np.linspace(-5.0, -20.0, num=100, dtype=np.float32).reshape(10, 10)
    write_test_tif(raster, arr)
    assert validate_reprojected_emodnet(str(raster), min_valid_pixels=1) is True


# ---------------------------------------------------------------------------
# stumpf_log_ratio — boundary conditions
# ---------------------------------------------------------------------------

def test_stumpf_log_ratio_depth_zero_boundary():
    """At depth=0 (surface) reflectances are equal → ratio ≈ 1.0."""
    equal = np.full((4, 4), 0.05, dtype=np.float32)
    ratio = stumpf_log_ratio(equal, equal)
    np.testing.assert_allclose(ratio, 1.0, atol=1e-5)


def test_stumpf_log_ratio_zero_input_no_crash():
    """Zero input (all-black pixel) must not raise, thanks to epsilon clip."""
    zeros = np.zeros((4, 4), dtype=np.float32)
    ones = np.full((4, 4), 0.05, dtype=np.float32)
    ratio = stumpf_log_ratio(zeros, ones)
    assert np.all(np.isfinite(ratio)), "Zero-input ratio should be finite"


def test_stumpf_log_ratio_max_valid_depth():
    """Deep clear water has low blue/green → ratio < 1; should stay finite."""
    b_blue = np.full((4, 4), 0.01, dtype=np.float32)
    b_green = np.full((4, 4), 0.05, dtype=np.float32)
    ratio = stumpf_log_ratio(b_blue, b_green)
    assert np.all(np.isfinite(ratio)), "Deep-water ratio should be finite"
    assert np.all(ratio < 1.0), "Blue < Green ⟹ ratio < 1 for deep water"


# ---------------------------------------------------------------------------
# calibrate_stumpf_vs_emodnet — error paths
# ---------------------------------------------------------------------------

def test_calibrate_raises_on_too_few_training_points(tmp_path):
    """Calibration must raise ValueError when fewer than 100 valid samples exist."""
    b02_path = tmp_path / "b02.tif"
    b03_path = tmp_path / "b03.tif"
    emodnet_path = tmp_path / "emodnet.tif"
    out_path = tmp_path / "sdb_out.tif"

    # EMODnet all land (0.0) — no pixels fall in the -5 to -20 m training window.
    b02 = np.full((12, 12), 0.07, dtype=np.float32)
    b03 = np.full((12, 12), 0.03, dtype=np.float32)
    emodnet = np.zeros((12, 12), dtype=np.float32)  # all "land" — depth = 0

    write_test_tif(b02_path, b02)
    write_test_tif(b03_path, b03)
    write_test_tif(emodnet_path, emodnet)

    with pytest.raises(ValueError, match="[Ii]nsuf"):
        calibrate_stumpf_vs_emodnet(
            str(b02_path), str(b03_path), str(emodnet_path), str(out_path)
        )


# ---------------------------------------------------------------------------
# compute_s2_depth_inversion — integration tests
# ---------------------------------------------------------------------------

def test_compute_s2_depth_inversion_uses_emodnet_calibration(tmp_path):
    b02_path = tmp_path / "S2_B02_test.tif"
    b03_path = tmp_path / "S2_B03_test.tif"
    emodnet_path = tmp_path / "bathy_emodnet_test.tif"

    b02 = np.full((12, 12), 0.07, dtype=np.float32)
    b03 = np.full((12, 12), 0.03, dtype=np.float32)
    emodnet = np.linspace(-6.0, -18.0, num=144, dtype=np.float32).reshape(12, 12)

    write_test_tif(b02_path, b02)
    write_test_tif(b03_path, b03)
    write_test_tif(emodnet_path, emodnet)

    output_path = compute_s2_depth_inversion(
        str(b02_path),
        str(b03_path),
        str(tmp_path),
        date_str="test",
    )

    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("bathy_s2_stumpf_test.tif")

    with rasterio.open(output_path) as src:
        depth_arr = src.read(1)

    assert np.isfinite(depth_arr).any(), "Output depth raster should contain valid depth values"


def test_compute_s2_depth_inversion_falls_back_with_invalid_emodnet(tmp_path):
    b02_path = tmp_path / "S2_B02_test.tif"
    b03_path = tmp_path / "S2_B03_test.tif"
    emodnet_path = tmp_path / "bathy_emodnet_test.tif"

    b02 = np.full((12, 12), 0.07, dtype=np.float32)
    b03 = np.full((12, 12), 0.03, dtype=np.float32)
    emodnet = np.full((12, 12), np.nan, dtype=np.float32)
    emodnet[0, 0] = -10.0

    write_test_tif(b02_path, b02)
    write_test_tif(b03_path, b03)
    write_test_tif(emodnet_path, emodnet)

    output_path = compute_s2_depth_inversion(
        str(b02_path),
        str(b03_path),
        str(tmp_path),
        date_str="test",
    )

    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("bathy_s2_stumpf_test.tif")

    with rasterio.open(output_path) as src:
        depth_arr = src.read(1)

    assert np.isfinite(depth_arr).any(), "Fallback output depth raster should contain valid depth values"
