"""Offline e2e-style assertion for run_icesat2_validation.

Feeds synthetic ATL03/ATL08-style photons (via a mocked ``_search_atl08``)
plus a synthetic SDB depth GeoTIFF, then asserts the returned report carries a
finite ``rmse_m`` and that a generous threshold check works.

No network: OpenAltimetry is never contacted (the search is mocked out).
Deliberately NOT marked ``@pytest.mark.network``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import icesat2_validation as iv  # noqa: E402

# A small AOI on the Algarve coast.
_LON0, _LAT0 = -8.21, 37.069
_PX_DEG = 0.0005
_N = 32
# Constant synthetic seafloor depth (metres).
_TRUE_DEPTH_M = 12.0


def _write_synthetic_sdb(path: Path) -> None:
    """Write a flat synthetic SDB depth raster (~constant 12 m) in EPSG:4326."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    arr = np.full((_N, _N), _TRUE_DEPTH_M, dtype=np.float32)
    transform = from_origin(_LON0, _LAT0 + _N * _PX_DEG, _PX_DEG, _PX_DEG)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": _N,
        "width": _N,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def _synthetic_photons(n: int = 40, depth_offset_m: float = 0.5):
    """Synthetic ATL08-style photon dicts near the centre of the raster.

    ICESat-2 elevations are negative below sea surface; the validator converts
    ``-elev`` to a positive depth. We place photons at ~_TRUE_DEPTH plus a
    small offset so RMSE is small but non-zero.
    """
    rng = np.random.default_rng(7)
    cx = _LON0 + (_N / 2) * _PX_DEG
    cy = _LAT0 + (_N / 2) * _PX_DEG
    photons = []
    for _ in range(n):
        lon = cx + (rng.standard_normal() * _PX_DEG * 2)
        lat = cy + (rng.standard_normal() * _PX_DEG * 2)
        depth = _TRUE_DEPTH_M + depth_offset_m + rng.standard_normal() * 0.3
        photons.append({"lat": lat, "lon": lon, "h": -depth})
    return photons


def test_icesat2_validation_reports_finite_rmse_offline(tmp_path):
    sdb_path = tmp_path / "sdb_depth_map.tif"
    _write_synthetic_sdb(sdb_path)
    photons = _synthetic_photons()

    with patch.object(iv, "_search_atl08", return_value=photons):
        report = iv.run_icesat2_validation(
            depth_map_path=sdb_path,
            output_dir=tmp_path / "validation",
            scene_date="2024-09-15",
            bbox=(_LON0, _LAT0, _LON0 + _N * _PX_DEG, _LAT0 + _N * _PX_DEG),
        )

    assert report is not None
    assert report["status"] == "ok"

    # Core assertion: rmse_m present, a finite float.
    assert "rmse_m" in report
    rmse = report["rmse_m"]
    assert isinstance(rmse, float)
    assert math.isfinite(rmse)
    assert rmse >= 0.0

    # bias_m present and finite too.
    assert math.isfinite(report["bias_m"])

    # Generous threshold for the synthetic case (offset ~0.5 m + 0.3 m noise).
    assert rmse < 5.0, f"synthetic RMSE unexpectedly large: {rmse}"

    # Report JSON was written to disk.
    assert (tmp_path / "validation" / "validation_report.json").is_file()


def test_icesat2_validation_skips_when_no_photons_offline(tmp_path):
    """No photons → graceful 'skipped' report, never a crash."""
    sdb_path = tmp_path / "sdb_depth_map.tif"
    _write_synthetic_sdb(sdb_path)

    with patch.object(iv, "_search_atl08", return_value=[]):
        report = iv.run_icesat2_validation(
            depth_map_path=sdb_path,
            output_dir=tmp_path / "validation",
            scene_date="2024-09-15",
            bbox=(_LON0, _LAT0, _LON0 + _N * _PX_DEG, _LAT0 + _N * _PX_DEG),
        )

    assert report is not None
    assert report["status"] == "skipped"


def _write_negative_sdb(path: Path) -> None:
    """Write an SDB raster with NEGATIVE depths (~ -12 m), the real convention."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    arr = np.full((_N, _N), -_TRUE_DEPTH_M, dtype=np.float32)  # negative below surface
    transform = from_origin(_LON0, _LAT0 + _N * _PX_DEG, _PX_DEG, _PX_DEG)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": _N,
        "width": _N,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def test_icesat2_validation_handles_negative_depth_sign(tmp_path):
    """Regression: SDB stored as negative must not yield ~2x-depth bias.

    Before the sign-normalization fix, comparing a -12 m SDB raster against
    +12 m photon depth produced bias ~ -24 m. After the fix, bias is near zero.
    """
    sdb_path = tmp_path / "sdb_depth_neg.tif"
    _write_negative_sdb(sdb_path)
    photons = _synthetic_photons(depth_offset_m=0.0)  # photons at ~true 12 m depth

    with patch.object(iv, "_search_atl08", return_value=photons):
        report = iv.run_icesat2_validation(
            depth_map_path=sdb_path,
            output_dir=tmp_path / "validation",
            scene_date="2024-09-15",
            bbox=(_LON0, _LAT0, _LON0 + _N * _PX_DEG, _LAT0 + _N * _PX_DEG),
        )

    assert report is not None and report["status"] == "ok"
    # The whole point: bias must be small, NOT ~-2*depth (~-24 m).
    assert abs(report["bias_m"]) < 2.0, f"sign bug not fixed: bias={report['bias_m']}"
    assert report["rmse_m"] < 2.0
    # Guard against silently passing because the depth got zeroed out.
    assert report["sdb_depth_mean_m"] > 5.0
