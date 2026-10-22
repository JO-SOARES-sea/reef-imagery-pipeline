"""
test_bathy_calibrator.py
========================
Pytest-based tests for bathy_calibrator.py (converted from ok()/fail() runner).

All tests that hit the IH/DGRM network API or require pre-downloaded raster
files from reef_Output_Master/ are marked @pytest.mark.network and skipped in
offline CI.  Run them locally with:

    pytest tests/test_bathy_calibrator.py -m network -v
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.warp import transform_bounds

from src.constants import STUMPF_LOG_EPSILON, STUMPF_N

ROOT = Path(__file__).parent.parent
MASTER = ROOT / "reef_Output_Master"

BUF_M = 3000.0
DEG = BUF_M / 111_000.0

M0_DEFAULT = -16.0
M1_DEFAULT = 20.0

SPOTS = [
    {
        "name": "Pedra do Alto (2024-09-30)",
        "dir": MASTER / "reef_output_ai_prediction_spot",
        "b02": "S2_B02_20240930.tif",
        "b03": "S2_B03_20240930.tif",
        "lat": 37.0636, "lon": -8.2193,
        "primary_isobath": 10,
        "expect_zone": "very_shallow",
        "bias_threshold_m": 3.5,
        "rmse_threshold_m": 5.0,
    },
    {
        "name": "Mar 2022 New Spot (2022-02-28)",
        "dir": MASTER / "reef_output_mar_2022_new_spot",
        "b02": "S2_B02_20220228.tif",
        "b03": "S2_B03_20220228.tif",
        "lat": 37.0581, "lon": -8.2098,
        "primary_isobath": 10,
        "expect_zone": "nearshore_mid",
        "bias_threshold_m": 9.5,
        "rmse_threshold_m": 10.0,
    },
    {
        "name": "Aug 2022 Target (2022-08-12)",
        "dir": MASTER / "reef_output_aug_2022_target1",
        "b02": "S2_B02_20220812.tif",
        "b03": "S2_B03_20220812.tif",
        "lat": 37.0636, "lon": -8.2193,
        "primary_isobath": 20,
        "expect_zone": "very_shallow",
        "bias_threshold_m": 10.0,
        "rmse_threshold_m": 12.0,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raster_wgs84(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326",
            src.bounds.left, src.bounds.bottom,
            src.bounds.right, src.bounds.top,
        )
    if np.nanmax(arr) > 2.0:
        arr /= 10000.0
    return arr, profile, (south, west, north, east)


def _compute_sdb(b02, b03, m0, m1, n=STUMPF_N):
    eps = STUMPF_LOG_EPSILON
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (b02 > eps) & (b03 > eps),
            np.log(n * b02 + eps) / (np.log(n * b03 + eps) + eps),
            np.nan,
        )
    return np.clip(m1 * ratio + m0, 0, 40).astype(np.float32)


def _skip_if_missing(spot):
    b02_path = spot["dir"] / spot["b02"]
    if not b02_path.exists():
        pytest.skip(f"Test data not found: {b02_path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_zone_classification():
    """classify_benthic_zone returns expected zone label for known Algarve spots."""
    from src.bathy_calibrator import classify_benthic_zone, fetch_isobaths_for_bbox

    expected = [
        (37.0636, -8.2193, "very_shallow"),
        (37.0468, -7.6603, "shallow_reef"),
        (37.0581, -8.2098, "nearshore_mid"),
    ]
    for lat, lon, exp_zone in expected:
        feats = fetch_isobaths_for_bbox(lon - DEG, lat - DEG, lon + DEG, lat + DEG)
        z = classify_benthic_zone(lon, lat, feats)
        assert z["zone"] == exp_zone, (
            f"({lat},{lon}): expected zone='{exp_zone}', got '{z['zone']}'"
        )
        assert z["optically_viable"], f"({lat},{lon}) should be optically viable"


@pytest.mark.network
@pytest.mark.parametrize("spot", SPOTS, ids=[s["name"] for s in SPOTS])
def test_buffer_sampling_improvement(spot):
    """Buffer sampling produces >= as many isobath samples as point sampling."""
    _skip_if_missing(spot)
    from src.bathy_calibrator import (
        BENTHIC_ISOBATHS,
        BUF_PIX,
        _sample_pixels_near_isobath,
        fetch_isobaths_for_bbox,
    )
    b02, _, bounds = _load_raster_wgs84(spot["dir"] / spot["b02"])
    b03, _, _ = _load_raster_wgs84(spot["dir"] / spot["b03"])
    lat, lon = spot["lat"], spot["lon"]
    feats = fetch_isobaths_for_bbox(lon - DEG, lat - DEG, lon + DEG, lat + DEG)

    for depth in BENTHIC_ISOBATHS:
        n_buf = len(_sample_pixels_near_isobath(b02, b03, feats, float(depth), bounds, buf=BUF_PIX))
        n_pt = len(_sample_pixels_near_isobath(b02, b03, feats, float(depth), bounds, buf=0))
        if n_buf == 0 and n_pt == 0:
            continue  # isobath not in raster bbox — skip silently
        assert n_buf >= n_pt, (
            f"{spot['name']} depth={depth}m: buffer ({n_buf}) < point ({n_pt}) samples"
        )


@pytest.mark.network
@pytest.mark.parametrize("spot", SPOTS[1:], ids=[s["name"] for s in SPOTS[1:]])
def test_single_isobath_offset_calibration(spot):
    """Single-isobath offset calibration sets m1=STUMPF_M1_LITERATURE."""
    _skip_if_missing(spot)
    from src.bathy_calibrator import (
        STUMPF_M1_LITERATURE,
        calibrate_stumpf_from_isobaths,
        fetch_isobaths_for_bbox,
    )
    b02, _, bounds = _load_raster_wgs84(spot["dir"] / spot["b02"])
    b03, _, _ = _load_raster_wgs84(spot["dir"] / spot["b03"])
    lat, lon = spot["lat"], spot["lon"]
    feats = fetch_isobaths_for_bbox(lon - DEG, lat - DEG, lon + DEG, lat + DEG)
    _m0, m1, cal = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)

    if not cal["calibrated"]:
        pytest.skip(f"{spot['name']}: calibration skipped — {cal.get('reason', '?')}")

    if "offset" in cal.get("method", ""):
        assert abs(m1 - STUMPF_M1_LITERATURE) < 1e-6, (
            f"offset calibration: m1={m1} should equal STUMPF_M1_LITERATURE={STUMPF_M1_LITERATURE}"
        )


@pytest.mark.network
def test_before_after_bias():
    """IH-calibrated Stumpf reduces mean |bias| across all spots."""
    from src.bathy_calibrator import (
        calibrate_stumpf_from_isobaths,
        fetch_isobaths_for_bbox,
        validate_sdb_vs_chart,
    )
    all_before, all_after = [], []

    for spot in SPOTS:
        _skip_if_missing(spot)
        b02, _, bounds = _load_raster_wgs84(spot["dir"] / spot["b02"])
        b03, _, _ = _load_raster_wgs84(spot["dir"] / spot["b03"])
        lat, lon = spot["lat"], spot["lon"]
        feats = fetch_isobaths_for_bbox(lon - DEG, lat - DEG, lon + DEG, lat + DEG)

        sdb_before = _compute_sdb(b02, b03, M0_DEFAULT, M1_DEFAULT)
        val_before = validate_sdb_vs_chart(sdb_before, feats, bounds)
        bias_b = val_before.get("overall", {}).get("overall_bias_m")

        m0, m1, _ = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)
        sdb_after = _compute_sdb(b02, b03, m0, m1)
        val_after = validate_sdb_vs_chart(sdb_after, feats, bounds)
        bias_a = val_after.get("overall", {}).get("overall_bias_m")

        if bias_b is not None and bias_a is not None:
            all_before.append(abs(bias_b))
            all_after.append(abs(bias_a))

    if not all_before:
        pytest.skip("No bias data available (no valid test data found)")

    mean_before = np.mean(all_before)
    mean_after = np.mean(all_after)
    assert abs(mean_after) < abs(mean_before), (
        f"Calibration should reduce mean |bias|: before={mean_before:.2f}m, after={mean_after:.2f}m"
    )


@pytest.mark.network
@pytest.mark.parametrize("spot", SPOTS, ids=[s["name"] for s in SPOTS])
def test_per_spot_bias_threshold(spot):
    """Per-spot bias and RMSE must stay below configured thresholds after calibration."""
    _skip_if_missing(spot)
    from src.bathy_calibrator import (
        calibrate_stumpf_from_isobaths,
        fetch_isobaths_for_bbox,
        validate_sdb_vs_chart,
    )
    b02, _, bounds = _load_raster_wgs84(spot["dir"] / spot["b02"])
    b03, _, _ = _load_raster_wgs84(spot["dir"] / spot["b03"])
    lat, lon = spot["lat"], spot["lon"]
    feats = fetch_isobaths_for_bbox(lon - DEG, lat - DEG, lon + DEG, lat + DEG)
    m0, m1, _ = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)
    sdb = _compute_sdb(b02, b03, m0, m1)
    ov = validate_sdb_vs_chart(sdb, feats, bounds).get("overall", {})

    bias_a = ov.get("overall_bias_m")
    rmse_a = ov.get("overall_rmse_m")

    if bias_a is None:
        pytest.skip(f"{spot['name']}: no bias data")
    assert abs(bias_a) < spot["bias_threshold_m"], (
        f"{spot['name']}: |bias|={abs(bias_a):.2f}m >= {spot['bias_threshold_m']}m"
    )
    if rmse_a is not None:
        assert rmse_a < spot["rmse_threshold_m"], (
            f"{spot['name']}: RMSE={rmse_a:.2f}m >= {spot['rmse_threshold_m']}m"
        )


@pytest.mark.network
@pytest.mark.parametrize("spot", SPOTS, ids=[s["name"] for s in SPOTS])
def test_sdb_depth_distribution(spot):
    """SDB median depth must be in physically plausible range [0–35 m]."""
    _skip_if_missing(spot)
    from src.bathy_calibrator import (
        calibrate_stumpf_from_isobaths,
        fetch_isobaths_for_bbox,
    )
    b02, _, bounds = _load_raster_wgs84(spot["dir"] / spot["b02"])
    b03, _, _ = _load_raster_wgs84(spot["dir"] / spot["b03"])
    lat, lon = spot["lat"], spot["lon"]
    feats = fetch_isobaths_for_bbox(lon - DEG, lat - DEG, lon + DEG, lat + DEG)
    m0, m1, _ = calibrate_stumpf_from_isobaths(b02, b03, feats, bounds)
    sdb = _compute_sdb(b02, b03, m0, m1)
    valid = sdb[sdb > 0]
    if valid.size == 0:
        pytest.skip(f"{spot['name']}: no valid SDB pixels")
    median = float(np.median(valid))
    assert 0 < median < 35, (
        f"{spot['name']}: SDB median={median:.1f}m outside physical range [0–35m]"
    )
