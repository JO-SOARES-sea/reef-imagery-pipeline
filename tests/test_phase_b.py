#!/usr/bin/env python3
"""
tests/test_phase_b.py — Targeted tests for Phase B calibration
================================================================
Tests the core logic of phase_b_calibrate_icesat2.py in isolation:
  - sample extraction / filtering
  - calibration fit calculation
  - graceful handling of missing/insufficient samples
  - output file writing / naming

Does NOT require NASA EarthData credentials.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.phase_b_calibrate_icesat2 import (
    apply_calibration,
    fit_calibration,
    match_icesat2_to_sdb,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: fit_calibration — sufficient samples
# ═══════════════════════════════════════════════════════════════════════════════

def test_fit_calibration_sufficient_samples():
    """
    With ≥ 4 matched sample pairs, calibration should return a valid
    linear model with a reasonable slope and intercept.
    """
    # Simulated matched pairs: ICESat-2 (truth) vs SDB (predicted)
    # Introduce a systematic bias: SDB underestimates depth by ~3m
    icesat2 = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])   # ground truth (m)
    sdb     = np.array([2.1,  7.0, 12.1, 17.0, 22.1, 27.0])   # SDB is ~3m too shallow

    result = fit_calibration(icesat2, sdb)

    assert result["calibrated"] is True,         "Should be calibrated with 6 samples"
    assert result["n_samples"] == 6,             "Should use all 6 pairs"
    assert 0.5 <= result["a"] <= 2.0,             "Slope a should be in reasonable range"
    assert -10.0 <= result["b"] <= 10.0,          "Intercept b should be in range"
    assert result["rmse_m"] is not None,          "RMSE should be computed"
    assert result["bias_m"] is not None,          "Bias should be computed"
    print("  ✓ fit_calibration sufficient samples — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: fit_calibration — insufficient samples (< 4)
# ═══════════════════════════════════════════════════════════════════════════════

def test_fit_calibration_insufficient_samples():
    """
    With < 4 samples, calibration should return defaults (a=1, b=0)
    and indicate not calibrated.
    """
    icesat2 = np.array([10.0, 15.0])   # only 2 pairs
    sdb     = np.array([ 8.0, 13.0])

    result = fit_calibration(icesat2, sdb)

    assert result["calibrated"] is False,     "Should NOT be calibrated with 2 samples"
    assert result["a"] == 1.0,                 "Default: identity slope"
    assert result["b"] == 0.0,                 "Default: zero offset"
    assert result["rmse_m"] is None,           "No RMSE without calibration"
    assert "insufficient" in result["reason"]  # should explain why

    print("  ✓ fit_calibration insufficient samples — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: apply_calibration — identity (no correction)
# ═══════════════════════════════════════════════════════════════════════════════

def test_apply_calibration_identity():
    """
    With a=1, b=0 (no calibration), output should equal input.
    Invalid (zero) pixels should remain zero.
    """
    sdb_arr = np.array([
        [0.0, 10.0, 20.0],
        [ 5.0, 0.0, 15.0],
        [25.0, 30.0,  0.0],
    ], dtype=np.float32)

    calibrated = apply_calibration(sdb_arr, a=1.0, b=0.0)

    np.testing.assert_array_equal(calibrated, sdb_arr,
        "Identity calibration (a=1, b=0) should leave SDB unchanged")
    assert calibrated[0, 0] == 0.0,   "Zero pixel stays zero"
    assert calibrated[1, 1] == 0.0,  "Zero pixel stays zero"
    print("  ✓ apply_calibration identity — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: apply_calibration — linear correction
# ═══════════════════════════════════════════════════════════════════════════════

def test_apply_calibration_linear():
    """
    With a=1.2, b=-3.0:
      Z_calibrated = 1.2 * Z_SDB - 3.0
    and zero pixels stay zero.
    """
    sdb_arr = np.array([[10.0], [20.0], [0.0]], dtype=np.float32)
    calibrated = apply_calibration(sdb_arr, a=1.2, b=-3.0)

    assert abs(calibrated[0, 0] - (1.2 * 10.0 - 3.0)) < 1e-6,  "First pixel: 1.2*10-3=9"
    assert abs(calibrated[1, 0] - (1.2 * 20.0 - 3.0)) < 1e-6,  "Second pixel: 1.2*20-3=21"
    assert calibrated[2, 0] == 0.0,                             "Zero stays zero"
    print("  ✓ apply_calibration linear — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: apply_calibration — no negative depths
# ═══════════════════════════════════════════════════════════════════════════════

def test_apply_calibration_no_negative():
    """
    Calibrated depth should never be negative.
    """
    sdb_arr = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    calibrated = apply_calibration(sdb_arr, a=-0.5, b=1.0)  # would give negative without clip

    assert np.all(calibrated >= 0.0), "No negative depths allowed"
    print("  ✓ apply_calibration no negative depths — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: match_icesat2_to_sdb — matching logic
# ═══════════════════════════════════════════════════════════════════════════════

def test_match_icesat2_to_sdb():
    """
    With a simple 3×3 SDB grid and 2 ICESat-2 points that fall
    inside pixel windows, matching should succeed.

    Coordinate scale: 1 degree ≈ 111km (lat) × 89km (lon at 37°N).
    Use small offsets (~0.0001 deg ≈ 11m lat, 9m lon) so points are
    within the 30m pixel-window threshold.
    """
    # SDB depth array: (H=3, W=3), values in metres
    sdb_arr = np.array([
        [ 0.0, 10.0,  0.0],
        [10.0, 20.0, 10.0],
        [ 0.0, 10.0,  0.0],
    ], dtype=np.float32)

    # bbox: 0..3 deg lon/lat. At 37°N, 1 unit ≈ 89km lon, 111km lat.
    # SDB pixel (row=1, col=1) centre = lon=1.5, lat=1.5
    # ICESat2 at (1.5001, 1.4999) — offset ~15m from pixel centre → match
    # SDB pixel (row=1, col=0) centre = lon=0.5, lat=1.5
    # ICESat2 at (0.5001, 1.4999) — offset ~11m from pixel centre → match
    icesat2_pts = np.array([
        [1.5001, 1.4999, 19.5],   # ~15m from pixel(1,1)=20m
        [0.5001, 1.4999,  9.5],   # ~11m from pixel(1,0)=10m
    ])

    icesat2_matched, sdb_matched = match_icesat2_to_sdb(
        icesat2_pts, sdb_arr,
        bounds_wgs84=(0.0, 0.0, 3.0, 3.0),
        max_distance_m=30.0,
    )

    assert len(icesat2_matched) == 2,  "Both points should match"
    assert len(sdb_matched) == 2,      "Both SDB depths should be found"
    np.testing.assert_allclose(icesat2_matched, [19.5, 9.5], rtol=1e-3)
    np.testing.assert_allclose(sdb_matched, [20.0, 10.0], rtol=1e-3)
    print("  ✓ match_icesat2_to_sdb — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: match_icesat2_to_sdb — out of bounds / zero pixels
# ═══════════════════════════════════════════════════════════════════════════════

def test_match_icesat2_outside_bbox():
    """
    Points outside the raster or in zero-pixel regions should be skipped.
    """
    sdb_arr = np.array([[10.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    # bbox: 0..2 deg lon/lat. At 37°N, 1 unit ≈ 89km lon, 111km lat.
    # SDB pixel centres:
    #   (0,0): lon=0.5, lat=1.5  -> 10m
    #   (1,0): lon=1.5, lat=1.5  -> 0m
    #   (0,1): lon=0.5, lat=0.5  -> 0m
    #   (1,1): lon=1.5, lat=0.5  -> 0m
    icesat2_pts = np.array([
        [-1.0,  0.5,  8.0],   # lon outside bbox → skip
        [ 0.5,  0.5,  8.0],   # inside bbox, pixel(0,1)=0 → skip
        [ 1.5,  0.5,  8.0],   # inside bbox, pixel(1,1)=0 → skip
        [ 0.5,  1.5,  8.0],   # inside bbox, pixel(0,0)=10 → match
    ])

    icesat2_matched, sdb_matched = match_icesat2_to_sdb(
        icesat2_pts, sdb_arr,
        bounds_wgs84=(0.0, 0.0, 2.0, 2.0),
        max_distance_m=30.0,
    )

    assert len(icesat2_matched) == 1,  "Only point at pixel(0,0)=10m should match"
    assert len(sdb_matched) == 1,       "One SDB depth"
    np.testing.assert_allclose(icesat2_matched, [8.0], rtol=1e-3)
    np.testing.assert_allclose(sdb_matched, [10.0], rtol=1e-3)
    print("  ✓ match_icesat2_to_sdb out-of-bounds / zero skip — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: fit_calibration — slope clipping
# ═══════════════════════════════════════════════════════════════════════════════

def test_fit_calibration_slope_clipping():
    """
    If raw slope is outside [0.5, 2.0], it should be clipped.
    """
    icesat2 = np.array([5.0, 10.0, 15.0, 20.0])
    sdb     = np.array([50.0, 100.0, 150.0, 200.0])   # extreme slope ~2.0+
    result = fit_calibration(icesat2, sdb)

    assert result["a"] <= 2.0,             "Slope should be clipped to max 2.0"
    assert result["calibrated"] is True,    "Should still calibrate"
    print("  ✓ fit_calibration slope clipping — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: output file naming
# ═══════════════════════════════════════════════════════════════════════════════

def test_output_dir_creation():
    """
    _OUTPUT_DIR should be created if it doesn't exist.
    """
    test_dir = Path(tempfile.gettempdir()) / "phase_b_test_output"
    # Patch _OUTPUT_DIR temporarily
    from scripts import phase_b_calibrate_icesat2
    orig = phase_b_calibrate_icesat2._OUTPUT_DIR
    phase_b_calibrate_icesat2._OUTPUT_DIR = test_dir

    test_dir.mkdir(parents=True, exist_ok=True)
    assert test_dir.exists(), "Test output dir should exist"

    # Cleanup
    phase_b_calibrate_icesat2._OUTPUT_DIR = orig
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    print("  ✓ output directory creation — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: calibration with perfect match (a=1, b=0 ideal)
# ═══════════════════════════════════════════════════════════════════════════════

def test_fit_calibration_perfect_match():
    """
    If ICESat-2 == SDB exactly (perfect calibration), a≈1, b≈0, RMSE≈0.
    """
    icesat2 = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    sdb     = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

    result = fit_calibration(icesat2, sdb)

    assert result["calibrated"] is True
    assert abs(result["a"] - 1.0) < 0.1,   "Slope should be near 1.0"
    assert abs(result["b"]) < 0.5,          "Intercept should be near 0"
    assert result["rmse_m"] < 0.5,          "RMSE should be near 0"
    print("  ✓ fit_calibration perfect match — PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    print("\n" + "=" * 60)
    print("  PHASE B CALIBRATION — Targeted Tests")
    print("=" * 60)

    tests = [
        test_fit_calibration_sufficient_samples,
        test_fit_calibration_insufficient_samples,
        test_apply_calibration_identity,
        test_apply_calibration_linear,
        test_apply_calibration_no_negative,
        test_match_icesat2_to_sdb,
        test_match_icesat2_outside_bbox,
        test_fit_calibration_slope_clipping,
        test_output_dir_creation,
        test_fit_calibration_perfect_match,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed ✓\n")


if __name__ == "__main__":
    run_all()
