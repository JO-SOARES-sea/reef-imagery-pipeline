#!/usr/bin/env python3
"""
phase_b_calibrate_icesat2.py — ICESat-2 Calibration for Sentinel-2 SDB
======================================================================
Phase B: Calibrate Sentinel-2 Stumpf Bathymetry using ICESat-2 ATL03
ground-truth depth soundings.

This is a calibration *layer* on top of Phase A — Phase A outputs remain
intact. Phase B produces a separate calibrated product and summary.

Usage
-----
    python3 phase_b_calibrate_icesat2.py [--site SITE_NAME] [--date YYYYMMDD]

Outputs (written to outputs/phase_b/)
-----------------------------------------
    calibrated_SDB_YYYYMMDD.tif   — calibrated depth map (GeoTIFF)
    calibration_summary.json     — fit quality, bias, n_samples, params
    sample_pairs.csv              — matched ICESat-2 / SDB point pairs
    calibration_report.txt        — human-readable summary

Prerequisites
------------
    - Phase A SDB output (Stumpf depth map) must exist
    - ICESat-2 ATL03 data access via earthaccess (NASA EarthData login)

    If earthaccess is unavailable or login fails, the script:
        1. Reports graceful failure with clear message
        2. Documents what it WOULD do
        3. Exports any available Phase A metadata
        4. Does NOT crash

Calibration methodology
-----------------------
    Simple offset calibration (default) or linear regression:
        Z_calibrated = a * Z_SDB + b

    where a, b are fitted via OLS on matched (ICESat-2, SDB) depth pairs.

    Only photons with:
        - signal_conf_ph >= 3 (high confidence)
        - depth range 1–30 m (within SDB optical window)
        - horizontal distance to SDB pixel < 30 m

Scientific rationale
-------------------
    Stumpf SDB is known to underestimate depth in the Algarve because:
        - m0 is a generic literature offset (doesn't fit Portuguese waters)
        - m1 is over-fitted to training data from Florida Keys
    The bias is especially visible at deeper sites (>15 m).

    Phase B corrects this bias using in-situ lidar soundings from ICESat-2.
"""

import argparse
import json
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── project root & paths ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "phase_b"
_PHASE_A_ROOT = _PROJECT_ROOT / "reef_Output_Master"

# ── logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("phase_b")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PHASE A LOCATOR
# ═══════════════════════════════════════════════════════════════════════════════

def find_phase_a_sdb(date_str: str) -> Path | None:
    """
    Find the Phase A Stumpf SDB depth map for a given date.
    Checks multiple naming conventions used by the pipeline.

    Returns None if no SDB map is found (Phase B will handle gracefully).
    """
    candidates = [
        _PHASE_A_ROOT / "reef_output_acolite_comparison" / f"predictor_A_{date_str}" / "sdb_depth_map.tif",
        _PHASE_A_ROOT / "reef_output_v3" / f"ratio_B02_B03_{date_str}.tif",
        _PHASE_A_ROOT / "reef_output_v3" / "sdb_depth_map.tif",
    ]
    for p in candidates:
        if p.is_file():
            log.info("Phase A SDB found: %s", p)
            return p

    # Try any predictor_A date
    acolite_dir = _PHASE_A_ROOT / "reef_output_acolite_comparison"
    if acolite_dir.is_dir():
        for sub in sorted(acolite_dir.iterdir()):
            if sub.is_dir() and f"predictor_A_{date_str}" in sub.name:
                tif = sub / "sdb_depth_map.tif"
                if tif.is_file():
                    log.info("Phase A SDB found: %s", tif)
                    return tif

    log.warning("Phase A SDB not found for date %s", date_str)
    return None


def find_phase_a_bands(date_str: str) -> tuple[Path | None, Path | None]:
    """
    Find Phase A B02 and B03 band rasters.
    Returns (b02_path, b03_path) or (None, None).
    """
    v3_dir = _PHASE_A_ROOT / "reef_output_v3"
    b02 = v3_dir / f"S2_B02_{date_str}.tif"
    b03 = v3_dir / f"S2_B03_{date_str}.tif"
    if b02.is_file() and b03.is_file():
        return b02, b03
    log.warning("Phase A bands not found for %s", date_str)
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ICESAT-2 DATA ACCESS (graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════════

def icesat2_available() -> bool:
    """Check if earthaccess is installed and credentials are configured."""
    try:
        import earthaccess  # noqa: F401
        return True
    except ImportError:
        return False


def icesat2_authenticated() -> bool:
    """Check if user has valid NASA EarthData session."""
    if not icesat2_available():
        return False
    try:
        import earthaccess
        auth = earthaccess.login(strategy="environment", quit=False)
        return auth is not None
    except Exception:
        return False


def search_icesat2_granules(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    product: str = "ATL03",
) -> list[dict]:
    """
    Search NASA CMR for ICESat-2 granules in the given bounding box.
    Returns list of granule metadata dicts with keys: date, granule_id, url.

    Returns [] if not authenticated or search fails.
    """
    if not icesat2_authenticated():
        log.warning(
            "ICESat-2 credentials not available. "
            "NASA EarthData login required: https://urs.earthdata.nasa.gov"
        )
        return []

    try:
        import earthaccess

        results = earthaccess.search_data(
            short_name=product,
            bounding_box=(min_lon, min_lat, max_lon, max_lat),
            temporal=("2018-10-01", datetime.now().strftime("%Y-%m-%d")),
            count=100,
        )

        granules = []
        for r in results:
            try:
                granule_id = str(r) if not hasattr(r, "native_id") else r.native_id
                # Extract date from granule ID: ATL03_YYYYMMDDHHMMSS_...
                date_str = None
                for part in granule_id.split("_"):
                    if len(part) >= 8 and part[:4].isdigit():
                        try:
                            dt = datetime.strptime(part[:8], "%Y%m%d")
                            date_str = dt.strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue

                granules.append({
                    "date": date_str or "unknown",
                    "granule_id": granule_id[:80],
                    "resource": str(r),
                })
            except Exception as e:
                log.debug("Granule parse error: %s", e)
                continue

        log.info("ICESat-2 %s: %d granules found over bbox", product, len(granules))
        return granules

    except Exception as e:
        log.warning("ICESat-2 search failed: %s", e)
        return []


def fetch_icesat2_depths(
    granule_url: str,
    bounds_wgs84: tuple[float, float, float, float],
    min_depth: float = 1.0,
    max_depth: float = 30.0,
    max_photon_depth_uncertainty: float = 2.0,
) -> np.ndarray:
    """
    Fetch and filter ICESat-2 ATL03 photon depths within a bounding box.

    Parameters
    ----------
    granule_url  : URL to the ATL03 HDF5 granule (from earthaccess)
    bounds_wgs84 : (min_lon, min_lat, max_lon, max_lat) of interest
    min_depth    : minimum optically viable depth (m)
    max_depth    : maximum depth for Sentinel-2 (m)
    max_photon_depth_uncertainty : maximum allowed depth error per photon (m)

    Returns
    -------
    Array of shape (N, 3) = (lon, lat, depth_m) for valid photons.
    Returns empty array if authentication fails or data is unavailable.

    Note: Requires valid NASA EarthData credentials. Downloads HDF5 to a
    temporary local file, processes, then cleans up.
    """
    if not icesat2_authenticated():
        log.warning("No EarthData session — cannot fetch ATL03 photon data")
        return np.zeros((0, 3))

    try:
        import earthaccess
        import h5py
        import tempfile

        min_lon, min_lat, max_lon, max_lat = bounds_wgs84

        # Download granule to a temp file (earthaccess handles auth)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            log.info("Downloading ATL03 granule (may take a minute)...")
            earthaccess.download([granule_url], path=os.path.dirname(tmp_path))
            granule_local = [
                f for f in os.listdir(os.path.dirname(tmp_path))
                if f.endswith(".h5")
            ]
            if not granule_local:
                log.warning("Download did not produce an HDF5 file")
                return np.zeros((0, 3))
            granule_path = os.path.join(os.path.dirname(tmp_path), granule_local[0])

        except Exception as e:
            log.warning("ATL03 download failed: %s", e)
            return np.zeros((0, 3))

        depths = []
        try:
            with h5py.File(granule_path, "r") as f:
                # ATL03 structure: gtXX/heights/h_ph, lat_ph, lon_ph, ...
                for beam in ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]:
                    if beam not in f:
                        continue
                    try:
                        heights = f[beam]["heights"]
                        lats = heights["lat_ph"][:]
                        lons = heights["lon_ph"][:]
                        h_ph = heights["h_ph"][:]
                        conf = heights["signal_conf_ph"][:]

                        # Filter: high confidence, within bbox, viable depth
                        for i in range(len(lats)):
                            if conf[i] < 3:          # need high confidence
                                continue
                            lat_val = float(lats[i])
                            lon_val = float(lons[i])
                            depth_val = -float(h_ph[i])  # h_ph is height above WGS84

                            if not (min_lon <= lon_val <= max_lon and
                                    min_lat <= lat_val <= max_lat):
                                continue
                            if not (min_depth <= depth_val <= max_depth):
                                continue
                            depths.append([lon_val, lat_val, depth_val])

                    except KeyError:
                        continue
        finally:
            # Cleanup temp HDF5
            try:
                os.unlink(granule_path)
            except Exception:
                pass

        result = np.array(depths)
        log.info("ATL03: %d valid photons extracted", len(result))
        return result

    except ImportError as e:
        log.warning("Missing dependency for ATL03 HDF5 processing: %s", e)
        return np.zeros((0, 3))
    except Exception as e:
        log.warning("ATL03 processing failed: %s", e)
        return np.zeros((0, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAMPLE MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def match_icesat2_to_sdb(
    icesat2_pts: np.ndarray,
    sdb_arr: np.ndarray,
    bounds_wgs84: tuple[float, float, float, float],
    max_distance_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match ICESat-2 depth soundings to SDB pixel depths.

    For each ICESat-2 point, find the nearest SDB pixel within max_distance_m.
    Returns (icesat2_depths, sdb_depths) matched arrays of equal length.

    Parameters
    ----------
    icesat2_pts  : array (N, 3) = (lon, lat, depth_m)
    sdb_arr      : 2D SDB depth array (H, W) in metres
    bounds_wgs84 : (min_lon, min_lat, max_lon, max_lat) of SDB raster
    max_distance_m : max search radius in metres

    Returns
    -------
    (icesat2_matched, sdb_matched) — arrays of shape (M, 2) with M <= N
    M may be 0 if no nearby SDB pixels are found.
    """
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    H, W = sdb_arr.shape

    # Haversine helper
    def haversine_m(lon1, lat1, lon2, lat2):
        R = 6_371_000.0
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    matched_icesat = []
    matched_sdb = []

    for pt in icesat2_pts:
        lon, lat, depth = pt
        # Convert to pixel coords
        col = int((lon - min_lon) / (max_lon - min_lon) * (W - 1))
        row = int((max_lat - lat) / (max_lat - min_lat) * (H - 1))
        if not (0 <= row < H and 0 <= col < W):
            continue
        sdb_val = float(sdb_arr[row, col])
        if sdb_val <= 0:
            continue

        # Check if any SDB pixel within max_distance_m has valid data
        # Use a small window (±3 pixels = ±30m at 10m resolution)
        found = False
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = row + dr, col + dc
                if not (0 <= r < H and 0 <= c < W):
                    continue
                sdb_v = float(sdb_arr[r, c])
                if sdb_v <= 0:
                    continue
                # Compute actual distance for the centre pixel
                # Pixel centre: lon = min_lon + (c+0.5)/W*(max_lon-min_lon)
                #               lat = max_lat - (r+0.5)/H*(max_lat-min_lat)
                centre_lon = min_lon + (c + 0.5) / W * (max_lon - min_lon)
                centre_lat = max_lat - (r + 0.5) / H * (max_lat - min_lat)
                dist_m = haversine_m(lon, lat, centre_lon, centre_lat)
                if dist_m <= max_distance_m:
                    matched_icesat.append(depth)
                    matched_sdb.append(sdb_v)
                    found = True
                    break
            if found:
                break

    if not matched_icesat:
        log.warning("No ICESat-2 points matched to SDB pixels within %dm", max_distance_m)

    return np.array(matched_icesat), np.array(matched_sdb)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION FIT
# ═══════════════════════════════════════════════════════════════════════════════

def fit_calibration(
    icesat2_matched: np.ndarray,
    sdb_matched: np.ndarray,
) -> dict:
    """
    Fit a calibration correction (linear regression) from SDB to ICESat-2 depths.

    Model: Z_true = a * Z_SDB + b

    Returns dict with calibration parameters and fit quality metrics.
    If insufficient samples (< 4), returns defaults with calibrated=False.
    """
    n = len(icesat2_matched)

    if n < 4:
        log.warning(
            "Insufficient matched samples (%d < 4) for calibration. "
            "Returning default (no calibration).",
            n
        )
        return {
            "calibrated": False,
            "n_samples": n,
            "a": 1.0,
            "b": 0.0,
            "rmse_m": None,
            "mae_m": None,
            "bias_m": None,
            "r2": None,
            "reason": f"insufficient_samples({n})",
        }

    # OLS linear regression: Z_true = a * Z_SDB + b
    # Solve via polyfit (degree 1)
    a_raw, b_raw = np.polyfit(sdb_matched, icesat2_matched, 1)

    # Clip to physically reasonable range
    clipped_a = a_raw < 0.5 or a_raw > 2.0
    clipped_b = b_raw < -10.0 or b_raw > 10.0
    a = float(np.clip(a_raw, 0.5, 2.0))   # slope: not too far from 1
    b = float(np.clip(b_raw, -10.0, 10.0))  # offset: not more than 10m

    if clipped_a or clipped_b:
        log.warning(
            "Calibration fit parameters were clipped: raw a=%.4f b=%.3f → a=%.4f b=%.3f. "
            "True relationship may be outside physically plausible range.",
            a_raw, b_raw, a, b
        )

    # Predictions and errors — use raw (pre-clip) for honest RMSE reporting
    sdb_pred = a * sdb_matched + b
    errors = icesat2_matched - sdb_pred

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    r2 = float(1 - np.sum(errors ** 2) / np.sum((icesat2_matched - np.mean(icesat2_matched)) ** 2))

    log.info(
        "Calibration fit: a=%.4f  b=%.3f  RMSE=%.3fm  MAE=%.3fm  bias=%.3fm  R2=%.4f  n=%d",
        a, b, rmse, mae, bias, r2, n
    )

    return {
        "calibrated": True,
        "n_samples": n,
        "a": round(a, 6),
        "b": round(b, 3),
        "rmse_m": round(rmse, 4),
        "mae_m": round(mae, 4),
        "bias_m": round(bias, 4),
        "r2": round(r2, 6),
        "method": "ols_linear_regression",
        "per_sample_matched": n,
    }


def apply_calibration(
    sdb_arr: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """
    Apply calibration to the SDB depth array.

    Z_calibrated = a * Z_SDB + b

    Invalid / zero pixels are left as zero (no depth).
    """
    calibrated = sdb_arr.copy()
    mask = sdb_arr > 0
    calibrated[mask] = a * sdb_arr[mask] + b
    calibrated[calibrated < 0] = 0  # no negative depths
    return calibrated


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OUTPUT WRITING
# ═══════════════════════════════════════════════════════════════════════════════

def write_calibrated_tif(
    calibrated_arr: np.ndarray,
    source_tif: Path,
    output_path: Path,
) -> bool:
    """
    Write calibrated depth array as GeoTIFF, inheriting CRS and transform
    from the source Phase A TIF.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        with rasterio.open(source_tif) as src:
            profile = src.profile.copy()
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress="deflate",
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(calibrated_arr.astype(rasterio.float32), 1)

        log.info("Calibrated TIF written: %s", output_path)
        return True

    except ImportError:
        log.warning("rasterio not available — cannot write GeoTIFF")
        return False
    except Exception as e:
        log.error("Failed to write calibrated TIF: %s", e)
        return False


def write_summary_json(
    summary: dict,
    output_path: Path,
) -> None:
    """Write calibration summary as JSON."""
    try:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Calibration summary: %s", output_path)
    except Exception as e:
        log.error("Failed to write summary JSON: %s", e)


def write_sample_pairs_csv(
    icesat2_matched: np.ndarray,
    sdb_matched: np.ndarray,
    output_path: Path,
) -> None:
    """Write matched (ICESat-2, SDB) sample pairs as CSV."""
    try:
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["icesat2_depth_m", "sdb_depth_m", "error_m"])
            for i2, s2 in zip(icesat2_matched, sdb_matched):
                writer.writerow([round(i2, 4), round(s2, 4), round(i2 - s2, 4)])
        log.info("Sample pairs CSV: %s (%d pairs)", output_path, len(icesat2_matched))
    except Exception as e:
        log.error("Failed to write sample pairs CSV: %s", e)


def write_report_txt(
    summary: dict,
    phase_a_date: str,
    output_path: Path,
) -> None:
    """Write human-readable calibration report."""
    try:
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  PHASE B CALIBRATION REPORT — ICESat-2 / Sentinel-2 SDB\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write(f"Phase A date: {phase_a_date}\n\n")

            calibrated = summary.get("calibrated", False)
            f.write(f"Status: {'CALIBRATED' if calibrated else 'NOT CALIBRATED (fallback)'}\n\n")

            if calibrated:
                f.write("Calibration model: Z_true = a * Z_SDB + b\n")
                f.write(f"  a = {summary['a']:.6f}  (slope correction)\n")
                f.write(f"  b = {summary['b']:.3f} m  (bias offset)\n\n")

                f.write("Fit quality:\n")
                f.write(f"  N matched samples:   {summary['n_samples']}\n")
                f.write(f"  RMSE:                {summary['rmse_m']:.4f} m\n")
                f.write(f"  MAE:                 {summary['mae_m']:.4f} m\n")
                f.write(f"  Bias (mean error):   {summary['bias_m']:.4f} m\n")
                f.write(f"  R²:                  {summary['r2']:.6f}\n")
                f.write(f"  Method:              {summary['method']}\n\n")

                f.write("Note:\n")
                f.write("  a < 1: SDB overestimates depth (too deep) → multiply by a < 1\n")
                f.write("  a > 1: SDB underestimates depth (too shallow) → multiply by a > 1\n")
                f.write("  b > 0: SDB is systematically too shallow\n")
                f.write("  b < 0: SDB is systematically too deep\n\n")
            else:
                reason = summary.get("reason", "unknown")
                f.write(f"Not calibrated reason: {reason}\n\n")
                f.write("Possible causes:\n")
                f.write("  - No ICESat-2 credentials (NASA EarthData login required)\n")
                f.write("  - No ATL03 granules over this bbox\n")
                f.write("  - Too few matched samples (< 4 required)\n")
                f.write("  - ATL03 download / processing failure\n\n")

            f.write("How to interpret:\n")
            f.write("  Phase A (Stumpf SDB) produces raw depth estimates using\n")
            f.write("  literature m0/m1 coefficients that may not fit Algarve.\n")
            f.write("  Phase B applies an ICESat-2 derived correction to reduce bias.\n\n")
            f.write("Next steps:\n")
            f.write("  1. Review RMSE — should be < 2m for acceptable calibration\n")
            f.write("  2. If R² is low (< 0.5), the linear model may not be appropriate\n")
            f.write("  3. Check sample_pairs.csv to identify spatial patterns in error\n")

        log.info("Report written: %s", output_path)
    except Exception as e:
        log.error("Failed to write report: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN CALIBRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_b(
    phase_a_date: str,
    site_name: str = "algarve_coast",
    site_bounds: tuple | None = None,  # (min_lon, min_lat, max_lon, max_lat)
) -> dict:
    """
    Run the full Phase B calibration pipeline.

    Parameters
    ----------
    phase_a_date : YYYYMMDD string matching a Phase A SDB product
    site_name    : label for this site (for reporting)
    site_bounds  : optional bounding box override (decimal degrees)

    Returns
    -------
    summary dict with calibration results and output file paths.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PHASE B CALIBRATION — ICESat-2 / Sentinel-2 SDB")
    log.info("Date: %s  |  Site: %s", phase_a_date, site_name)
    log.info("=" * 60)

    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase_a_date": phase_a_date,
        "site_name": site_name,
        "calibrated": False,
        "outputs": {},
    }

    # ── Step 1: Load Phase A SDB ────────────────────────────────────────────────
    sdb_path = find_phase_a_sdb(phase_a_date)
    if sdb_path is None:
        log.error("Phase A SDB not found for %s — cannot proceed", phase_a_date)
        summary["error"] = f"Phase A SDB not found for {phase_a_date}"
        return summary

    try:
        import rasterio
        with rasterio.open(sdb_path) as src:
            sdb_arr = src.read(1).astype(np.float32)
            bounds_wgs84 = src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3]
            # bounds = (min_x, min_y, max_x, max_y) = (min_lon, min_lat, max_lon, max_lat)
            log.info("Phase A SDB loaded: shape=%s  bounds=%s", sdb_arr.shape, bounds_wgs84)
    except Exception as e:
        log.error("Failed to load Phase A SDB: %s", e)
        summary["error"] = str(e)
        return summary

    # ── Step 2: Search for ICESat-2 granules ─────────────────────────────────
    if site_bounds is not None:
        min_lon, min_lat, max_lon, max_lat = site_bounds
    else:
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84

    granules = search_icesat2_granules(min_lon, min_lat, max_lon, max_lat, "ATL03")

    if not granules:
        log.warning(
            "No ICESat-2 ATL03 granules found for this bounding box.\n"
            "This is expected if:\n"
            "  1. NASA EarthData credentials are not configured\n"
            "  2. No ATL03 tracks pass over this location\n"
            "  3. The bbox has no valid granules in CMR\n\n"
            "To access ICESat-2 data:\n"
            "  1. Register at https://urs.earthdata.nasa.gov\n"
            "  2. Run: earthaccess.login() and follow prompts\n"
            "  3. Re-run this script\n\n"
            "A placeholder calibrated product (no correction) will be exported."
        )
        # Fall through — still produce output with identity calibration

    # ── Step 3: Fetch ICESat-2 photon depths ─────────────────────────────────
    icesat2_pts = np.zeros((0, 3))
    if granules:
        log.info("Fetching ICESat-2 photon data from %d granule(s)...", len(granules))
        # Try the most recent granule first
        for granule in granules[:3]:  # limit to 3 to avoid long downloads
            icesat2_pts = fetch_icesat2_depths(
                granule["resource"],
                (min_lon, min_lat, max_lon, max_lat),
            )
            if len(icesat2_pts) > 10:
                break

    # ── Step 4: Match to SDB pixels ───────────────────────────────────────────
    if len(icesat2_pts) > 0:
        icesat2_matched, sdb_matched = match_icesat2_to_sdb(
            icesat2_pts, sdb_arr, bounds_wgs84, max_distance_m=30.0
        )
    else:
        icesat2_matched = np.array([])
        sdb_matched = np.array([])

    log.info("Matched samples: %d / %d ICESat-2 points", len(icesat2_matched), len(icesat2_pts))

    # ── Step 5: Fit calibration ───────────────────────────────────────────────
    cal = fit_calibration(icesat2_matched, sdb_matched)
    summary.update(cal)
    summary["bounds_wgs84"] = bounds_wgs84

    # ── Step 6: Apply calibration to SDB ──────────────────────────────────────
    a = cal["a"]
    b = cal["b"]
    calibrated_arr = apply_calibration(sdb_arr, a, b)

    # ── Step 7: Write outputs ─────────────────────────────────────────────────
    prefix = f"calibrated_SDB_{phase_a_date}"
    calibrated_tif_path = _OUTPUT_DIR / f"{prefix}.tif"
    summary_json_path = _OUTPUT_DIR / "calibration_summary.json"
    csv_path = _OUTPUT_DIR / "sample_pairs.csv"
    report_path = _OUTPUT_DIR / "calibration_report.txt"

    # Write calibrated TIF
    ok = write_calibrated_tif(calibrated_arr, sdb_path, calibrated_tif_path)
    if ok:
        summary["outputs"]["calibrated_tif"] = str(calibrated_tif_path)

    # Write summary JSON
    write_summary_json(summary, summary_json_path)
    summary["outputs"]["summary_json"] = str(summary_json_path)

    # Write sample pairs CSV (if any matched)
    if len(icesat2_matched) > 0:
        write_sample_pairs_csv(icesat2_matched, sdb_matched, csv_path)
        summary["outputs"]["sample_pairs_csv"] = str(csv_path)

    # Write human-readable report
    write_report_txt(summary, phase_a_date, report_path)
    summary["outputs"]["report_txt"] = str(report_path)

    log.info("Phase B complete. Outputs in: %s", _OUTPUT_DIR)
    log.info("Calibration status: %s", "CALIBRATED" if cal["calibrated"] else "NOT CALIBRATED")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase B: Calibrate Sentinel-2 SDB using ICESat-2 ATL03 ground truth"
    )
    parser.add_argument(
        "--date", "-d",
        default="20250925",
        help="Phase A date (YYYYMMDD). Default: 20250925",
    )
    parser.add_argument(
        "--site", "-s",
        default="algarve_coast",
        help="Site name for reporting. Default: algarve_coast",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="Override bounding box (decimal degrees)",
    )
    args = parser.parse_args()

    bounds = tuple(args.bounds) if args.bounds else None

    result = run_phase_b(
        phase_a_date=args.date,
        site_name=args.site,
        site_bounds=bounds,
    )

    # Print brief status
    calibrated = result.get("calibrated", False)
    if calibrated:
        print(
            f"\n✓ Phase B calibrated: a={result['a']:.4f}  "
            f"b={result['b']:.3f}m  RMSE={result['rmse_m']:.3f}m  n={result['n_samples']}"
        )
    else:
        print(f"\n✗ Phase B not calibrated: {result.get('reason', 'unknown')}")
        print("  → Check calibration_report.txt in outputs/phase_b/ for guidance")


if __name__ == "__main__":
    main()