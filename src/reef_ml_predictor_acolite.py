#!/usr/bin/env python3
"""
reef_ml_predictor_acolite.py — v3.0
Adds:
  (A) Full Gordon/QAA-style Kd inversion using B02/B03/B04 bands
  (B) Stumpf log-ratio Satellite Derived Bathymetry (SDB) depth map
  (C) Gordon band-ratio Kd estimator integrated into run_predictor
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    beer_lambert_transmittance,
    get_kd490,
    optical_path,
    read_band,
    snell_sza,
    write_band,
)

try:
    from src.bathy_calibrator import run_bathy_integration

    _BATHY_AVAILABLE = True
except ImportError:
    _BATHY_AVAILABLE = False

try:
    from src.ih_bathy_features import get_bathy_features_for_summary

    _IH_BATHY_FEATURES_AVAILABLE = True
except ImportError:
    _IH_BATHY_FEATURES_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Physical constants ────────────────────────────────────────────────────────
from src.constants import (
    DEFAULT_DEPTH_TARGET,
    FFT_CLEAN_THRESHOLD,
    GLINT_PENALTY,
    GLINT_PENALTY_DEFAULT,
    KD490_MAP_SATURATION_CEILING,
    KD490_TABLE,
    REFLECTANCE_DN_SCALE,
    REFLECTANCE_DN_THRESHOLD,
    ROCK_R,
    SAND_R,
    SDB_OPTICAL_LIMIT_M,
    STUMPF_LOG_EPSILON,
    STUMPF_M0_DEFAULT,
    STUMPF_M1_DEFAULT,
    STUMPF_N,
)

# Aliases for backward compat
DEPTH_TARGET = DEFAULT_DEPTH_TARGET
_log = logging.getLogger(__name__)
try:
    from src.cmems_kd490 import KD490_TABLE_LIVE as DEFAULT_KD_TABLE
except Exception as _cmems_err:
    _log.warning("cmems_kd490 unavailable (%s) — using static Kd490 table.", _cmems_err)
    DEFAULT_KD_TABLE = KD490_TABLE  # static fallback

# Sentinel-2 band pure-water attenuation (aw, m⁻¹) — Pope & Fry 1997
AW = {"B02": 0.0145, "B03": 0.0612, "B04": 0.4300}


# ── A: Gordon/QAA Kd inversion ───────────────────────────────────────────────
def gordon_kd_inversion(
    b02: np.ndarray, b03: np.ndarray, b04: np.ndarray | None = None, kd_prior: float = 0.045
) -> tuple[float, float, float | None]:
    """
    Quasi-Analytical Algorithm (QAA, Lee et al. 2002) simplified for B02/B03/B04.
    Returns: (kd_b02_est, kd_b03_est, kd_b04_est | None)

    kd_b04_est is None when no genuine B04 is supplied. (Until 2026-10-09 it was
    filled with the unvalidated scaling ``Kd_B03 * aw_B04/aw_B03 * 0.8``; that
    fallback was removed — Kd(665) is a derived optical property, not a spectral
    band, but it still must not be invented. It feeds only the
    ``kd_b04_estimated`` summary field and a log line.)

    Step 1 – rrs (sub-surface remote sensing reflectance):
        rrs = Rrs / (0.52 + 1.7 * Rrs)   [Lee 2002, eq. 4]

    Step 2 – u (ratio bb / (a + bb)):
        u = (-g0 + sqrt(g0² + 4*g1*rrs)) / (2*g1)
        g0=0.0895, g1=0.1247  [Gordon 1988]

    Step 3 – total absorption at 555nm (green):
        a555 = aw_555 + 10^(-1.146 - 1.366*chi - 0.469*chi²)   where chi = log10(rrs_B02/rrs_B03)

    Step 4 – Kd at each band:
        Kd = (1 + 0.005 * sza) * a + 4.18*(1 - 0.52*exp(-10.8*a)) * bb
        Simplified: Kd ≈ (a + bb) / cos(theta_sun)   [Morel 2007]
    """
    G0, G1 = 0.0895, 0.1247
    AW_B02, AW_B03 = AW["B02"], AW["B03"]
    AW_B04 = AW["B04"]

    def to_rrs(rrs_surf):
        return rrs_surf / (0.52 + 1.7 * rrs_surf + 1e-9)

    mask = (b02 > 0) & (b03 > 0)
    if mask.sum() < 10:
        # Too few valid pixels: B02/B03 fall back to the seasonal prior (pre-existing behaviour);
        # Kd at 665 nm is NOT invented from the prior.
        return kd_prior, kd_prior * (490 / 560) ** 0.5, None

    rrs02 = to_rrs(b02[mask])
    rrs03 = to_rrs(b03[mask])

    # Step 2: u at green (B03)
    u_green = (-G0 + np.sqrt(G0**2 + 4 * G1 * rrs03 + 1e-12)) / (2 * G1)
    u_blue = (-G0 + np.sqrt(G0**2 + 4 * G1 * rrs02 + 1e-12)) / (2 * G1)

    # Step 3: total absorption at green (560nm)
    chi = np.log10(np.clip(rrs02 / (rrs03 + 1e-9), 1e-3, 100))
    a555 = AW_B03 + 10 ** (-1.146 - 1.366 * chi - 0.469 * chi**2)

    # Step 4: backscattering at green
    bb555 = u_green * a555 / (1 - u_green + 1e-9)
    bb555 = np.clip(bb555, 0, 1)

    # Scale bb to blue (spectral power law: bb(λ) ~ bb(555) * (555/λ)^Y, Y≈1)
    bb_blue = bb555 * (560 / 490)

    # Total absorption at blue
    a_blue = AW_B02 + bb_blue * (1 - u_blue + 1e-9) / (u_blue + 1e-9)
    a_blue = np.clip(a_blue, AW_B02, 1.0)

    # Kd ≈ a + bb  (nadir viewing, simplified Morel)
    kd_b02 = float(np.nanmedian(a_blue + bb_blue))
    kd_b03 = float(np.nanmedian(a555 + bb555))

    # B04 (665nm) — only from a genuine B04 band; otherwise unavailable (no scaling from B03)
    kd_b04 = None if b04 is None else _kd_from_band(b04, AW_B04)

    # Sanity clamp: Algarve coastal range
    kd_b02 = float(np.clip(kd_b02, 0.010, 0.500))
    kd_b03 = float(np.clip(kd_b03, 0.020, 0.500))
    kd_b04 = None if kd_b04 is None else float(np.clip(kd_b04, 0.050, 2.000))

    return kd_b02, kd_b03, kd_b04


def _kd_from_band(band: np.ndarray, aw: float) -> float:
    """Fallback: estimate Kd from single band reflectance level."""
    mask = band > 0
    if mask.sum() < 5:
        return aw * 3
    rrs = band[mask] / (0.52 + 1.7 * band[mask])
    G0, G1 = 0.0895, 0.1247
    u = (-G0 + np.sqrt(G0**2 + 4 * G1 * rrs + 1e-12)) / (2 * G1)
    a = aw + u * aw / (1 - u + 1e-9) * 0.5
    bb = np.clip(u * a / (1 - u + 1e-9), 0, 1)
    return float(np.nanmedian(a + bb))


# ── A½: Water-column reflectance inversion ───────────────────────────────────
def invert_water_column(
    rrs_surface: np.ndarray,
    kd: float,
    depth_m: float,
) -> np.ndarray:
    """
    Recover bottom reflectance from surface Rrs by inverting two-way Beer-Lambert:
        Rrs_bottom = Rrs_surface / exp(-2 * Kd * z)
    Negative values (noisy pixels) are clipped to 0.
    """
    return np.clip(rrs_surface / math.exp(-2.0 * kd * depth_m), 0, None).astype(np.float32)


# ── B: Stumpf Log-Ratio SDB depth map ────────────────────────────────────────
def stumpf_sdb(
    b02: np.ndarray, b03: np.ndarray, m0: float = STUMPF_M0_DEFAULT, m1: float = STUMPF_M1_DEFAULT, n: float = STUMPF_N
) -> np.ndarray:
    """
    Stumpf et al. (2003) log-ratio Satellite Derived Bathymetry:
        depth = m1 * ln(n * B02) / ln(n * B03) + m0
    Default m0/m1 calibrated for Algarve oligotrophic waters (Kd≈0.045).
    Returns depth map in metres (positive = deeper). Values >40m set to NaN
    (optical limit exceeded, unreliable extrapolation).
    """
    eps = STUMPF_LOG_EPSILON
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((b02 > eps) & (b03 > eps), np.log(n * b02 + eps) / (np.log(n * b03 + eps) + eps), np.nan)
    depth = m1 * ratio + m0
    # Clip negative values to 0, but set beyond optical limit to NaN
    depth = np.where(depth < 0, np.nan, depth)
    depth = np.where(depth > SDB_OPTICAL_LIMIT_M, np.nan, depth)
    return depth.astype(np.float32)


# ── C: Integrated Kd estimator (simple band-ratio, fallback when QAA fails) ──
def estimate_kd_bandratio(b02: np.ndarray, b03: np.ndarray, kd_prior: float) -> tuple[float, bool]:
    """Gordon approximation: Kd scales with B02/B03 ratio residual."""
    mask = (b02 > 0) & (b03 > 0)
    if mask.sum() < 10:
        return kd_prior, False
    ratio = np.mean(b02[mask]) / (np.mean(b03[mask]) + 1e-9)
    kd_est = kd_prior * (1 + (ratio - 1.0) * 0.15)
    high_uncert = abs(kd_est - kd_prior) / kd_prior > 0.30
    return float(np.clip(kd_est, 0.010, 0.500)), high_uncert


# ── SNR map ───────────────────────────────────────────────────────────────────
def make_snr_map(arr: np.ndarray, window: int = 7) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter

        m = uniform_filter(arr.astype(np.float64), size=window)
        sq = uniform_filter(arr.astype(np.float64) ** 2, size=window)
        std = np.sqrt(np.clip(sq - m**2, 0, None))
        return np.where(std > 0, m / std, 0).astype(np.float32)
    except ImportError:
        sig = np.mean(arr[arr > 0]) if np.any(arr > 0) else 0
        std = np.std(arr[arr > 0]) + 1e-9
        return np.full_like(arr, sig / std, dtype=np.float32)


# ── Main predictor ────────────────────────────────────────────────────────────
def run_predictor(
    boa_b02_path,
    metadata,
    output_dir,
    kd_prior: dict | None = None,
    cloud_threshold: float = 0.2,
    snr_threshold: float = 3.0,
    date: str | None = None,
    b03_path: str | None = None,
    b04_path: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    depth_target: float = DEFAULT_DEPTH_TARGET,
    with_bathy_features: bool = False,
    stumpf_m0_override: float | None = None,
    stumpf_m1_override: float | None = None,
) -> dict:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date = date or metadata.get("date", "unknown")
    month = int(date.split("-")[1]) if "-" in date else 9
    kd_tbl = kd_prior or DEFAULT_KD_TABLE
    kd_seas = get_kd490(month, kd_tbl)
    glint_pen = GLINT_PENALTY.get(month, GLINT_PENALTY_DEFAULT)

    b02_arr, profile = read_band(boa_b02_path)
    b02_arr = np.nan_to_num(b02_arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalise to BOA reflectance [0..1].
    # nanmax guards against NaN-only tiles silently skipping the conversion.
    # Values in (1.0, REFLECTANCE_DN_THRESHOLD] are ambiguous — warn but accept.
    _b02_max = float(np.nanmax(b02_arr))
    if _b02_max > REFLECTANCE_DN_THRESHOLD:
        logging.info("B02 looks like raw DN (max=%.1f) — scaling by 1/10000", _b02_max)
        b02_arr = b02_arr / REFLECTANCE_DN_SCALE
    elif _b02_max > 1.0:
        logging.warning(
            "B02 max=%.4f is >1.0 but <=%.1f — ambiguous DN/reflectance; " "assuming reflectance, no scaling applied",
            _b02_max,
            REFLECTANCE_DN_THRESHOLD,
        )

    b03_arr = b04_arr = None
    if b03_path:
        b03_arr, _ = read_band(b03_path)
        if np.nanmax(b03_arr) > REFLECTANCE_DN_THRESHOLD:
            b03_arr /= REFLECTANCE_DN_SCALE
    if b04_path:
        b04_arr, _ = read_band(b04_path)
        if np.nanmax(b04_arr) > REFLECTANCE_DN_THRESHOLD:
            b04_arr /= REFLECTANCE_DN_SCALE

    # ── Kd estimation: QAA if B03 available, else band-ratio, else prior ──────
    kd_method = "seasonal_prior"
    kd_high_uncert = False
    kd_b02, kd_b03 = kd_seas, kd_seas
    kd_b04 = None  # Kd(665): only from genuine B04 via QAA; never copied from the Kd490 prior
    kd_map = None  # per-pixel Kd490 map; computed when B04 is available

    if b03_arr is not None:
        try:
            kd_b02, kd_b03, kd_b04 = gordon_kd_inversion(b02_arr, b03_arr, b04_arr, kd_seas)
            # If QAA saturated to boundary, fall back to band-ratio
            if kd_b02 >= 0.500:
                raise ValueError("QAA Kd saturated — using band-ratio fallback")
            kd_high_uncert = abs(kd_b02 - kd_seas) / kd_seas > 0.30
            kd_method = "gordon_qaa"
            logging.info("Gordon/QAA Kd: B02=%.4f B03=%.4f B04=%s (prior=%.4f)", kd_b02, kd_b03,
                         "unavailable" if kd_b04 is None else f"{kd_b04:.4f}", kd_seas)
        except Exception as e:
            logging.warning("Gordon inversion issue (%s) — falling back to band-ratio", e)
            kd_b02, kd_high_uncert = estimate_kd_bandratio(b02_arr, b03_arr, kd_seas)
            kd_b03 = kd_b02 * (490 / 560) ** 0.5
            kd_method = "band_ratio_fallback"

        # Per-pixel Kd490 spatial map (Lee et al. 2013) — captures turbidity gradient
        if b04_arr is not None:
            try:
                from src.utils import get_kd490_map

                kd_map = get_kd490_map(b02_arr, b03_arr, b04_arr)
                write_band(str(out / "kd490_map.tif"), kd_map, profile)
                logging.info(
                    "Per-pixel Kd490 map: mean=%.4f min=%.4f max=%.4f → kd490_map.tif",
                    float(np.nanmean(kd_map)),
                    float(np.nanmin(kd_map)),
                    float(np.nanmax(kd_map)),
                )
            except Exception as _kd_err:
                logging.debug("Kd490 map skipped: %s", _kd_err)
    else:
        logging.info("No B03 — using seasonal Kd prior %.4f", kd_seas)

    if kd_high_uncert:
        logging.warning("Kd diverges >30%% from prior — high uncertainty flag")

    # ── Physics: Snell + Beer-Lambert ─────────────────────────────────────────
    sza_deg = metadata.get("solar_zenith_deg", 40.5)
    sza_water_deg, theta_water = snell_sza(sza_deg)
    path_m = optical_path(depth_target, theta_water)
    # Use per-pixel Kd490 map when available and physically valid; scene-mean
    # scalar otherwise. Over bright sand / sunglint the Lee-2013 band ratio
    # over-estimates and the whole map pins at the 2.0 m^-1 clip — that field is
    # meaningless, so we fall back to the scalar kd_b02 for transmittance.
    if kd_map is not None:
        _kd_mean = float(np.nanmean(kd_map))
        _frac_pinned = float(np.nanmean(kd_map >= 1.99))
        if _kd_mean > KD490_MAP_SATURATION_CEILING or _frac_pinned > 0.5:
            logging.warning(
                "Kd490 map saturated (mean=%.2f, pinned=%.0f%%) — " "falling back to scalar kd_b02=%.3f for transmittance",
                _kd_mean,
                100 * _frac_pinned,
                kd_b02,
            )
            trans_map = None
            trans = beer_lambert_transmittance(kd_b02, path_m)
        else:
            trans_map = np.exp(-2.0 * kd_map * path_m).astype(np.float32)
            trans = float(np.nanmean(trans_map))  # scalar summary for logging/CSV
            write_band(str(out / "transmittance_map.tif"), trans_map, profile)
    else:
        trans_map = None
        trans = beer_lambert_transmittance(kd_b02, path_m)

    # ── Bottom reflectance estimate ───────────────────────────────────────────
    # Rrs_deep ≈ 0 for clear oligotrophic water (Algarve); depth_target is z.
    bottom_est = invert_water_column(b02_arr, kd_b02, depth_target)

    # ── SNR map ───────────────────────────────────────────────────────────────
    snr_map = make_snr_map(bottom_est)
    snr_mean = float(np.nanmean(snr_map))
    snr_median = float(np.nanmedian(snr_map))

    # ── SDB depth map (Stumpf) — with IH chart calibration ───────────────────
    sdb_map = None
    bathy_result = {}
    stumpf_m0 = -16.0
    stumpf_m1 = 20.0
    tf = None
    bounds_wgs = None

    if b03_arr is not None:
        # Try to calibrate Stumpf coefficients from IH isobaths
        calibration_status = "skipped_no_location"  # default if lat/lon missing
        if _BATHY_AVAILABLE and lat is not None and lon is not None:
            calibration_status = "skipped_no_transform"  # will be overwritten if transform exists
            try:
                tf = profile.get("transform")
                if tf is not None:
                    h, w = b02_arr.shape
                    # Native raster bounds (may be projected, e.g. UTM metres)
                    _native_left = tf.c
                    _native_top = tf.f
                    _native_right = tf.c + tf.a * w
                    _native_bottom = tf.f + tf.e * h
                    _raster_crs = profile.get("crs")
                    # Reproject to WGS84 if the raster is in a projected CRS
                    try:
                        from rasterio.warp import transform_bounds as _tb

                        _wgs = _tb(_raster_crs, "EPSG:4326", _native_left, _native_bottom, _native_right, _native_top)
                        # _wgs = (min_lon, min_lat, max_lon, max_lat)
                        bounds_wgs = (_wgs[1], _wgs[0], _wgs[3], _wgs[2])  # → (min_lat, min_lon, max_lat, max_lon)
                    except Exception as _reproj_err:
                        logging.warning("bounds CRS reproject failed (%s); raster_crs=%s", _reproj_err, _raster_crs)
                        # Only trust native bounds if the raster is already geographic.
                        # Otherwise they are projected metres (e.g. UTM) and must NOT
                        # flow into CMEMS fetch or chart validation — leave them None.
                        try:
                            _is_geographic = _raster_crs is not None and _raster_crs.is_geographic
                        except Exception:
                            _is_geographic = False
                        if _is_geographic:
                            bounds_wgs = (_native_bottom, _native_left, _native_top, _native_right)
                        else:
                            bounds_wgs = None
                    bathy_result = run_bathy_integration(
                        lat=lat,
                        lon=lon,
                        b02_arr=b02_arr,
                        b03_arr=b03_arr,
                        b04_arr=b04_arr,
                        bounds_wgs84=bounds_wgs,
                    )
                    stumpf_m0 = bathy_result.get("recommended_m0", -16.0)
                    stumpf_m1 = bathy_result.get("recommended_m1", 20.0)
                    zone_info = bathy_result.get("zone", {})
                    calibrated = bathy_result.get("calibration", {}).get("calibrated", False)
                    calibration_status = "success" if calibrated else "failed_insufficient_data"
                    logging.info(
                        "IH bathy zone=%s | optically_viable=%s | " "Stumpf m0=%.2f m1=%.2f (calibrated=%s, status=%s)",
                        zone_info.get("zone"),
                        zone_info.get("optically_viable"),
                        stumpf_m0,
                        stumpf_m1,
                        calibrated,
                        calibration_status,
                    )
                else:
                    logging.warning("IH calibration skipped: no raster transform in profile")
            except Exception as bathy_err:
                calibration_status = f"failed_error: {type(bathy_err).__name__}"
                logging.warning("IH calibration failed with error: %s — using Stumpf defaults", bathy_err)
        else:
            if not _BATHY_AVAILABLE:
                calibration_status = "skipped_module_unavailable"
                logging.info("IH calibration skipped: bathy_calibrator module not available")
            elif lat is None or lon is None:
                calibration_status = "skipped_no_location"
                logging.info("IH calibration skipped: lat/lon not provided")
        bathy_result["calibration_status"] = calibration_status

        # ── CMEMS SDB prior (shadow layer — never raises) ─────────────────────
        # Fetch CMEMS coastal SDB for this scene's bbox and save alongside other
        # outputs.  Also used as fallback calibration when IH isobath data is
        # insufficient.
        cmems_10m_path = None
        try:
            if bounds_wgs is not None:
                from src.cmems_sdb import fetch_cmems_sdb, reproject_cmems_to_s2

                # bounds_wgs is WGS84 (min_lat, min_lon, max_lat, max_lon) or None
                min_lat_c, min_lon_c, max_lat_c, max_lon_c = bounds_wgs
                cmems_raw = out / "cmems_sdb_100m.tif"
                fetched = fetch_cmems_sdb(
                    bbox=(min_lon_c, min_lat_c, max_lon_c, max_lat_c),
                    output_path=cmems_raw,
                    method="composite",
                    min_qi=3,
                )
                if fetched is not None:
                    cmems_10m_path = out / "cmems_sdb_10m.tif"
                    reproject_cmems_to_s2(cmems_raw, boa_b02_path, cmems_10m_path)
                    logging.info("CMEMS SDB tile saved → %s", cmems_10m_path)
                    bathy_result["cmems_sdb_path"] = str(cmems_10m_path)
        except Exception as _cmems_err:
            logging.debug("CMEMS SDB shadow layer skipped: %s", _cmems_err)

        # ── ICESat-2 ATL03 calibration (fallback — only when IH failed) ─────
        # Prefer merged cache (1–34 m) over deep-only cache (15–30 m); the
        # deep-only cache biases m0/m1 toward deep targets and degrades SDB
        # accuracy for shallow reefs (<15 m).  Only activates when IH
        # calibration was unavailable or failed.
        _SURVEY_DIR = Path(__file__).resolve().parents[1] / "outputs" / "icesat2_deep_survey"
        _ATL03_CACHE = (
            _SURVEY_DIR / "atl03_all_photons.json"  # merged 1–34 m (preferred)
            if (_SURVEY_DIR / "atl03_all_photons.json").exists()
            else _SURVEY_DIR / "atl03_seafloor_photons.json"  # deep-only fallback
        )
        _ih_succeeded = calibration_status == "success"
        if _ATL03_CACHE.exists() and not _ih_succeeded:
            try:
                from src.icesat2_calibrator import calibrate_from_icesat2

                _b03_for_calib = str(out / "BOA_B03.tif") if (out / "BOA_B03.tif").exists() else None
                if _b03_for_calib:
                    _icesat2_result = calibrate_from_icesat2(
                        b02_path=boa_b02_path,
                        b03_path=_b03_for_calib,
                        photon_cache=_ATL03_CACHE,
                    )
                    if _icesat2_result["status"] == "success":
                        stumpf_m0 = _icesat2_result["m0"]
                        stumpf_m1 = _icesat2_result["m1"]
                        bathy_result["icesat2_calibration"] = _icesat2_result
                        bathy_result["calibration_status"] = "icesat2_atl03"
                        logging.info(
                            "ICESat-2 calibration adopted (IH unavailable): " "m0=%.3f m1=%.3f RMSE=%.3f m (%d photons)",
                            stumpf_m0,
                            stumpf_m1,
                            _icesat2_result["rmse"],
                            _icesat2_result["calibration_samples"],
                        )
                    else:
                        logging.debug("ICESat-2 calibration: %s", _icesat2_result["message"])
            except Exception as _ic_err:
                logging.debug("ICESat-2 calibration skipped: %s", _ic_err)

        # CMEMS fallback calibration: re-calibrate Stumpf when IH data was
        # insufficient and we have a CMEMS prior at 10 m resolution.
        if calibration_status not in ("success",) and cmems_10m_path is not None:
            try:

                import rasterio as _rio

                from src.stumpf_emodnet_calibration import calibrate_stumpf_vs_emodnet

                # Need a temporary EMODnet-style prior — use the CMEMS tile as sole prior
                with _rio.open(cmems_10m_path) as _src:
                    _cmems_arr = _src.read(1)

                # Only proceed if we have enough deep-zone pixels (5–34 m)
                _calib_mask = np.isfinite(_cmems_arr) & (_cmems_arr <= -5.0) & (_cmems_arr >= -34.0)
                if _calib_mask.sum() >= 100:
                    _sdb_calib_path = out / "sdb_depth_map_cmems_calib.tif"
                    _calib_result = calibrate_stumpf_vs_emodnet(
                        s2_blue_path=boa_b02_path,
                        s2_green_path=str(out / "BOA_B03.tif") if (out / "BOA_B03.tif").exists() else boa_b02_path,
                        emodnet_10m_path=str(cmems_10m_path),
                        output_path=str(_sdb_calib_path),
                        depth_min_m=5.0,
                        depth_max_m=34.0,
                    )
                    stumpf_m0 = _calib_result["m0"]
                    stumpf_m1 = _calib_result["m1"]
                    bathy_result["cmems_calibration"] = _calib_result
                    bathy_result["calibration_status"] = "cmems_fallback"
                    logging.info(
                        "CMEMS fallback calibration: m0=%.3f m1=%.3f RMSE=%.3f m (%d samples)",
                        stumpf_m0,
                        stumpf_m1,
                        _calib_result["rmse"],
                        _calib_result["calibration_samples"],
                    )
            except Exception as _fb_err:
                logging.debug("CMEMS fallback calibration skipped: %s", _fb_err)

        # ── Ensemble calibration: blend all available sources ─────────────────
        # Produces a weighted m0/m1 rather than winner-takes-all fallback chain.
        # Sources included: IH (if calibrated), ICESat-2 (if succeeded).
        # CMEMS-derived coefficients are already in stumpf_m0/m1 if fallback ran.
        try:
            from src.ensemble_calibrator import ensemble_calibrate

            _ih_cal = bathy_result.get("calibration", {}) if bathy_result else {}
            _ic_cal = bathy_result.get("icesat2_calibration") if bathy_result else None
            _ens = ensemble_calibrate(
                ih_result=_ih_cal if _ih_cal.get("calibrated") else None,
                icesat2_result=_ic_cal,
            )
            if _ens["status"] == "ensemble":
                # Only override if we got a genuine blend (not single-source = same as before)
                stumpf_m0 = _ens["m0"]
                stumpf_m1 = _ens["m1"]
                bathy_result["ensemble_calibration"] = _ens
                bathy_result["calibration_status"] = "ensemble"
                logging.info(
                    "Ensemble calibration: m0=%.3f m1=%.3f sources=%s weights=%s",
                    stumpf_m0,
                    stumpf_m1,
                    _ens["sources_used"],
                    _ens["weights"],
                )
        except Exception as _ens_err:
            logging.debug("Ensemble calibration skipped: %s", _ens_err)

        # Multi-scene fused coefficients (from stumpf_multiscene.fuse_scenes) take
        # final precedence: they are a robust median across N cloud-free scenes and
        # are insensitive to the per-scene artefacts the single-scene chain above can
        # be biased by. Still ran the chain so zone_info/CMEMS/validation are populated.
        if stumpf_m0_override is not None and stumpf_m1_override is not None:
            stumpf_m0 = float(stumpf_m0_override)
            stumpf_m1 = float(stumpf_m1_override)
            if bathy_result is not None:
                bathy_result["calibration_status"] = "multiscene_fused"
            logging.info("Multi-scene fused coefficients applied: m0=%.3f m1=%.3f", stumpf_m0, stumpf_m1)

        # Compute SDB with (possibly calibrated) coefficients
        sdb_map = stumpf_sdb(b02_arr, b03_arr, m0=stumpf_m0, m1=stumpf_m1)
        sdb_path = out / "sdb_depth_map.tif"
        write_band(str(sdb_path), sdb_map, profile)
        _sdb_pos = sdb_map[sdb_map > 0]
        sdb_mean = float(np.nanmean(_sdb_pos)) if _sdb_pos.size > 0 else 0.0
        logging.info("SDB depth map: mean=%.1fm, written to %s", sdb_mean, sdb_path)

        # Validate SDB vs IH chart (if calibration ran)
        if bathy_result and _BATHY_AVAILABLE and lat is not None and tf is not None and bounds_wgs is not None:
            try:
                from src.bathy_calibrator import fetch_isobaths_for_bbox, validate_sdb_vs_chart

                deg_buf = 3000 / 111_000.0
                feats = fetch_isobaths_for_bbox(lon - deg_buf, lat - deg_buf, lon + deg_buf, lat + deg_buf)
                if tf is not None:
                    val = validate_sdb_vs_chart(sdb_map, feats, bounds_wgs)
                    bathy_result["validation"] = val
                    ov = val.get("overall", {})
                    if ov:
                        logging.info(
                            "SDB validation vs IH chart: bias=%.2fm RMSE=%.2fm n=%d",
                            ov.get("overall_bias_m", 0),
                            ov.get("overall_rmse_m", 0),
                            ov.get("n_total", 0),
                        )
            except Exception as val_err:
                logging.warning("SDB validation failed: %s", val_err)
    else:
        sdb_path, sdb_mean = None, None

    # ── Masks & scores ────────────────────────────────────────────────────────
    cloud_pct = metadata.get("cloud_cover_pct", 2.0)
    usable_frac = max(0.0, 1.0 - cloud_pct / 100.0)
    useful_mask = (snr_map >= snr_threshold) & (bottom_est > 0)
    pct_useful = 100.0 * float(useful_mask.sum()) / max(1, (bottom_est > 0).sum())

    conf_map = np.where(snr_map >= snr_threshold * 2, 2, np.where(useful_mask, 1, 0)).astype(np.uint8)
    pct_high_conf = 100.0 * float((conf_map == 2).sum()) / max(1, (bottom_est > 0).sum())

    # Use spatially-varying transmittance where available for contrast calculation
    _trans_for_contrast = float(np.nanmean(trans_map)) if trans_map is not None else trans
    sand_btm = SAND_R * _trans_for_contrast
    rock_btm = ROCK_R * _trans_for_contrast
    # Normalise by surface reflectance (SAND_R), not sand_btm, so that trans does not
    # cancel out and contrast properly decreases with depth.
    contrast = (sand_btm - rock_btm) / SAND_R * glint_pen if SAND_R > 0 else 0.0

    # ── Bathymetry-derived features (IH/DGRM) ─────────────────────────────────
    # Computed BEFORE the ranker so the real-data model schema (which includes
    # bathy_slope_proxy, contour_density_proxy, n_isobaths_aoi, zone one-hots)
    # can be populated.  Always attempted when the module + location are
    # available — the ML ranker needs these features to avoid heuristic fallback.
    bathy_feats: dict = {}
    if _IH_BATHY_FEATURES_AVAILABLE and lat is not None and lon is not None:
        try:
            bathy_feats = get_bathy_features_for_summary(lon=lon, lat=lat)
            logging.info(
                "IH bathy features | zone=%s | nearest=%sm (%.0fm) | slope_proxy=%.2f",
                bathy_feats.get("bathymetry_zone_class"),
                bathy_feats.get("nearest_isobath_distance_m"),
                bathy_feats.get("nearest_isobath_depth_m", 0),
                bathy_feats.get("bathymetry_slope_proxy", 0),
            )
        except Exception as bf_err:
            logging.warning("Bathymetry feature generation failed: %s", bf_err)

    # ML Ranker inference instead of manual heuristic
    from src.ranking_model import predict_score

    ranker_features = {
        "kd_b02": kd_b02,
        "water_transmittance_twoway": _trans_for_contrast,
        "contrast_benthic_mean": contrast,  # canonical: ratio [0, 1] (NOT percentage)
        "SNR_mean_16m": snr_mean,
        "cloud_cover": cloud_pct,
        "cleanliness": FFT_CLEAN_THRESHOLD,  # Proxy when FFT is not run — sits at threshold boundary, no penalty
    }
    prediction = predict_score(ranker_features, bathy_features=bathy_feats or None)
    vis_score = prediction["score"]
    ranker_mode = prediction["mode"]

    # ── Benthic substrate classification ─────────────────────────────────────
    substrate_path = None
    substrate_stats: dict = {}
    if b03_arr is not None and b04_arr is not None:
        try:
            from src.substrate_classifier import (
                CLASS_NAMES,
                classify_substrate,
                write_substrate_tiff,
            )

            _sdb_for_sub = sdb_map if sdb_map is not None else None
            substrate = classify_substrate(
                b02=b02_arr,
                b03=b03_arr,
                b04=b04_arr,
                sdb_depth=_sdb_for_sub,
            )
            substrate_path = out / "substrate.tif"
            write_substrate_tiff(substrate, profile, substrate_path)
            _n_total = substrate.size
            substrate_stats = {
                name: round(100.0 * float((substrate == cls).sum()) / _n_total, 2)
                for cls, name in CLASS_NAMES.items()
                if cls != -1
            }
            logging.info("Substrate: %s", substrate_stats)
        except Exception as _sub_err:
            logging.debug("Substrate classification skipped: %s", _sub_err)

    # ── Save GeoTIFFs ─────────────────────────────────────────────────────────
    write_band(str(out / "snr_map.tif"), snr_map, profile)
    write_band(str(out / "confidence_map.tif"), conf_map.astype(np.float32), profile)
    write_band(str(out / "bottom_est.tif"), bottom_est, profile)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = {
        "image_date": date,
        "kd_estimation_method": kd_method,
        "kd_seasonal_prior": kd_seas,
        "kd_b02_estimated": round(kd_b02, 5),
        "kd_b03_estimated": round(kd_b03, 5),
        "kd_b04_estimated": None if kd_b04 is None else round(kd_b04, 5),
        "kd_high_uncertainty": kd_high_uncert,
        "sza_air_deg": sza_deg,
        "sza_water_deg": round(sza_water_deg, 3),
        "optical_path_m": round(path_m, 3),
        "water_transmittance_twoway": round(trans, 5),
        "glint_penalty": glint_pen,
        "snr_mean_16m": round(snr_mean, 4),
        "snr_median_16m": round(snr_median, 4),
        "percent_pixels_useful": round(pct_useful, 2),
        "percent_area_high_confidence": round(pct_high_conf, 2),
        "contrast_benthic_mean": round(contrast, 5),
        "visibility_score": round(vis_score, 5),
        "ranker_mode": ranker_mode,
        "sdb_depth_mean_m": round(sdb_mean, 2) if sdb_mean is not None else None,
        "snr_map": str(out / "snr_map.tif"),
        "confidence_map": str(out / "confidence_map.tif"),
        "bottom_est_map": str(out / "bottom_est.tif"),
        "sdb_depth_map": str(sdb_path) if sdb_path else None,
        "substrate_map": str(substrate_path) if substrate_path else None,
        "substrate_pct_sand": substrate_stats.get("Sand"),
        "substrate_pct_seagrass": substrate_stats.get("Seagrass/Macroalgae"),
        "substrate_pct_reef": substrate_stats.get("Rock-reef"),
        "kd490_map": str(out / "kd490_map.tif") if (out / "kd490_map.tif").exists() else None,
        "stumpf_m0_used": stumpf_m0,
        "stumpf_m1_used": stumpf_m1,
        "calibration_ensemble_sources": (
            bathy_result.get("ensemble_calibration", {}).get("sources_used") if bathy_result else None
        ),
        "bathy_zone": bathy_result.get("zone", {}).get("zone") if bathy_result else None,
        "bathy_optically_viable": bathy_result.get("zone", {}).get("optically_viable") if bathy_result else None,
        "bathy_calibration_rmse_m": bathy_result.get("calibration", {}).get("rmse_m") if bathy_result else None,
        "sdb_vs_chart_bias_m": (
            bathy_result.get("validation", {}).get("overall", {}).get("overall_bias_m") if bathy_result else None
        ),
        "sdb_vs_chart_rmse_m": (
            bathy_result.get("validation", {}).get("overall", {}).get("overall_rmse_m") if bathy_result else None
        ),
        # --- IH/DGRM bathymetry-derived features (new) ---
        "bathy_nearest_isobath_dist_m": bathy_feats.get("nearest_isobath_distance_m") if bathy_feats else None,
        "bathy_nearest_isobath_depth_m": bathy_feats.get("nearest_isobath_depth_m") if bathy_feats else None,
        "bathy_dist_10m_m": bathy_feats.get("dist_to_isobath_10m") if bathy_feats else None,
        "bathy_dist_20m_m": bathy_feats.get("dist_to_isobath_20m") if bathy_feats else None,
        "bathy_dist_30m_m": bathy_feats.get("dist_to_isobath_30m") if bathy_feats else None,
        "bathy_dist_50m_m": bathy_feats.get("dist_to_isobath_50m") if bathy_feats else None,
        "bathy_dist_100m_m": bathy_feats.get("dist_to_isobath_100m") if bathy_feats else None,
        "bathy_zone_class": bathy_feats.get("bathymetry_zone_class") if bathy_feats else None,
        "bathy_slope_proxy": bathy_feats.get("bathymetry_slope_proxy") if bathy_feats else None,
        "bathy_contour_density": bathy_feats.get("contour_density_proxy") if bathy_feats else None,
        "bathy_n_isobaths_aoi": bathy_feats.get("n_isobaths_in_aoi") if bathy_feats else None,
    }
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    logging.info(
        "Done | date=%s | Kd=%s(%.4f) | vis=%.4f | SNR=%.2f | SDB_mean=%.1fm",
        date,
        kd_method,
        kd_b02,
        vis_score,
        snr_mean,
        sdb_mean if sdb_mean is not None else 0,
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Reef ML Predictor v3 — Gordon/QAA + SDB")
    p.add_argument("--boa-b02", required=True)
    p.add_argument("--b03", default=None)
    p.add_argument("--b04", default=None)
    p.add_argument("--date", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=float, default=16.0)
    p.add_argument("--snr-threshold", type=float, default=3.0)
    p.add_argument(
        "--with-bathy-features", action="store_true", help="Compute IH/DGRM bathymetry-derived features (requires lat/lon)"
    )
    args = p.parse_args()
    from src.utils import compute_metadata_stub

    run_predictor(
        args.boa_b02,
        compute_metadata_stub(args.date),
        args.output,
        date=args.date,
        b03_path=args.b03,
        b04_path=args.b04,
        snr_threshold=args.snr_threshold,
        depth_target=args.depth,
        with_bathy_features=args.with_bathy_features,
    )
