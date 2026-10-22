#!/usr/bin/env python3
"""
Utilities: raster I/O, sunglint correction, refraction, Beer-Lambert.
"""

import math
from datetime import UTC

import numpy as np
import rasterio


def read_band(path, handle_nodata=True):
    """Read raster band as float32. If handle_nodata=True, converts nodata to np.nan."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if handle_nodata and nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def write_band(path, arr, profile, nodata=None):
    """Write array to GeoTIFF. If nodata is None and arr contains NaN, uses np.nan as nodata."""
    profile = profile.copy()
    has_nan = np.isnan(arr).any()
    if nodata is None:
        nodata = np.nan if has_nan else None
    # For integer-typed arrays, use 255 as nodata sentinel if not set
    if nodata is None and np.issubdtype(arr.dtype, np.integer):
        nodata = 255
    profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=nodata)
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def simulate_acolite_boa(input_tif, output_tif, b03_tif=None, sunglint_strength=0.8):
    """
    Simulates ACOLITE BOA from L2A B02:
    - Hedley linear sunglint correction (if B03 provided), else empirical subtraction.
    - Converts raw DN to BOA reflectance (divide by 10000).
    - No negative values.
    Drop-in: replace with real acolite_cli output GeoTIFFs and remove this function.
    """
    arr, profile = read_band(str(input_tif))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert L2A DN to reflectance if not already (values > 2 = raw DN).
    # nanmax guards against all-NaN tiles crashing with ValueError.
    if np.nanmax(arr) > 2.0:
        arr = arr / 10000.0

    if b03_tif is not None:
        b03, _ = read_band(str(b03_tif))
        if np.nanmax(b03) > 2.0:
            b03 = b03 / 10000.0
        # Hedley linear: slope from deep-water pixels
        b03_pos = b03[b03 > 0]
        if b03_pos.size == 0:
            b03_pos = b03[b03 >= 0]  # fall back to non-negative if all-zero
        p20 = np.percentile(b03_pos, 20) if b03_pos.size > 0 else 0.0
        mask = (arr > 0) & (b03 > 0) & (b03 < p20)
        if mask.sum() > 10:
            slope = np.cov(arr[mask].ravel(), b03[mask].ravel())[0, 1] / (np.var(b03[mask]) + 1e-12)
            slope = np.clip(slope, 0, 2.0)
            arr = arr - slope * (b03 - b03[mask].min())
    else:
        # Empirical: subtract fraction of high-percentile tail
        p95 = np.percentile(arr[arr > 0], 95) if np.any(arr > 0) else 0.0
        arr = arr - sunglint_strength * p95 * 0.05

    arr = np.clip(arr, 0, 1.0)  # physical reflectance range; NO min-max stretch
    write_band(str(output_tif), arr.astype(np.float32), profile)
    return str(output_tif)


def snell_air_to_water(theta_air_rad, n_water=1.333):
    """Snell's law: returns refracted angle (radians) in water for ray entering from air."""
    s = math.sin(theta_air_rad) / n_water
    s = max(-0.999999, min(0.999999, s))
    return math.asin(s)  # angle in [0, pi/2]


def snell_sza(sza_deg, n_water=1.333):
    """Return (sza_water_deg, theta_water_rad)."""
    rad = math.radians(sza_deg)
    sin_w = math.sin(rad) / n_water
    sin_w = max(-0.999999, min(0.999999, sin_w))
    theta_w = math.asin(sin_w)
    return math.degrees(theta_w), theta_w


def optical_path(depth_m, theta_water_rad):
    return depth_m / max(1e-6, math.cos(theta_water_rad))


def beer_lambert_transmittance(kd, path_m):
    """Two-way transmittance (surface → bottom → sensor)."""
    return math.exp(-2 * kd * path_m)


def get_kd490(month, kd_prior: dict):
    return kd_prior.get(int(month), kd_prior.get(str(month), 0.080))


def get_kd490_map(
    b02_arr: "np.ndarray",
    b03_arr: "np.ndarray",
    b04_arr: "np.ndarray",
) -> "np.ndarray":
    """
    Per-pixel diffuse attenuation coefficient Kd(490) in m⁻¹.

    Implements the Lee et al. (2013) empirical band-ratio model adapted for
    Sentinel-2 (B02≈490 nm, B03≈560 nm, B04≈665 nm):

        X = log10( max(Rrs_blue, Rrs_green) / (0.5·Rrs_green + 1.5·Rrs_red) )
        Kd = 10^(a0 + a1·X + a2·X² + a3·X³)    [Lee 2013, Table 2]

    Captures the 3–5× turbidity gradient between the Guadiana plume (east)
    and open Atlantic water (west) that a single scene-average Kd misses.

    Returns a float32 array clipped to [0.01, 2.0] m⁻¹.
    """
    import numpy as _np

    eps = 1e-6
    Rb = _np.maximum(b02_arr, eps).astype(_np.float64)
    Rg = _np.maximum(b03_arr, eps).astype(_np.float64)
    Rr = _np.maximum(b04_arr, eps).astype(_np.float64)

    # Band-ratio numerator: whichever of blue/green is higher
    num = _np.maximum(Rb, Rg)
    den = 0.5 * Rg + 1.5 * Rr + eps

    X = _np.log10(_np.maximum(num / den, eps))

    # Lee 2013 Table 2 polynomial coefficients for Kd(490)
    a0, a1, a2, a3 = 0.0428, -1.2281, -0.6079, -0.2024
    kd = _np.power(10.0, a0 + a1 * X + a2 * X**2 + a3 * X**3)

    # Physical range: clear ocean (0.01 m⁻¹) to very turbid coastal (2.0 m⁻¹)
    kd = _np.clip(kd, 0.01, 2.0)
    return kd.astype(_np.float32)


def build_coastal_geojson(csv_path, geojson_path):
    """Convert algarve_coastal_features.csv → GeoJSON FeatureCollection.

    Creates a timestamped backup of the existing GeoJSON before overwriting.
    Writes atomically (temp file → rename) so a concurrent dashboard read never
    sees a half-written file.  Returns the number of features written.
    """
    import csv as _csv
    import json as _json
    import os as _os
    import shutil as _shutil
    import tempfile as _tmpfile
    from datetime import datetime

    csv_path = str(csv_path)
    geojson_path = str(geojson_path)

    with open(csv_path, newline="") as fh:
        rows = list(_csv.DictReader(fh))

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    # Non-numeric columns (strings / ISO timestamps)
    _str_cols = {"site_name", "timestamp"}

    features = []
    ts = datetime.now(UTC).isoformat()
    for row in rows:
        props = {}
        for k, v in row.items():
            props[k] = v if k in _str_cols else float(v)
        props["timestamp"] = ts
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )

    collection = {
        "type": "FeatureCollection",
        "name": "algarve_coastal_features",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }

    # Backup existing file before overwriting
    if _os.path.exists(geojson_path):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        _shutil.copy2(geojson_path, geojson_path + f".bak_{stamp}")

    # Atomic write
    dir_ = _os.path.dirname(_os.path.abspath(geojson_path))
    fd, tmp = _tmpfile.mkstemp(dir=dir_, suffix=".geojson.tmp")
    try:
        with _os.fdopen(fd, "w") as fh:
            _json.dump(collection, fh, indent=1, ensure_ascii=False)
        _os.replace(tmp, geojson_path)
    except Exception:
        _os.unlink(tmp)
        raise

    return len(features)


def compute_metadata_stub(date):
    """
    Minimal metadata for simulated mode.
    Replace with real STAC/MTD metadata extraction in production.
    """
    known = {
        "2025-09-25": {"sza": 40.498, "saa": 158.883, "cloud": 1.245, "level": "L2A"},
        "2023-10-01": {"sza": 42.413, "saa": 160.459, "cloud": 0.007, "level": "L2A"},
    }
    m = known.get(date, {"sza": 40.0, "saa": 150.0, "cloud": 2.0, "level": "L2A"})
    return {
        "date": date,
        "crs": "EPSG:32629",
        "datum": "WGS84",
        "level": m["level"],
        "solar_zenith_deg": m["sza"],
        "solar_azimuth_deg": m["saa"],
        "satellite_zenith_deg": 5.0,
        "satellite_azimuth_deg": 10.0,
        "cloud_cover_pct": m["cloud"],
    }
