"""
coastsat_preprocess.py — CoastSat wrapper for Sentinel-2 BOA preprocessing.

Replaces the custom 'sentinel' + 'ratio' pipeline steps with CoastSat's
satellite imagery retrieval and cloud-masking pipeline, which is CPU-friendly
and handles the GEE auth + download in one call.

Falls back gracefully to the existing src.stac_ingest / src.reef_ml_predictor_acolite
path if CoastSat is not installed.

Output is compatible with the existing 'bathy' step — produces the same
BOA reflectance GeoTIFFs in the output_dir that the Stumpf SDB step expects.

Usage:
    from pipelines.coastsat_preprocess import preprocess_site

    bands = preprocess_site(
        site_name="pedra_do_alto",
        lon=-8.2103, lat=37.069,
        date="2024-09-30",
        output_dir="outputs/pedra_do_alto",
    )
    # bands: {"B02": Path(...), "B03": Path(...), "B04": Path(...)}

Requirements (optional — falls back if absent):
    pip install coastsat
    # CoastSat also needs Google Earth Engine credentials:
    # earthengine authenticate
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from pipelines.cache_manager import S2Cache

log = logging.getLogger(__name__)

# Algarve bounding box used as default search window
_ALGARVE_BBOX = (-9.5, 36.95, -7.35, 37.20)

# Cloud-cover threshold accepted by CoastSat
_CLOUD_THRESHOLD = 0.05


def _has_coastsat() -> bool:
    try:
        import coastsat  # noqa: F401
        return True
    except ImportError:
        return False


def _coastsat_download(
    site_name: str,
    lon: float,
    lat: float,
    date: str,
    output_dir: Path,
    buffer_m: float = 1000.0,
) -> dict[str, Path] | None:
    """
    Download and cloud-mask a Sentinel-2 scene via CoastSat + GEE.

    Returns a dict {band_name: Path} pointing to the BOA TIFs, or None on failure.
    """
    try:
        from coastsat import SDS_download, SDS_tools  # type: ignore
        import ee

        ee.Initialize()

        polygon = SDS_tools.bbox_to_polygon(lon - buffer_m / 111_320,
                                            lat - buffer_m / 111_320,
                                            lon + buffer_m / 111_320,
                                            lat + buffer_m / 111_320)
        inputs = {
            "sitename":       site_name,
            "polygon":        polygon,
            "dates":          [date, date],
            "sat_list":       ["S2"],
            "filepath":       str(output_dir.parent),
            "cloud_thresh":   _CLOUD_THRESHOLD,
            "cloud_mask_issue": False,
            "s2cloudless_prob": 40,
        }

        t0 = time.perf_counter()
        metadata = SDS_download.retrieve_images(inputs)
        log.info("CoastSat download: %.1f s", time.perf_counter() - t0)

        # CoastSat stores scenes under filepath/sitename/S2/
        s2_dir = Path(inputs["filepath"]) / site_name / "S2"
        tifs = sorted(s2_dir.glob("*.tif"))
        if not tifs:
            log.warning("CoastSat: no TIFs found in %s", s2_dir)
            return None

        # Map standard band names to files (CoastSat names them by band number)
        band_map: dict[str, Path] = {}
        for tif in tifs:
            stem = tif.stem.upper()
            for band in ("B02", "B03", "B04"):
                if band in stem:
                    band_map[band] = tif
        return band_map if len(band_map) >= 2 else None

    except Exception as exc:
        log.warning("CoastSat download failed: %s", exc)
        return None


def _fallback_stac(
    site_name: str,
    lon: float,
    lat: float,
    date: str,
    output_dir: Path,
) -> dict[str, Path] | None:
    """
    Fallback: use the existing STAC + ACOLITE pipeline steps.
    Returns band TIF paths if successful, else None.
    """
    try:
        from src.stac_ingest import search_sentinel2_scene
        from src.reef_ml_predictor_acolite import run_predictor

        scene = search_sentinel2_scene(lon=lon, lat=lat, date=date)
        if scene is None:
            log.warning("STAC fallback: no scene found for %s @ %s", site_name, date)
            return None

        result = run_predictor(scene, output_dir=str(output_dir))
        if result is None:
            return None

        # run_predictor writes bathy + band TIFs under output_dir
        band_map = {
            b: output_dir / f"{b}.tif"
            for b in ("B02", "B03", "B04")
            if (output_dir / f"{b}.tif").exists()
        }
        return band_map if band_map else None

    except Exception as exc:
        log.warning("STAC fallback failed: %s", exc)
        return None


def preprocess_site(
    site_name: str,
    lon: float,
    lat: float,
    date: str,
    output_dir: str | Path,
    cache: S2Cache | None = None,
    force_refresh: bool = False,
) -> dict[str, Path] | None:
    """
    Download and preprocess Sentinel-2 BOA reflectance for one site + date.

    Tries CoastSat first; falls back to the existing STAC pipeline if CoastSat
    is unavailable or fails.  Results are cached — set force_refresh=True to
    re-download.

    Args:
        site_name:     Human-readable site key (used for cache keys and filenames).
        lon, lat:      Site centre coordinates (WGS84 decimal degrees).
        date:          Target acquisition date "YYYY-MM-DD".
        output_dir:    Directory where output TIFs will be written.
        cache:         S2Cache instance; creates a default one if None.
        force_refresh: Ignore cached data and re-download.

    Returns:
        Dict mapping band name → Path to BOA TIF, or None on total failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache is None:
        cache = S2Cache()

    # Check BOA cache
    if not force_refresh and cache.has_boa(site_name, date):
        log.info("[%s %s] BOA cache hit — skipping download", site_name, date)
        _, meta = cache.load_boa(site_name, date)
        return {b: output_dir / f"{b}.tif" for b in meta.get("bands", [])}

    log.info("[%s %s] Preprocessing — CoastSat=%s", site_name, date, _has_coastsat())
    t0 = time.perf_counter()

    band_map: dict[str, Path] | None = None

    if _has_coastsat():
        band_map = _coastsat_download(site_name, lon, lat, date, output_dir)

    if band_map is None:
        log.info("[%s %s] Falling back to STAC/ACOLITE", site_name, date)
        band_map = _fallback_stac(site_name, lon, lat, date, output_dir)

    if band_map is None:
        log.error("[%s %s] Preprocessing failed — all methods exhausted", site_name, date)
        return None

    # Persist BOA arrays to cache
    try:
        import rasterio
        bands_np: dict[str, np.ndarray] = {}
        for band_name, tif_path in band_map.items():
            with rasterio.open(tif_path) as src:
                bands_np[band_name] = src.read(1)
        cache.save_boa(site_name, date, bands_np, meta={"source": "coastsat" if _has_coastsat() else "stac"})
    except Exception as exc:
        log.warning("Could not cache BOA arrays: %s", exc)

    log.info("[%s %s] Preprocessing complete in %.1f s", site_name, date, time.perf_counter() - t0)
    return band_map
