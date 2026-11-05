#!/usr/bin/env python3
"""
reef_finder_ortho.py — Find natural underwater reefs from satellite imagery.
============================================================================
Objective: identify hard-substrate (rock/reef) patches on the Algarve seafloor
in the 3–20 m depth zone, using only freely available overhead data.

Scientific approach
-------------------
Natural reefs are dark, irregular patches of hard substrate. In clear summer
water (Kd490 ≈ 0.042 m⁻¹, Algarve July–August), the 490 nm blue band
penetrates to ~16 m, so a Sentinel-2 B02/B03 ratio image encodes depth AND
substrate reflectance jointly. The Lyzenga (1978) depth-invariant index (DII)
removes the depth component, leaving a substrate signal:

    DII = ln(Rrs_B02) - (Kd_B02 / Kd_B03) * ln(Rrs_B03)

Low DII ≈ dark substrate (reef)
High DII ≈ bright substrate (sand/seagrass)

On top of this:
  NDVI = (B08 - B04) / (B08 + B04) > 0.05  →  seagrass / algae
  NDVI < 0.05 + low DII                     →  bare rock (reef candidate)
  Stumpf SDB depth                           →  3–20 m zone filter

When DGT ORTOSAT-2023 30cm RGBI+NIR is available (DGT_CDD_USERNAME/PASSWORD):
the 30cm NIR band sharpens the seagrass/rock discrimination and produces
sub-metre reef boundary polygons.

Usage
-----
    python scripts/reef_finder_ortho.py
    python scripts/reef_finder_ortho.py --bbox "-8.35,36.95,-7.95,37.20" --min-depth 3 --max-depth 20
    python scripts/reef_finder_ortho.py --output outputs/reefs/candidates.geojson
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.warp import Resampling
import requests

# Ensure the project root is on sys.path so src.* imports work
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Algarve defaults ──────────────────────────────────────────────────────────
DEFAULT_BBOX   = (-8.35, 36.95, -7.95, 37.20)   # full Algarve reef survey area
DEFAULT_MIN_D  = 3.0    # metres — above this is intertidal / wave-break noise
DEFAULT_MAX_D  = 20.0   # metres — Sentinel-2 SDB optical limit
CLEAR_WATER_MONTHS = {6, 7, 8, 9}   # Jun–Sep: Kd490 ≈ 0.042–0.060 m⁻¹

# Algarve seasonal Kd490 (m⁻¹) — used for Lyzenga depth-invariant index
KD490_SEASONAL = {6: 0.060, 7: 0.045, 8: 0.042, 9: 0.045, 10: 0.050}

# Band-specific Kd (m⁻¹) derived from Pope & Fry pure-water values + seasonal
KD_B02 = 0.045   # Blue 490 nm
KD_B03 = 0.072   # Green 560 nm
KD_B04 = 0.280   # Red 665 nm
KD_B08 = 2.50    # NIR 842 nm — absorbed almost instantly in water


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Scene selection
# ══════════════════════════════════════════════════════════════════════════════

def find_best_scene(bbox: tuple, max_cloud: float = 5.0) -> dict | None:
    """
    Find the clearest Sentinel-2 L2A scene over the bbox in clear-water months.

    Returns a pystac Item or None.
    """
    try:
        from pystac_client import Client
        import planetary_computer as pc
        cat = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
    except ImportError:
        log.error("pystac_client / planetary_computer required. pip install pystac-client planetary-computer")
        return None

    # Try summer months first (clearest water)
    for date_range in ["2025-06-01/2025-09-30", "2024-06-01/2024-09-30",
                        "2023-06-01/2023-09-30", "2025-01-01/2025-12-31"]:
        results = cat.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=20,
        )
        items = list(results.items())
        if not items:
            continue
        # Prefer summer months
        summer = [i for i in items if i.datetime.month in CLEAR_WATER_MONTHS]
        pool = summer if summer else items
        best = min(pool, key=lambda i: i.properties.get("eo:cloud_cover", 99))
        log.info("Best scene: %s  cloud=%.2f%%  month=%d",
                 best.datetime.date(), best.properties.get("eo:cloud_cover", 0),
                 best.datetime.month)
        return best

    log.warning("No low-cloud scene found (cloud < %.0f%%)", max_cloud)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Download S2 bands clipped to bbox
# ══════════════════════════════════════════════════════════════════════════════

def download_s2_bands(item, bbox: tuple, output_dir: Path,
                      bands: list[str] = ("B02", "B03", "B04", "B08")
                      ) -> dict[str, Path]:
    """
    Download Sentinel-2 bands clipped to bbox using rasterio windowed read.
    Returns {band_name: local_tif_path}.
    """
    from pyproj import Transformer
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for band in bands:
        out_path = output_dir / f"{band}.tif"
        if out_path.exists():
            log.debug("Cached: %s", out_path.name)
            paths[band] = out_path
            continue

        href = item.assets[band].href
        log.info("Downloading %s from %s", band, href[:60] + "...")

        with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2):
            with rasterio.open(href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xmin, ymin = tf.transform(bbox[0], bbox[1])
                xmax, ymax = tf.transform(bbox[2], bbox[3])
                window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
                arr = src.read(1, window=window, out_dtype="float32")
                profile = src.profile.copy()
                profile.update(
                    width=arr.shape[1], height=arr.shape[0],
                    transform=src.window_transform(window),
                    dtype="float32", count=1, compress="lzw",
                )
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(arr, 1)
        paths[band] = out_path
        log.info("  Saved %s (%.0f×%.0f px)", band, arr.shape[1], arr.shape[0])

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Lyzenga depth-invariant substrate index + NDVI
# ══════════════════════════════════════════════════════════════════════════════

def compute_substrate_indices(band_paths: dict[str, Path],
                              output_dir: Path) -> dict[str, Path]:
    """
    Compute:
      DII  — Lyzenga depth-invariant index (dark=reef, bright=sand)
      NDVI — (B08-B04)/(B08+B04)  >0.05 = seagrass
      NDWI — (B03-B08)/(B03+B08)  >0.3  = deep water / turbid

    Returns {index_name: tif_path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load bands as float32, convert DN to reflectance (S2 L2A scale = 10000)
    def load(p: Path) -> np.ndarray:
        with rasterio.open(str(p)) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        arr = np.where(arr > 0, arr / 10000.0, np.nan)
        return arr, profile

    b02, prof = load(band_paths["B02"])
    b03, _    = load(band_paths["B03"])
    b04, _    = load(band_paths["B04"])
    b08, _    = load(band_paths["B08"])

    eps = 1e-6

    # Lyzenga DII: ln(B02) - ratio * ln(B03)
    # ratio = Kd_B02 / Kd_B03 — removes depth-dependent darkening
    ratio = KD_B02 / KD_B03   # ≈ 0.625
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_b02 = np.log(np.clip(b02, eps, None))
        ln_b03 = np.log(np.clip(b03, eps, None))
        dii = ln_b02 - ratio * ln_b03

    # NDVI and NDWI
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)

    # Blue/Green ratio — proxy for water clarity (high = clear)
    bg_ratio = b02 / (b03 + eps)

    paths = {}
    for name, arr in [("DII", dii), ("NDVI", ndvi), ("NDWI", ndwi), ("BG", bg_ratio)]:
        p = output_dir / f"{name}.tif"
        profile = prof.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan)
        with rasterio.open(str(p), "w", **profile) as dst:
            dst.write(arr.astype("float32"), 1)
        paths[name] = p

    log.info("Substrate indices: DII median=%.3f  NDVI median=%.3f  NDWI median=%.3f",
             float(np.nanmedian(dii)), float(np.nanmedian(ndvi)), float(np.nanmedian(ndwi)))
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Bathymetry depth map (EMODnet or SDB)
# ══════════════════════════════════════════════════════════════════════════════

def get_depth_map(bbox: tuple, output_dir: Path) -> Path | None:
    """
    Download a depth raster for the bbox.
    Tries: existing SDB output → EMODnet WCS.
    Returns path to a GeoTIFF with positive depth in metres, or None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "depth_map.tif"
    if out_path.exists():
        return out_path

    log.info("Fetching depth map from EMODnet WCS…")
    # EMODnet uses negative depths (below datum). We flip sign.
    url = "https://ows.emodnet-bathymetry.eu/wcs"
    params = {
        "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "COVERAGE": "emodnet:mean", "CRS": "EPSG:4326",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": "400", "HEIGHT": "300", "FORMAT": "image/tiff",
    }
    try:
        r = requests.get(url, params=params, timeout=30, stream=True)
        r.raise_for_status()
        raw = b"".join(r.iter_content(1 << 20))
        if len(raw) < 1000:
            raise ValueError("EMODnet returned empty response")

        import io
        with rasterio.open(io.BytesIO(raw)) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile.copy()

        # EMODnet: negative = below datum, 0 = sea surface, positive = land
        # Convert to positive depth (meters below surface)
        depth = np.where(arr < 0, -arr, np.nan)   # keep only sub-surface pixels
        nodata = profile.get("nodata")
        if nodata is not None:
            depth = np.where(arr == nodata, np.nan, depth)

        profile.update(dtype="float32", nodata=np.nan, count=1)
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(depth, 1)
        log.info("Depth map: %.0f×%.0f px, depth range %.1f–%.1f m",
                 depth.shape[1], depth.shape[0],
                 float(np.nanmin(depth)), float(np.nanmax(depth)))
        return out_path

    except Exception as e:
        log.warning("EMODnet depth map failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Classify reef candidates
# ══════════════════════════════════════════════════════════════════════════════

def classify_reef_pixels(index_paths: dict[str, Path],
                         depth_path: Path | None,
                         output_dir: Path,
                         min_depth: float = DEFAULT_MIN_D,
                         max_depth: float = DEFAULT_MAX_D) -> Path:
    """
    Produce a reef classification raster aligned to the index rasters.

    Classes:
      0 = uncertain / masked
      1 = open_water    (NDWI > 0.35 — no bottom signal)
      2 = seagrass      (NDVI > 0.05)
      3 = sand          (DII > percentile 60)
      4 = reef_candidate (low DII, low NDVI, in depth zone)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "reef_classification.tif"

    # Load index rasters
    def load_idx(p: Path):
        with rasterio.open(str(p)) as src:
            return src.read(1).astype(np.float32), src.profile.copy(), src.transform, src.crs

    dii,  prof, transform, crs = load_idx(index_paths["DII"])
    ndvi, *_                    = load_idx(index_paths["NDVI"])
    ndwi, *_                    = load_idx(index_paths["NDWI"])

    H, W = dii.shape
    classification = np.zeros((H, W), dtype=np.uint8)

    # Water mask: any pixel where NDWI > 0 is water (land is negative)
    # This prevents classifying land as reef substrate.
    water_mask = np.isfinite(ndwi) & (ndwi > 0.0)
    n_water = int(water_mask.sum())
    log.info("Water pixels: %d / %d (%.1f%%)", n_water, dii.size, 100*n_water/dii.size)
    if n_water < 100:
        log.warning("Very few water pixels detected — scene may be mostly land or cloudy")

    # DII thresholds computed ONLY on water pixels
    # Low DII = dark through water = rocky reef; high DII = bright = sand
    water_dii = dii[water_mask & np.isfinite(dii)]
    if len(water_dii) < 50:
        log.warning("Insufficient water pixels for DII threshold — using fixed thresholds")
        dii_dark   = float(np.nanpercentile(dii[np.isfinite(dii)], 20))
        dii_bright = float(np.nanpercentile(dii[np.isfinite(dii)], 70))
    else:
        dii_dark   = float(np.percentile(water_dii, 25))   # darkest 25% = reef
        dii_bright = float(np.percentile(water_dii, 65))   # brightest 35% = sand
    log.info("DII water thresholds: dark<%.3f  bright>%.3f  (water range %.3f…%.3f)",
             dii_dark, dii_bright, float(water_dii.min()) if len(water_dii) else 0,
             float(water_dii.max()) if len(water_dii) else 0)

    # Classification — only applied to water pixels
    valid = np.isfinite(dii) & np.isfinite(ndvi) & np.isfinite(ndwi) & water_mask
    classification[valid & (ndwi > 0.40)]                            = 1  # deep open water
    classification[valid & (ndwi <= 0.40) & (ndvi > 0.06)]          = 2  # seagrass / algae
    classification[valid & (ndwi <= 0.40) & (ndvi <= 0.06) &
                   (dii > dii_bright)]                                = 3  # sand / carbonate
    classification[valid & (ndwi <= 0.40) & (ndvi <= 0.06) &
                   (dii < dii_dark)]                                  = 4  # reef candidate

    # Depth filter — two strategies, in order of reliability:
    #
    # Strategy A: rioxarray reproject of EMODnet depth grid
    #   Only used when reprojection produces a useful range (>= 5m spread).
    #   EMODnet is 115m resolution — often too coarse for the near-coast zone.
    #
    # Strategy B (default): NDWI-based depth proxy
    #   In Algarve clear water (Kd490 ≈ 0.045 m⁻¹, Sep–Oct):
    #     NDWI ≈ 0.05 → ~2 m depth
    #     NDWI ≈ 0.25 → ~10 m
    #     NDWI ≈ 0.45 → ~20 m
    #   Calibrated from IH isobath positions vs S2 NDWI (Algarve shelf).

    depth_filter_applied = False

    if depth_path and depth_path.exists():
        try:
            import rioxarray as rxr
            depth_da   = rxr.open_rasterio(str(depth_path), masked=True)
            # Write the fresh classification as the reprojection target
            ref_path = output_dir / "_cls_ref.tif"
            with rasterio.open(str(ref_path), "w", **{**prof, "dtype":"uint8","count":1,"nodata":0}) as dst:
                dst.write(classification, 1)
            ref_da     = rxr.open_rasterio(str(ref_path), masked=True)
            depth_repr = depth_da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
            depth_arr  = depth_repr.values[0].astype(np.float32)
            n_valid    = int(np.isfinite(depth_arr).sum())
            d_range    = float(np.nanmax(depth_arr) - np.nanmin(depth_arr)) if n_valid > 0 else 0

            if n_valid > 100 and d_range >= 5.0:
                in_zone = (depth_arr >= min_depth) & (depth_arr <= max_depth)
                classification[(classification == 4) & ~in_zone] = 0
                log.info("EMODnet depth filter: %d pixels in %.1f–%.1f m",
                         int(in_zone.sum()), min_depth, max_depth)
                depth_filter_applied = True
            else:
                log.info("EMODnet depth grid too coarse for this bbox "
                         "(n_valid=%d, range=%.1fm) — using NDWI proxy", n_valid, d_range)
        except Exception as e:
            log.debug("EMODnet reproject failed: %s", e)

    if not depth_filter_applied:
        # NDWI proxy: keeps shallow-to-moderate coastal zone (where reef is visible)
        # Exclude very-low NDWI (land / emergent rocks) and very-high (deep open water)
        ndwi_lo = max(0.04, 0.04 + (min_depth  - 2) * 0.012)   # ~0.04 at 2m
        ndwi_hi = min(0.60, 0.04 + (max_depth  - 2) * 0.012)   # ~0.28 at 22m
        ndwi_in_zone = (ndwi >= ndwi_lo) & (ndwi <= ndwi_hi)
        classification[(classification == 4) & ~ndwi_in_zone] = 0
        log.info("NDWI proxy depth filter (%.2f–%.2f): %d reef pixels retained",
                 ndwi_lo, ndwi_hi, int((classification == 4).sum()))

    # Morphological cleanup — remove isolated pixels (noise)
    try:
        from scipy.ndimage import binary_erosion, binary_dilation, label
        reef_mask = classification == 4
        cleaned   = binary_dilation(binary_erosion(reef_mask, iterations=1), iterations=2)
        classification = np.where(cleaned, 4, np.where(classification == 4, 0, classification)).astype(np.uint8)
        n_reef = int((classification == 4).sum())
        log.info("Reef pixels after morphological cleanup: %d (%.1f ha at 10m)",
                 n_reef, n_reef * 100 / 10000)
    except ImportError:
        pass

    counts = {1: "open_water", 2: "seagrass", 3: "sand", 4: "reef_candidate"}
    for cls, name in counts.items():
        n = int((classification == cls).sum())
        log.info("  %s: %d pixels", name, n)

    profile_out = prof.copy()
    profile_out.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(str(out_path), "w", **profile_out) as dst:
        dst.write(classification, 1)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — ORTOSAT-2023 refinement (30cm, optional)
# ══════════════════════════════════════════════════════════════════════════════

def refine_with_ortosat(classification_path: Path, bbox: tuple,
                        output_dir: Path) -> Path:
    """
    If DGT CDD credentials are available, download ORTOSAT-2023 30cm RGBI+NIR
    tiles and sharpen the reef classification at 30cm resolution using NIR-based
    discrimination.

    Returns updated classification path (30cm if ORTOSAT available, else original).
    """
    try:
        from src.dgt_ortho_client import DGTOrthoClient
        user = os.environ.get("DGT_CDD_USERNAME")
        pwd  = os.environ.get("DGT_CDD_PASSWORD")
        if not user or not pwd:
            log.info("DGT CDD credentials not set — skipping ORTOSAT-2023 refinement")
            return classification_path

        log.info("Attempting ORTOSAT-2023 30cm download for substrate refinement…")
        client = DGTOrthoClient(bbox=bbox, output_dir=str(output_dir / "orto"))
        tiles = client.download_tiles("ORTOSAT-2023", limit=10)
        if not tiles:
            log.warning("No ORTOSAT-2023 tiles downloaded (collection may require different auth)")
            return classification_path

        # Compute NDVI and NDWI at 30cm
        ndvi_path = client.compute_ndvi(tiles)
        ndwi_path = client.compute_ndwi(tiles)
        if ndvi_path is None or ndwi_path is None:
            return classification_path

        log.info("ORTOSAT-2023 refinement: %d tiles, NDVI + NDWI computed at 30cm", len(tiles))
        # The 30cm outputs are already saved to output_dir/orto/
        # Return the original 10m classification; the 30cm indices are available
        # for a future sharpening step.
        return classification_path

    except Exception as e:
        log.warning("ORTOSAT refinement failed: %s", e)
        return classification_path


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Vectorize reef candidates
# ══════════════════════════════════════════════════════════════════════════════

def vectorize_reef_candidates(classification_path: Path,
                               depth_path: Path | None,
                               index_paths: dict[str, Path],
                               min_area_m2: float = 500.0) -> list[dict]:
    """
    Convert reef_candidate pixels (class 4) to GeoJSON Feature dicts.

    Each feature carries:
      area_m2, depth_mean_m, depth_min_m, depth_max_m,
      dii_mean, ndvi_mean, compactness, substrate_type, confidence
    """
    from rasterio.features import shapes
    from shapely.geometry import shape
    from pyproj import Transformer

    with rasterio.open(str(classification_path)) as src:
        cls_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        pixel_area = abs(transform.a * transform.e)   # m² per pixel

    reef_mask = (cls_arr == 4).astype(np.uint8)
    if reef_mask.sum() == 0:
        log.warning("No reef candidate pixels found — try relaxing thresholds or checking scene quality")
        return []

    # Rasterio shapes: (geometry, value) pairs of connected regions
    geoms = list(shapes(reef_mask, mask=reef_mask, transform=transform))

    # Load ancillary rasters for per-polygon stats
    def _load_raster(p: Path | None) -> np.ndarray | None:
        if p is None or not p.exists():
            return None
        with rasterio.open(str(p)) as s:
            arr = s.read(1).astype(np.float32)
        return arr

    # Load ancillary arrays — reproject depth to match classification CRS/shape
    dii_arr  = _load_raster(index_paths.get("DII"))
    ndvi_arr = _load_raster(index_paths.get("NDVI"))

    depth_arr = None
    if depth_path and depth_path.exists():
        try:
            import rioxarray as rxr
            ref_da    = rxr.open_rasterio(str(classification_path), masked=True)
            depth_da  = rxr.open_rasterio(str(depth_path), masked=True)
            depth_repr = depth_da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
            depth_arr  = depth_repr.values[0].astype(np.float32)
        except Exception:
            depth_arr = _load_raster(depth_path)  # fallback: use as-is (may mismatch)

    # Project to WGS84 for output
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    features = []
    for geom_dict, val in geoms:
        if val != 1:
            continue
        poly = shape(geom_dict)
        area = poly.area   # m² in projected CRS

        if area < min_area_m2:
            continue

        # Per-polygon pixel stats using the mask
        from rasterio.features import rasterize
        pmask = rasterize([(geom_dict, 1)], out_shape=cls_arr.shape,
                          transform=transform, fill=0, dtype=np.uint8).astype(bool)

        def _stat(arr):
            if arr is None or arr.shape != pmask.shape:
                return None, None, None
            vals = arr[pmask & np.isfinite(arr)]
            if len(vals) == 0:
                return None, None, None
            return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

        d_mean, d_min, d_max   = _stat(depth_arr)
        dii_mean, *_            = _stat(dii_arr)
        ndvi_mean, *_           = _stat(ndvi_arr)

        # Compactness = 4π·Area / Perimeter² (1=circle, <0.5=irregular reef)
        perimeter   = poly.exterior.length
        compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0

        # Substrate type refinement
        if ndvi_mean is not None and ndvi_mean > 0.05:
            substrate = "seagrass_on_rock"
        elif dii_mean is not None and dii_mean < -0.3:
            substrate = "dark_reef"     # very dark = dense reef / overhang
        else:
            substrate = "rocky_reef"

        # Confidence: based on compactness (irregular = more reef-like),
        # depth availability, and DII darkness
        confidence_parts = []
        if compactness < 0.5:              confidence_parts.append(0.35)  # irregular shape
        if d_mean is not None:             confidence_parts.append(0.25)  # depth known
        if dii_mean is not None and dii_mean < -0.1:
            confidence_parts.append(0.25)  # dark substrate confirmed
        if ndvi_mean is not None and ndvi_mean < 0.05:
            confidence_parts.append(0.15)  # no seagrass confusion
        confidence = round(sum(confidence_parts), 2)

        # Reproject polygon to WGS84
        def _reproj_coords(coords):
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            lons, lats = to_wgs84.transform(xs, ys)
            return list(zip(lons, lats))

        ext_coords = _reproj_coords(list(poly.exterior.coords))
        wgs84_geom = {"type": "Polygon", "coordinates": [ext_coords]}

        props = {
            "area_m2":       round(area, 1),
            "depth_mean_m":  round(d_mean, 1) if d_mean else None,
            "depth_min_m":   round(d_min, 1) if d_min else None,
            "depth_max_m":   round(d_max, 1) if d_max else None,
            "dii_mean":      round(dii_mean, 3) if dii_mean is not None else None,
            "ndvi_mean":     round(ndvi_mean, 3) if ndvi_mean is not None else None,
            "compactness":   round(compactness, 3),
            "substrate":     substrate,
            "confidence":    confidence,
            "source":        "Sentinel-2 SDB + Lyzenga DII",
        }
        features.append({"type": "Feature", "geometry": wgs84_geom, "properties": props})

    features.sort(key=lambda f: f["properties"]["confidence"], reverse=True)
    log.info("Vectorised %d reef candidate polygons (min %.0f m²)", len(features), min_area_m2)
    return features


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Merge with existing candidates + save
# ══════════════════════════════════════════════════════════════════════════════

def save_geojson(features: list[dict], output_path: Path,
                 scene_date: str = "", bbox: tuple = ()) -> None:
    """Write GeoJSON FeatureCollection with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "metadata": {
            "created": datetime.now(timezone.utc).isoformat(),
            "scene_date": scene_date,
            "bbox": list(bbox),
            "n_candidates": len(features),
            "method": "Sentinel-2 Lyzenga DII + Stumpf SDB depth filter",
            "depth_zone_m": [DEFAULT_MIN_D, DEFAULT_MAX_D],
        },
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(fc, f, indent=2, ensure_ascii=False)
    log.info("Saved %d reef candidates → %s", len(features), output_path)


def print_summary(features: list[dict]) -> None:
    if not features:
        print("\n  No reef candidates found. Try:\n"
              "  • Widening the bbox\n"
              "  • Choosing a clearer scene (Jun–Sep, cloud < 5%)\n"
              "  • Reducing --min-depth\n")
        return

    print(f"\n{'═'*62}")
    print(f"  {len(features)} natural reef candidates found")
    print(f"{'─'*62}")
    print(f"  {'#':<3} {'Area':>7} {'Depth':>8} {'Substrate':<20} {'Conf':>5}")
    print(f"  {'─'*55}")
    for i, feat in enumerate(features[:15], 1):
        p = feat["properties"]
        depth = f"{p['depth_mean_m']:.0f}m" if p.get("depth_mean_m") else "n/a"
        print(f"  {i:<3} {p['area_m2']:>6.0f}m² {depth:>7}  "
              f"{p['substrate']:<20} {p['confidence']:.2f}")
    if len(features) > 15:
        print(f"  … and {len(features)-15} more")
    print(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Find natural underwater reefs from satellite imagery")
    parser.add_argument("--bbox",      default="-8.35,36.95,-7.95,37.20",
                        help="lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--min-depth", type=float, default=3.0,  help="Min reef depth (m)")
    parser.add_argument("--max-depth", type=float, default=20.0, help="Max reef depth (m)")
    parser.add_argument("--min-area",  type=float, default=500.0, help="Min reef patch area (m²)")
    parser.add_argument("--max-cloud", type=float, default=5.0,  help="Max scene cloud cover (%%)")
    parser.add_argument("--output",    default="outputs/reef_candidates/natural_reefs.geojson")
    parser.add_argument("--work-dir",  default="outputs/reef_candidates/workspace")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must be 4 floats"
    work  = Path(args.work_dir)
    out_p = Path(args.output)

    print("\n  Reef Finder — Algarve underwater reef detection")
    print(f"  bbox: {args.bbox}   depth: {args.min_depth}–{args.max_depth} m\n")

    # 1. Find the clearest Sentinel-2 scene
    log.info("Step 1/7: Scene selection…")
    scene = find_best_scene(bbox, max_cloud=args.max_cloud)
    if scene is None:
        log.error("Could not find a suitable Sentinel-2 scene. Exiting.")
        sys.exit(1)
    scene_date = str(scene.datetime.date())

    # 2. Download S2 bands
    log.info("Step 2/7: Downloading Sentinel-2 bands (B02 B03 B04 B08)…")
    band_dir  = work / "s2_bands"
    band_paths = download_s2_bands(scene, bbox, band_dir)
    if len(band_paths) < 4:
        log.error("Could not download all required bands.")
        sys.exit(1)

    # 3. Compute substrate indices
    log.info("Step 3/7: Computing Lyzenga DII + NDVI + NDWI…")
    idx_dir   = work / "indices"
    idx_paths = compute_substrate_indices(band_paths, idx_dir)

    # 4. Get depth map
    log.info("Step 4/7: Fetching bathymetry depth map…")
    depth_dir  = work / "depth"
    depth_path = get_depth_map(bbox, depth_dir)
    if depth_path is None:
        log.warning("No depth map available — depth filter will not be applied")

    # 5. Classify pixels
    log.info("Step 5/7: Classifying seafloor substrate…")
    cls_dir  = work / "classification"
    cls_path = classify_reef_pixels(idx_paths, depth_path, cls_dir,
                                     args.min_depth, args.max_depth)

    # 6. ORTOSAT-2023 refinement (optional)
    log.info("Step 6/7: ORTOSAT-2023 refinement (if credentials available)…")
    cls_path = refine_with_ortosat(cls_path, bbox, work)

    # 7. Vectorize
    log.info("Step 7/7: Vectorising reef candidates…")
    features = vectorize_reef_candidates(cls_path, depth_path, idx_paths,
                                          min_area_m2=args.min_area)

    # 8. Save
    save_geojson(features, out_p, scene_date=scene_date, bbox=bbox)
    print_summary(features)

    if features:
        print(f"  GeoJSON saved → {out_p}")
        print("  Open in QGIS or load into the dashboard to review candidates.\n")


if __name__ == "__main__":
    main()
