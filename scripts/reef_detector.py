#!/usr/bin/env python3
"""
reef_detector.py — Find reef-like seafloor signatures in Sentinel-2 imagery.

Strategy (physically grounded, GPS-anchored):
  1. Download the 4 optical bands (B02/B03/B04/B08) for a scene over a bbox.
  2. Measure REAL clarity at the GPS reef pixel: NIR glint level + bottom contrast.
     (This answers "which scene has the better light" from the actual pixels,
     not from a precomputed SNR label.)
  3. Apply the Lyzenga depth-invariant transform: DII = X_B02 - (k02/k03)·X_B03,
     where Xi = ln(Ri - Lsw_i) and Lsw_i is the deep-water background.
     The k02/k03 attenuation ratio is estimated automatically per scene
     (Lyzenga 1978 variance method) — no hardcoded coefficient.
  4. Add local texture (rugosity proxy): reef is spatially heterogeneous,
     sand is smooth.
  5. Combine DII + texture into a reef-likelihood map over the WHOLE scene,
     anchored/validated at the 2 GPS-confirmed reef sites.
  6. Output: reef_likelihood.tif (GeoTIFF) + overlay PNG + report JSON.

The 2 GPS reef points are the only ground truth. Everything the map highlights
ELSEWHERE is a CANDIDATE with the same optical signature — to be confirmed by
dive survey, not asserted as fact.

Missing-band / radiometry policy (2026-10-09)
---------------------------------------------
* Bands are read on the scene's NATIVE grid (UTM CRS + the affine transform of the
  window actually read). Lon/lat points are projected into that CRS before indexing.
* DN → reflectance uses the asset's own ``raster:bands`` scale/offset (see
  src/s2_radiometry.py). No metadata → ``radiometry_unknown`` → band unusable.
* Nodata (DN 0) is masked before arithmetic; invalid pixels are NaN and excluded
  from every statistic. Valid negative reflectances are kept.
* Feature dependencies live in src/band_registry.py. B08 gates the water/glint
  mask and therefore everything downstream: without genuine, valid B08 the scene
  result is ``status="unavailable"`` — never a likelihood of 0, never a fabricated
  band, never a reduced-band model.

Usage
-----
    python scripts/reef_detector.py \
        --scene S2A_29SNB_20230901_0_L2A \
        --scene S2B_29SNB_20250925_0_L2A
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import rowcol, xy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import band_registry  # noqa: E402
from src.s2_grid import WindowOutsideRaster, bbox_window, unsigned_href  # noqa: E402
from src.s2_radiometry import (  # noqa: E402
    dn_to_reflectance,
    radiometry_from_stac_asset,
    unknown as radiometry_unknown,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Study area and ground truth ────────────────────────────────────────────────
# Western Algarve shelf (Quarteira → Armação de Pêra), includes open water to
# the south for reliable Lsw (deep-water background) estimation.
# Eastern sites (Tavira, Fuseta, Faro, Olhão) are on separate S2 tiles — run
# separately with tile 29SNC scenes.
DEFAULT_BBOX = (-8.45, 36.95, -8.05, 37.15)   # lon_min, lat_min, lon_max, lat_max

# GPS-confirmed dive sites are marked confirmed=True.
# Survey-hypothesis sites (from algarve_reef_survey.py) are unconfirmed candidates
# that share a reef-like bathymetric setting — to be validated by dive.
REEF_SITES = [
    # ── GPS dive-confirmed ─────────────────────────────────────────────────────
    {"name": "Pedra_Sta_Eulalia",    "lon": -8.2102,  "lat": 37.0691 },
    {"name": "Pedra_do_Alto",        "lon": -8.20982, "lat": 37.05815},
    # ── Fishing GPS (CarlosBarrinha.gpx, ≥3 repeated visits, 5–22 m depth) ────
    {"name": "fishing_N_eulalia_8m", "lon": -8.20294, "lat": 37.07504},  # 6 visits, 8 m
    {"name": "fishing_gale_16m",     "lon": -8.22965, "lat": 37.05822},  # 6 visits, 16 m
    {"name": "fishing_gale_15m_a",   "lon": -8.23384, "lat": 37.06220},  # 5 visits, 15 m
    {"name": "fishing_deep_E_19m",   "lon": -8.12089, "lat": 37.03144},  # 5 visits, 19 m
    {"name": "fishing_eulalia_11m",  "lon": -8.21477, "lat": 37.06629},  # 4 visits, 11 m
    {"name": "fishing_midshelf_19m", "lon": -8.17101, "lat": 37.04593},  # 4 visits, 19 m
    {"name": "fishing_gale_16m_b",   "lon": -8.23001, "lat": 37.05723},  # 4 visits, 16 m
    {"name": "fishing_gale_15m_b",   "lon": -8.23315, "lat": 37.06015},  # 4 visits, 15 m
    # ── Fishing GPS — EASTERN Algarve (tile 29SNA, top by reef_score) ───────────
    {"name": "e_faro_reef_12m_a",    "lon": -8.02339, "lat": 37.00917},  # 12 visits TRI=0.65 reef_score=26
    {"name": "e_faro_reef_11m",      "lon": -8.02498, "lat": 37.00989},  # 12 visits TRI=0.60 reef_score=24
    {"name": "e_tavira_13m",         "lon": -7.94326, "lat": 36.96016},  # 7 visits  TRI=0.82 reef_score=19
    {"name": "e_faro_reef_12m_b",    "lon": -8.02327, "lat": 37.00857},  # 10 visits TRI=0.57 reef_score=19
    {"name": "e_olhao_17m_a",        "lon": -7.97075, "lat": 36.96263},  # 12 visits TRI=0.47 reef_score=19
    {"name": "e_olhao_17m_b",        "lon": -7.97262, "lat": 36.96305},  # 12 visits TRI=0.45 reef_score=18
    {"name": "e_olhao_12m",          "lon": -7.94072, "lat": 36.95948},  # 11 visits TRI=0.47 reef_score=17
    {"name": "e_faro_16m_highTRI",   "lon": -7.99863, "lat": 36.98575},  # 5 visits  TRI=0.99 reef_score=17
    # ── Survey hypotheses (algarve_reef_survey.py, W cluster) ─────────────────
    {"name": "quarteira",            "lon": -8.1000,  "lat": 37.0650 },
    {"name": "vilamoura",            "lon": -8.1300,  "lat": 37.0750 },
    {"name": "olhos_de_agua",        "lon": -8.1600,  "lat": 37.0900 },
    {"name": "albufeira_reef",       "lon": -8.2105,  "lat": 37.0690 },
    {"name": "gale",                 "lon": -8.2296,  "lat": 37.0560 },
    {"name": "salgados",             "lon": -8.3000,  "lat": 37.0950 },
    {"name": "armacao_de_pera",      "lon": -8.3600,  "lat": 37.0700 },
]

# Names physically verified by divers. Fishing spots are high-confidence repeated
# GPS fixes but not dive-confirmed — keep separate for honest reporting.
CONFIRMED_REEF_NAMES = {"Pedra_Sta_Eulalia", "Pedra_do_Alto"}
FISHING_GPS_NAMES = {s["name"] for s in REEF_SITES if str(s["name"]).startswith("fishing_")}

# Earth Search asset names for the 10 m bands
_EARTH_SEARCH  = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"
_BAND_ASSETS   = {"blue": "blue", "green": "green", "red": "red", "nir": "nir"}
# Detector band key → Sentinel-2 band id (registry vocabulary)
_BAND_IDS      = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}


class DetectorUnavailable(RuntimeError):
    """Raised when genuine inputs are insufficient to compute a gated feature."""

OUTPUT_DIR = Path("outputs/reef_detector")

# Glint threshold: NIR BOA reflectance above this = surface contamination / land
GLINT_NIR_MAX = 0.015

# Optical depth window (m, negative = below MSL). Bottom is only visible in this
# range; beyond it the signal is sensor noise. Used to mask out deep water.
EMODNET_DEPTH = Path("outputs/reef_habitat/emodnet/depth_algarve.tif")
DEPTH_SHALLOW = -2.0     # exclude breaking-wave surf zone
DEPTH_DEEP    = -22.0    # Sentinel-2 optical floor in clear Algarve water

# Minimum shallow clean-water pixels for a meaningful Lyzenga k-ratio
MIN_DII_PIXELS = 50


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scene download (windowed COG read over bbox)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_assets(scene_id: str) -> dict:
    """
    Return the STAC item (as plain dicts) for *scene_id* via Earth Search.

    Keeps the full asset dicts (incl. ``raster:bands`` scale/offset/nodata) and the
    product provenance properties so radiometry is decided from metadata, not assumed.
    """
    from pystac_client import Client
    cat = Client.open(_EARTH_SEARCH)
    res = cat.search(collections=[_S2_COLLECTION], ids=[scene_id], max_items=1)
    items = list(res.items())
    if not items:
        raise RuntimeError(f"Scene not found in STAC: {scene_id}")
    it = items[0]
    return {
        "id":          it.id,
        "provider":    _EARTH_SEARCH,
        "datetime":    it.properties["datetime"][:10],
        "acquired":    it.properties.get("datetime"),
        "cloud_cover": it.properties.get("eo:cloud_cover", -1),
        "platform":    it.properties.get("platform", "?"),
        "properties":  dict(it.properties),
        "assets":      {k: v.to_dict() for k, v in it.assets.items()},
    }


_PROVENANCE_KEYS = ("s2:product_uri", "s2:datatake_id", "s2:processing_baseline", "s2:granule_id",
                    "earthsearch:boa_offset_applied", "processing:software", "s2:generation_time")


def _bbox_window(src, bbox: tuple) -> Window:
    """Integer native-grid window for *bbox* (see src/s2_grid.bbox_window)."""
    try:
        return bbox_window(src, bbox)
    except WindowOutsideRaster as exc:
        raise DetectorUnavailable(str(exc)) from exc


def load_scene_from_item(item: dict, bbox: tuple, opener=rasterio.open) -> dict:
    """
    Read blue/green/red/nir over *bbox* on the scene's native grid.

    Returns dict with:
        bands       : {band: (H,W) float32 reflectance, NaN where invalid} — only usable bands
        valid       : {band: (H,W) bool}  explicit validity masks
        band_status : {band: "ok" | "missing_asset" | "radiometry_unknown" | "no_valid_pixels" | "grid_mismatch"}
        radiometry  : {band: Radiometry.as_dict()}
        usable_bands: sorted Sentinel-2 ids (e.g. ["B02","B03","B08"])
        transform   : affine transform of the window actually read (native CRS)
        crs         : native CRS of the rasters
        meta        : id/date/cloud/platform + product provenance + asset hrefs
    No band is ever synthesised: an unusable band is simply absent from ``bands``.
    """
    props = item.get("properties", {})
    meta = {
        "id": item.get("id"), "datetime": item.get("datetime"), "acquired": item.get("acquired"),
        "cloud_cover": item.get("cloud_cover", -1), "platform": item.get("platform", "?"),
        "provider": item.get("provider"),
        "provenance": {k: props[k] for k in _PROVENANCE_KEYS if k in props},
        "asset_hrefs": {},
    }
    log.info("Loading %s  date=%s  cloud=%.2f%%  platform=%s",
             meta["id"], meta["datetime"], float(meta["cloud_cover"]), meta["platform"])

    bands: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    status: dict[str, str] = {}
    radiometry: dict[str, dict] = {}
    ref_grid = None  # (crs, window_transform, shape) of the first band read

    for band, asset_key in _BAND_ASSETS.items():
        asset = item.get("assets", {}).get(asset_key)
        if asset is None or not asset.get("href"):
            status[band] = "missing_asset"
            radiometry[band] = radiometry_unknown("asset absent from item").as_dict()
            log.warning("  %s (%s): asset missing — band unavailable, not substituted", band, _BAND_IDS[band])
            continue
        meta["asset_hrefs"][band] = unsigned_href(asset["href"])
        rad = radiometry_from_stac_asset(asset, props)
        radiometry[band] = rad.as_dict()
        if not rad.is_known:
            status[band] = "radiometry_unknown"
            log.warning("  %s (%s): radiometry_unknown (%s) — blocked", band, _BAND_IDS[band], rad.source)
            continue
        with opener(asset["href"]) as src:
            window = _bbox_window(src, bbox)
            wt = window_transform(window, src.transform)
            dn = src.read(1, window=window)
            grid = (src.crs, wt, dn.shape)
            if ref_grid is None:
                ref_grid = grid
            elif (grid[0] != ref_grid[0] or not grid[1].almost_equals(ref_grid[1]) or grid[2] != ref_grid[2]):
                # Never silently crop/resample one band onto another's grid.
                status[band] = "grid_mismatch"
                log.warning("  %s: grid differs from reference band — unavailable", band)
                continue
            refl, ok = dn_to_reflectance(dn, rad, extra_nodata=src.nodata)
        if not ok.any():
            status[band] = "no_valid_pixels"
            log.warning("  %s (%s): no valid pixels in window — unavailable", band, _BAND_IDS[band])
            continue
        bands[band], valid[band], status[band] = refl, ok, "ok"
        log.info("  %s (%s): %d/%d valid px  scale=%g offset=%g  [%s]", band, _BAND_IDS[band],
                 int(ok.sum()), ok.size, rad.scale, rad.offset, rad.source)

    if ref_grid is None:
        raise DetectorUnavailable(f"no band of {meta['id']} could be read")
    crs, transform, shape = ref_grid
    usable = sorted(_BAND_IDS[b] for b, st in status.items() if st == "ok")
    log.info("  Native grid %s  %dx%d px  usable=%s", crs, shape[0], shape[1], usable)
    return {"bands": bands, "valid": valid, "band_status": status, "radiometry": radiometry,
            "usable_bands": usable, "transform": transform, "crs": crs, "shape": shape, "meta": meta}


def load_scene(scene_id: str, bbox: tuple) -> dict:
    """Resolve *scene_id* on Earth Search and load it (see load_scene_from_item)."""
    return load_scene_from_item(_resolve_assets(scene_id), bbox)


def lonlat_to_rowcol(scene: dict, lon: float, lat: float) -> tuple[int, int]:
    """Project WGS-84 lon/lat into the scene CRS and return the containing pixel (row, col)."""
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", scene["crs"], always_xy=True)
    x, y = tf.transform(lon, lat)
    r, c = rowcol(scene["transform"], x, y)
    return int(r), int(c)


def rowcol_to_lonlat(scene: dict, row: int, col: int) -> tuple[float, float]:
    """WGS-84 lon/lat of the CENTRE of pixel (row, col)."""
    from pyproj import Transformer
    x, y = xy(scene["transform"], row, col, offset="center")
    tf = Transformer.from_crs(scene["crs"], "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(x, y)
    return float(lon), float(lat)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clarity measurement at GPS reef
# ─────────────────────────────────────────────────────────────────────────────

def measure_clarity(scene: dict, lon: float, lat: float, win: int = 3) -> dict:
    """
    Measure real water clarity at (lon, lat) from the scene pixels.

    Only valid pixels of genuine bands are used. A field whose band is unavailable
    (or has no valid pixel in the window) is None — never 0 and never borrowed
    from another band.

    Returns dict with:
        nir_glint      : mean NIR reflectance in window (low = clean, high = glint) | None
        blue, green, red : mean BOA reflectance at the reef | None
        bottom_contrast: (blue - green) | None
        n_valid_px     : valid pixels per band in the window
        valid          : point inside raster AND blue+green have valid pixels there
    """
    bands = scene["bands"]
    h, w = scene.get("shape") or next(iter(bands.values())).shape
    row, col = lonlat_to_rowcol(scene, lon, lat)
    if not (0 <= row < h and 0 <= col < w):
        return {"valid": False}

    half = win // 2
    r0, r1 = max(0, row - half), min(h, row + half + 1)
    c0, c1 = max(0, col - half), min(w, col + half + 1)
    valid_masks = scene.get("valid", {})

    n_valid: dict[str, int] = {}

    def _m(b):
        if b not in bands:
            n_valid[b] = 0
            return None
        sub = bands[b][r0:r1, c0:c1]
        ok = np.isfinite(sub)
        if b in valid_masks:
            ok &= valid_masks[b][r0:r1, c0:c1]
        n_valid[b] = int(ok.sum())
        return float(sub[ok].mean()) if ok.any() else None

    blue, green, red, nir = _m("blue"), _m("green"), _m("red"), _m("nir")
    _r = lambda v: None if v is None else round(v, 5)  # noqa: E731
    return {
        "valid":           blue is not None and green is not None,
        "row":             int(row),
        "col":             int(col),
        "nir_glint":       _r(nir),
        "blue":            _r(blue),
        "green":           _r(green),
        "red":             _r(red),
        "bottom_contrast": None if blue is None or green is None else round(blue - green, 5),
        "n_valid_px":      n_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lyzenga depth-invariant index
# ─────────────────────────────────────────────────────────────────────────────

def load_depth_on_grid(scene: dict, emodnet_path: Path = EMODNET_DEPTH) -> np.ndarray | None:
    """
    Resample the EMODnet depth raster onto the scene's pixel grid.
    Returns (H,W) depth array (negative = below MSL), or None if unavailable.
    """
    if not emodnet_path.exists():
        log.warning("EMODnet depth not found at %s — depth masking disabled", emodnet_path)
        return None
    from rasterio.warp import reproject, Resampling
    h, w = scene.get("shape") or scene["bands"]["blue"].shape
    dst = np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.open(emodnet_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nd  = src.nodata if src.nodata is not None else -9999.0
        src_arr[src_arr == src_nd] = np.nan
        reproject(
            source=src_arr, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            dst_transform=scene["transform"], dst_crs=scene["crs"],
            resampling=Resampling.bilinear,   # depth is continuous; NaN excluded via src_nodata
        )
    return dst


def water_mask(bands: dict, depth: np.ndarray | None = None,
               nir_max: float = GLINT_NIR_MAX, valid: dict | None = None) -> np.ndarray:
    """
    Boolean mask of clean, optically-shallow water:
      - genuine NIR below the glint/land threshold (B08 is REQUIRED — no substitute)
      - positive blue/green
      - every input pixel valid (not nodata, finite)
      - depth within the optical window [DEPTH_DEEP, DEPTH_SHALLOW] if depth given

    Valid negative NIR (possible after the BOA offset over very dark water) is kept;
    only non-finite / nodata pixels are dropped.
    """
    missing = [b for b in ("nir", "blue", "green") if b not in bands]
    if missing:
        raise DetectorUnavailable(f"water_mask needs genuine bands {missing} (B08 gates land/glint masking)")
    nir, blue, green = bands["nir"], bands["blue"], bands["green"]
    m = np.isfinite(nir) & np.isfinite(blue) & np.isfinite(green)
    m &= (nir < nir_max) & (blue > 0) & (green > 0)
    if valid:
        for k in ("nir", "blue", "green"):
            if k in valid:
                m &= valid[k]
    if depth is not None:
        m = m & np.isfinite(depth) & (depth >= DEPTH_DEEP) & (depth <= DEPTH_SHALLOW)
    return m


def _attenuation_ratio(xi: np.ndarray, xj: np.ndarray) -> float:
    """
    Lyzenga (1978) automatic estimate of the attenuation-coefficient ratio
    k_i/k_j from the variance and covariance of the log-transformed bands:

        a       = (var(Xi) - var(Xj)) / (2·cov(Xi, Xj))
        k_i/k_j = a + sqrt(a² + 1)
    """
    vi = np.var(xi)
    vj = np.var(xj)
    cov = np.mean((xi - xi.mean()) * (xj - xj.mean()))
    if abs(cov) < 1e-9:
        return 1.0
    a = (vi - vj) / (2.0 * cov)
    return float(a + np.sqrt(a * a + 1.0))


def compute_dii(scene: dict, depth: np.ndarray | None = None) -> dict:
    """
    Compute the Lyzenga depth-invariant index DII over the scene.

    Returns dict with:
        dii         : (H,W) float32, NaN where not clean water
        mask        : water mask
        lsw         : deep-water background per band
        k_ratio     : estimated k_blue/k_green
    """
    bands = scene["bands"]
    valid = scene.get("valid")
    # Shallow zone where we actually classify substrate
    mask  = water_mask(bands, depth, valid=valid)
    # Full water (all depths) for deep-water background estimation
    full_water = water_mask(bands, depth=None, valid=valid)
    if not full_water.any():
        raise DetectorUnavailable("no clean-water pixels (genuine NIR below glint threshold) for Lsw estimation")

    blue, green = bands["blue"], bands["green"]

    # Deep-water background Lsw = 1st percentile over ALL clean water (darkest = deepest)
    lsw_blue  = float(np.percentile(blue[full_water],  1))
    lsw_green = float(np.percentile(green[full_water], 1))

    # Log transform; clip to avoid log of non-positive after background subtraction
    eps = 1e-6
    x_blue  = np.full(blue.shape,  np.nan, dtype=np.float64)
    x_green = np.full(green.shape, np.nan, dtype=np.float64)
    db = blue  - lsw_blue
    dg = green - lsw_green
    ok = mask & (db > eps) & (dg > eps)
    x_blue[ok]  = np.log(db[ok])
    x_green[ok] = np.log(dg[ok])

    if ok.sum() < MIN_DII_PIXELS:
        raise DetectorUnavailable(f"only {int(ok.sum())} usable shallow-water pixels (< {MIN_DII_PIXELS})")

    # Auto attenuation ratio from the joint distribution of clean-water pixels
    k_ratio = _attenuation_ratio(x_blue[ok], x_green[ok])

    dii = np.full(blue.shape, np.nan, dtype=np.float32)
    dii[ok] = (x_blue[ok] - k_ratio * x_green[ok]).astype(np.float32)

    log.info("  Lyzenga: Lsw_blue=%.5f Lsw_green=%.5f  k_blue/k_green=%.4f  water_px=%d",
             lsw_blue, lsw_green, k_ratio, int(ok.sum()))
    return {"dii": dii, "mask": ok, "lsw": {"blue": lsw_blue, "green": lsw_green},
            "k_ratio": k_ratio}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Texture (rugosity proxy)
# ─────────────────────────────────────────────────────────────────────────────

def local_stats(arr: np.ndarray, mask: np.ndarray, win: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """
    NaN-aware local mean and standard deviation of *arr* within a win×win window.
    Computed via box filters using the E[x²]-E[x]² identity, ignoring masked-out
    pixels. Returns (mean, std), both NaN where insufficient valid neighbours.
    """
    from scipy.ndimage import uniform_filter
    a = np.where(mask, arr, 0.0).astype(np.float64)
    valid = mask.astype(np.float64)
    s1  = uniform_filter(a,       size=win, mode="constant")
    s2  = uniform_filter(a * a,   size=win, mode="constant")
    cnt = uniform_filter(valid,   size=win, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var  = s2 / cnt - mean * mean
    var = np.clip(var, 0, None)
    std = np.sqrt(var)
    too_few = cnt < 0.15
    mean[too_few] = np.nan
    std[too_few]  = np.nan
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reef-likelihood combination
# ─────────────────────────────────────────────────────────────────────────────

def _robust_minmax(arr: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Scale to [0,1] using robust percentiles; NaN stays NaN."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo = np.percentile(finite, lo_pct)
    hi = np.percentile(finite, hi_pct)
    if hi - lo < 1e-9:
        # No contrast → the normalised feature is undefined, not "zero likelihood".
        return np.full(arr.shape, np.nan, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def reef_likelihood(scene: dict, dii_result: dict, bg_win: int = 31) -> dict:
    """
    Detect reef as a LOCAL DARK ANOMALY in the green bottom signal.

    The diagnostic showed the confirmed reef sits at the ~10th percentile of green
    reflectance within its own ~800 m neighbourhood — it is darker than the
    surrounding sand at comparable depth. A global threshold can't capture this
    because brightness varies with depth across the shelf; a LOCAL contrast does.

    For each shallow-water pixel:
        darkness_z = (local_mean_green - green) / local_std_green
    Positive z = darker than local surroundings = reef-like. The large background
    window (bg_win px) removes the slow depth/brightness gradient, isolating
    compact dark structures (rock/reef) from smooth sand.

    Structure gate: reef is also spatially rough, so we multiply by a normalised
    fine-scale texture term to suppress isolated dark speckles.

    Returns dict with likelihood map (0–1) and the anchor diagnostics.
    """
    bands = scene["bands"]
    green = bands["green"]
    mask  = dii_result["mask"]            # shallow, clean-water pixels
    h, w = green.shape

    # Local background statistics (large window removes depth gradient)
    bg_mean, bg_std = local_stats(green, mask, win=bg_win)

    # Dark-anomaly z-score: how many local sigmas darker than surroundings
    eps = 1e-6
    darkness_z = (bg_mean - green) / (bg_std + eps)
    darkness_z[~mask] = np.nan

    # Fine-scale structure (3-px std) — reef is rough, sand is smooth
    _, fine_std = local_stats(green, mask, win=3)
    tex_norm = _robust_minmax(fine_std)

    z_norm = _robust_minmax(darkness_z)

    # Combine: dark AND structured. Geometric mean penalises pixels weak in either.
    likelihood = np.sqrt(np.clip(z_norm, 0, 1) * np.clip(tex_norm, 0, 1))
    likelihood[~mask] = np.nan

    # Anchor diagnostics at the GPS reef
    anchor_z = []
    for site in REEF_SITES:
        row, col = lonlat_to_rowcol(scene, float(site["lon"]), float(site["lat"]))
        if 0 <= row < h and 0 <= col < w:
            win = darkness_z[max(0, row-1):row+2, max(0, col-1):col+2]
            v = float(np.nanmean(win)) if np.isfinite(win).any() else float("nan")
            if np.isfinite(v):
                anchor_z.append(float(v))
    anchor_darkness_z = float(np.mean(anchor_z)) if anchor_z else float("nan")

    log.info("  Local dark-anomaly detector  bg_win=%dpx (~%dm)  anchor darkness_z=%.2f",
             bg_win, bg_win * 10, anchor_darkness_z)
    return {
        "likelihood":   likelihood.astype(np.float32),
        "darkness_z":   darkness_z.astype(np.float32),
        "z_norm":       z_norm.astype(np.float32),
        "tex_norm":     tex_norm.astype(np.float32),
        "anchor_darkness_z": round(anchor_darkness_z, 3),
        "bg_win_px":    bg_win,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validation at GPS anchors
# ─────────────────────────────────────────────────────────────────────────────

def validate_at_gps(scene: dict, like_result: dict) -> dict:
    """
    Report the likelihood value AND its scene percentile at each GPS reef.
    A working detector puts the confirmed reef in a high percentile.
    """
    like = like_result["likelihood"]
    h, w = like.shape
    finite = like[np.isfinite(like)]

    results: list[dict] = []
    for site in REEF_SITES:
        row, col = lonlat_to_rowcol(scene, float(site["lon"]), float(site["lat"]))
        if not (0 <= row < h and 0 <= col < w):
            results.append({"site": str(site["name"]), "in_bbox": False})
            continue
        win = like[max(0, row-1):row+2, max(0, col-1):col+2]
        if not np.isfinite(win).any() or finite.size == 0:
            # Masked (land/glint/nodata/outside depth window): no value, not 0.
            results.append({"site": str(site["name"]), "in_bbox": True,
                            "likelihood": None, "percentile": None})
            continue
        val = float(np.nanmean(win))
        pct = float((finite < val).mean() * 100)
        results.append({
            "site":       str(site["name"]),
            "in_bbox":    True,
            "likelihood": round(val, 4),
            "percentile": round(pct, 1),
        })
        log.info("  Anchor %s: likelihood=%.3f (%.1fth percentile of scene)",
                 site["name"], val, pct)
    return {"anchors": results}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output (GeoTIFF + overlay PNG)
# ─────────────────────────────────────────────────────────────────────────────

def write_geotiff(arr: np.ndarray, scene: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing raster {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 1,
        "dtype": "float32", "crs": scene["crs"], "transform": scene["transform"],
        "nodata": -9999.0, "compress": "deflate",
    }
    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    log.info("  Wrote %s", path)


def write_overlay_png(scene: dict, like: np.ndarray, path: Path) -> None:
    """RGB true-colour with reef-likelihood contours/heat overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable — skipping overlay PNG")
        return

    bands = scene["bands"]
    if not all(k in bands for k in ("red", "green", "blue")):
        # True colour needs genuine B04/B03/B02 — do not fake a red channel.
        log.warning("  true-colour overlay skipped: %s unavailable",
                    [k for k in ("red", "green", "blue") if k not in bands])
        return
    rgb = np.dstack([bands["red"], bands["green"], bands["blue"]])
    rgb = np.clip(rgb / np.nanpercentile(rgb, 99), 0, 1)   # stretch
    rgb = np.where(np.isfinite(rgb), rgb, 0.0)              # display only: invalid shown black

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{scene['meta']['datetime']} — True colour (B04/B03/B02)")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    hm = axes[1].imshow(np.ma.masked_invalid(like), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("Reef likelihood (GPS-anchored)")
    axes[1].axis("off")

    # Mark GPS reef sites
    h, w = like.shape
    for site in REEF_SITES:
        row, col = lonlat_to_rowcol(scene, float(site["lon"]), float(site["lat"]))
        if 0 <= row < h and 0 <= col < w:
            for ax in axes:
                ax.plot(col, row, "co", ms=10, mfc="none", mew=2)
            axes[1].annotate(str(site["name"]).replace("_", " "), (col, row),
                             color="cyan", fontsize=8, xytext=(col+5, row-5))

    fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04, label="likelihood")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Wrote %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def _unavailable(scene: dict | None, reason: str, clarity: dict | None = None,
                 availability: dict | None = None, meta: dict | None = None) -> dict:
    """Explicit 'no result' record — never a zero likelihood."""
    return {
        "status":       "unavailable",
        "reason":       reason,
        "scene":        (scene or {}).get("meta", meta or {}),
        "band_status":  (scene or {}).get("band_status"),
        "radiometry":   (scene or {}).get("radiometry"),
        "availability": availability,
        "clarity":      clarity or {},
        "validation":   None,
        "outputs":      {},
    }


def process_scene_loaded(scene: dict, out_dir: Path, depth: np.ndarray | None = None) -> dict:
    """Detector on an already-loaded scene (testable offline)."""
    av = band_registry.evaluate(scene["usable_bands"])
    # Clarity fields are computed per band; unavailable ones are None.
    clarity = {}
    for site in REEF_SITES:
        c = measure_clarity(scene, float(site["lon"]), float(site["lat"]))
        clarity[str(site["name"])] = c
        if c.get("valid"):
            log.info("  Clarity @ %s: NIR_glint=%s  blue=%s green=%s red=%s  bottom_contrast=%s",
                     site["name"], c["nir_glint"], c["blue"], c["green"], c["red"], c["bottom_contrast"])

    gating = av.gating_failures()
    if gating:
        reason = "gated features unavailable: " + ", ".join(
            f"{g} (missing {av.missing_bands.get(g) or av.blocked_by.get(g)})" for g in gating)
        log.warning("  UNAVAILABLE — %s", reason)
        return _unavailable(scene, reason, clarity, av.as_dict())

    try:
        dii_result  = compute_dii(scene, depth)
        like_result = reef_likelihood(scene, dii_result)
    except DetectorUnavailable as exc:
        log.warning("  UNAVAILABLE — %s", exc)
        return _unavailable(scene, str(exc), clarity, av.as_dict())
    validation  = validate_at_gps(scene, like_result)

    date = scene["meta"]["datetime"]
    outputs = {
        "likelihood": str(out_dir / f"reef_likelihood_{date}.tif"),
        "dii":        str(out_dir / f"dii_{date}.tif"),
        "overlay":    str(out_dir / f"overlay_{date}.png"),
    }
    write_geotiff(like_result["likelihood"], scene, Path(outputs["likelihood"]))
    write_geotiff(dii_result["dii"],          scene, Path(outputs["dii"]))
    write_overlay_png(scene, like_result["likelihood"], Path(outputs["overlay"]))

    return {
        "status":     "ok",
        "scene":      scene["meta"],
        "band_status": scene["band_status"],
        "radiometry": scene["radiometry"],
        "availability": av.as_dict(),
        "clarity":    clarity,
        "k_ratio":    round(dii_result["k_ratio"], 4),
        "anchor_darkness_z": like_result["anchor_darkness_z"],
        "bg_win_px":  like_result["bg_win_px"],
        "validation": validation,
        "outputs":    outputs,
        "caveat":     "software output only — not a scientifically validated reef map",
    }


def process_scene(scene_id: str, bbox: tuple, out_dir: Path) -> dict:
    """Full pipeline for one scene → clarity + likelihood map + validation, or status=unavailable."""
    try:
        scene = load_scene(scene_id, bbox)
    except DetectorUnavailable as exc:
        return _unavailable(None, str(exc), meta={"id": scene_id})
    depth = load_depth_on_grid(scene)
    if depth is not None:
        shallow_px = int(((depth >= DEPTH_DEEP) & (depth <= DEPTH_SHALLOW)).sum())
        log.info("  Depth grid loaded; %d px in optical window [%.0f, %.0f m]",
                 shallow_px, DEPTH_DEEP, DEPTH_SHALLOW)
    return process_scene_loaded(scene, out_dir, depth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reef detector — Lyzenga DII + texture")
    parser.add_argument("--scene", action="append", required=True,
                        help="Sentinel-2 scene ID (repeat for multiple)")
    parser.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX),
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    # One fresh directory per run: never overwrite earlier/legacy outputs.
    run_dir = OUTPUT_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    reports = []
    for scene_id in args.scene:
        print(f"\n{'='*70}\nPROCESSING {scene_id}\n{'='*70}")
        try:
            reports.append(process_scene(scene_id, bbox, run_dir))
        except Exception as exc:
            log.error("Failed on %s: %s", scene_id, exc, exc_info=True)
            reports.append(_unavailable(None, f"error: {exc}", meta={"id": scene_id}))

    print(f"\n{'='*70}\nCLARITY COMPARISON (real pixels at GPS reef)\n{'='*70}")
    for r in reports:
        c = r["clarity"].get("Pedra_Sta_Eulalia", {})
        if c.get("valid"):
            print(f"  {r['scene'].get('datetime')}  ({r['scene'].get('platform')}):  "
                  f"NIR_glint={c['nir_glint']}  bottom_contrast={c['bottom_contrast']}  "
                  f"cloud={r['scene'].get('cloud_cover')}%")
    print("\n  Lower NIR_glint = cleaner water surface (less sun glint).")
    print("  Higher bottom_contrast = stronger seafloor signal reaching the sensor.")

    print(f"\n{'='*70}\nDETECTOR STATUS / VALIDATION AT GPS REEF ANCHORS\n{'='*70}")
    for r in reports:
        print(f"  {r['scene'].get('datetime') or r['scene'].get('id')}: status={r['status']}")
        if r["status"] != "ok":
            print(f"    reason: {r['reason']}")
            continue
        for a in r["validation"]["anchors"]:
            if a.get("in_bbox"):
                lk = "n/a (masked)" if a["likelihood"] is None else f"{a['likelihood']:.3f}"
                pc = "" if a["percentile"] is None else f"  ({a['percentile']:.1f}th percentile)"
                print(f"    {a['site']:22s} likelihood={lk}{pc}")

    report_path = run_dir / "detector_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nReport: {report_path}")
    print("NOTE: software output only — the detector is not scientifically validated.")


if __name__ == "__main__":
    main()
