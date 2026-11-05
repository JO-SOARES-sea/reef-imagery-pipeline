#!/usr/bin/env python3
"""
multiband_reef_analysis.py
===========================
Multi-band satellite-derived bathymetry (SDB) and reef detection using
all downloaded Sentinel-2 bands (B01–B12).

Three bathymetry methods compared:
  1. Classic Stumpf 2-band log-ratio (B02/B03)
  2. Lyzenga 6-band linear regression (B01, B02, B03, B04, B05, B8A)
  3. PCA-based depth (PC1 of water-penetrating bands)

Enhanced indices:
  - NDWI  = (B03 − B08) / (B03 + B08)
  - NDI   = (B03 − B05) / (B03 + B05)
  - Depth-to-substrate ratio = B02 / B05
  - Water clarity index = B02 / B8A

Outputs:
  - stumpf_depth.tif, lyzenga_depth.tif, pca_depth.tif
  - ndwi.tif, ndi.tif, clarity_index.tif
  - reef_candidates_multiband.geojson
  - multiband_comparison.png
  - analysis_summary.json

CLI:
  python multiband_reef_analysis.py \\
      --input-dir sentinel_images/S2A_MSIL2A_20220926 \\
      --output-dir outputs/multiband \\
      --lat 37.069 --lon -8.210
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from scipy.ndimage import uniform_filter, label as ndimage_label

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from shapely.geometry import shape
    from shapely.ops import unary_union
    import geopandas as gpd
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

WATER_PENETRATING = ["B01", "B02", "B03", "B04", "B05"]

LYZENGA_BANDS = ["B01", "B02", "B03", "B04", "B05", "B8A"]

STUMPF_N = 1000.0
STUMPF_M0 = -16.0
STUMPF_M1 = 20.0

REFLECTANCE_THRESHOLD = 2.0

log = logging.getLogger("multiband_reef")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_write_tif(out_path, profile, arr):
    if os.path.exists(out_path):
        fd, tmp = tempfile.mkstemp(suffix=".tif", dir=tempfile.gettempdir())
        try:
            os.close(fd)
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(arr)
            shutil.copy2(tmp, out_path)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    else:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)


def _read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def _normalize_reflectance(arr):
    if np.nanmax(arr) > REFLECTANCE_THRESHOLD:
        arr = arr / 10000.0
    return arr


def _depth_stats(depth_arr):
    valid = np.isfinite(depth_arr)
    n_valid = int(valid.sum())
    total = depth_arr.size
    if n_valid == 0:
        return {"mean": None, "std": None, "valid_pct": 0.0,
                "min": None, "max": None, "n_valid": 0}
    vals = depth_arr[valid]
    return {
        "mean": round(float(np.mean(vals)), 3),
        "std": round(float(np.std(vals)), 3),
        "valid_pct": round(100.0 * n_valid / total, 2),
        "min": round(float(np.min(vals)), 3),
        "max": round(float(np.max(vals)), 3),
        "n_valid": n_valid,
    }


# ---------------------------------------------------------------------------
# Band loading
# ---------------------------------------------------------------------------

def load_bands(input_dir):
    """Load all available Sentinel-2 band GeoTIFFs from input_dir."""
    bands = {}
    profile = None
    import glob as _glob
    for name in ALL_BANDS:
        candidates = [
            os.path.join(input_dir, f"{name}.tif"),
            os.path.join(input_dir, f"{name}_10m.tif"),
            os.path.join(input_dir, f"{name}_20m.tif"),
            os.path.join(input_dir, f"{name}_60m.tif"),
        ]
        s2_matches = sorted(_glob.glob(os.path.join(input_dir, f"S2_{name}_*.tif")))
        candidates = s2_matches + candidates
        for path in candidates:
            if os.path.exists(path):
                arr, prof = _read_band(path)
                arr = _normalize_reflectance(arr)
                bands[name] = arr
                if profile is None:
                    profile = prof
                log.info("Loaded %s from %s (shape=%s)", name, path, arr.shape)
                break
        else:
            log.debug("Band %s not found in %s", name, input_dir)

    if not bands:
        log.error("No band files found in %s", input_dir)
        sys.exit(1)

    log.info("Loaded %d bands: %s", len(bands), sorted(bands.keys()))
    return bands, profile


def _get_common_shape(bands):
    shapes = set(b.shape for b in bands.values())
    if len(shapes) == 1:
        return list(shapes)[0]
    from collections import Counter
    counts = Counter(b.shape for b in bands.values())
    return counts.most_common(1)[0][0]


def _resize_to(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    from scipy.ndimage import zoom
    factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, factors, order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Method 1: Stumpf 2-band
# ---------------------------------------------------------------------------

def compute_stumpf_depth(bands, n=STUMPF_N, m0=STUMPF_M0, m1=STUMPF_M1):
    """Classic Stumpf log-ratio: depth = m1 * ln(n*B02) / ln(n*B03) + m0."""
    if "B02" not in bands or "B03" not in bands:
        log.warning("Stumpf requires B02 and B03 — skipping")
        return None

    b02 = bands["B02"]
    b03 = bands["B03"]
    valid = (b02 > 0) & (b03 > 0)
    b02_s = np.where(valid, b02, np.nan)
    b03_s = np.where(valid, b03, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_b = np.log(n * b02_s)
        log_g = np.log(n * b03_s)
        safe = (log_g > 0) & np.isfinite(log_b) & np.isfinite(log_g)
        ratio = np.where(safe, log_b / log_g, np.nan)
        depth = np.where(safe, m1 * ratio + m0, np.nan)

    depth = -np.abs(depth)
    depth = np.where((depth > 0) | (depth < -60), np.nan, depth)

    stats = _depth_stats(depth)
    log.info("Stumpf depth: mean=%.1f m, valid=%.0f%%, range=[%.1f, %.1f]",
             stats["mean"] or 0, stats["valid_pct"],
             stats["min"] or 0, stats["max"] or 0)
    return depth, stats


# ---------------------------------------------------------------------------
# Method 2: Lyzenga multi-band
# ---------------------------------------------------------------------------

def compute_lyzenga_depth(bands, calibration_points=None):
    """
    Lyzenga linear regression: depth = c0 + c1*X1 + ... + c5*X5
    where Xi = ln(Bi) for bands B01, B02, B03, B04, B05, B8A.

    If calibration_points is None, uses a synthetic calibration derived
    from the Stumpf depth as a reference (bootstrapping).
    """
    available = [b for b in LYZENGA_BANDS if b in bands]
    if len(available) < 3:
        log.warning("Lyzenga needs at least 3 of %s — got %s, skipping",
                    LYZENGA_BANDS, available)
        return None

    shape = _get_common_shape(bands)
    log_bands = {}
    for name in available:
        arr = _resize_to(bands[name], shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.log(np.where(arr > 0, arr, np.nan))
        log_bands[name] = lb

    valid_mask = np.ones(shape, dtype=bool)
    for name in available:
        valid_mask &= np.isfinite(log_bands[name]) & (bands.get(name, np.zeros(shape)) > 0)

    if calibration_points is None:
        stumpf_result = compute_stumpf_depth(bands)
        if stumpf_result is None:
            log.warning("No Stumpf depth for Lyzenga bootstrap — using uniform defaults")
            stumpf_depth = np.full(shape, -10.0, dtype=np.float32)
        else:
            stumpf_depth = stumpf_result[0]

        sample_mask = valid_mask & np.isfinite(stumpf_depth)
        n_samples = min(5000, sample_mask.sum())
        if n_samples < 10:
            log.warning("Too few valid pixels for Lyzenga calibration (%d)", n_samples)
            return None

        idxs = np.where(sample_mask)
        chosen = np.random.choice(len(idxs[0]), n_samples, replace=False)
        rows = idxs[0][chosen]
        cols = idxs[1][chosen]

        y = stumpf_depth[rows, cols]
        X = np.column_stack([log_bands[name][rows, cols] for name in available])
        X = np.column_stack([np.ones(n_samples), X])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            log.warning("Lyzenga least-squares failed")
            return None

        X_full = np.column_stack([log_bands[name] for name in available])
        ones = np.ones((shape[0] * shape[1], 1), dtype=np.float32)
        X_all = np.hstack([ones, X_full.reshape(-1, len(available))])
        depth_flat = X_all @ coeffs
        depth = depth_flat.reshape(shape)
    else:
        log.info("Using %d calibration points for Lyzenga", len(calibration_points))
        y = np.array([p["depth"] for p in calibration_points])
        X = np.array([[np.log(max(bands[name][p["row"], p["col"]], 1e-10))
                        for name in available]
                       for p in calibration_points])
        X = np.column_stack([np.ones(len(y)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        X_full = np.column_stack([log_bands[name] for name in available])
        ones = np.ones((shape[0] * shape[1], 1), dtype=np.float32)
        X_all = np.hstack([ones, X_full.reshape(-1, len(available))])
        depth = (X_all @ coeffs).reshape(shape)

    depth = np.where(valid_mask, depth, np.nan)
    depth = -np.abs(depth)
    depth = np.where((depth > 0) | (depth < -60), np.nan, depth)

    stats = _depth_stats(depth)
    log.info("Lyzenga depth: mean=%.1f m, valid=%.0f%%, range=[%.1f, %.1f]",
             stats["mean"] or 0, stats["valid_pct"],
             stats["min"] or 0, stats["max"] or 0)
    return depth, stats


# ---------------------------------------------------------------------------
# Method 3: PCA-based depth
# ---------------------------------------------------------------------------

def compute_pca_depth(bands):
    """
    PCA-based depth: stack water-penetrating bands (B01–B05),
    take PC1 as depth proxy, scale via Stumpf reference.
    """
    if not HAS_SKLEARN:
        log.warning("sklearn not available — cannot run PCA method")
        return None

    available = [b for b in WATER_PENETRATING if b in bands]
    if len(available) < 3:
        log.warning("PCA needs at least 3 of %s — got %s, skipping",
                    WATER_PENETRATING, available)
        return None

    shape = _get_common_shape(bands)
    stack = []
    for name in available:
        arr = _resize_to(bands[name], shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.log(np.where(arr > 0, arr, np.nan))
        stack.append(la)

    stack = np.stack(stack, axis=-1)
    valid_mask = np.all(np.isfinite(stack), axis=-1) & np.all(
        np.array([_resize_to(bands[n], shape) for n in available]) > 0, axis=0
    )

    n_valid = valid_mask.sum()
    if n_valid < 50:
        log.warning("Too few valid pixels for PCA (%d)", n_valid)
        return None

    pixels = stack[valid_mask]
    n_components = min(2, len(available))
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(pixels)

    pc1 = np.full(shape, np.nan, dtype=np.float32)
    pc1[valid_mask] = components[:, 0]

    stumpf_result = compute_stumpf_depth(bands)
    if stumpf_result is not None:
        stumpf_depth = stumpf_result[0]
        ref_mask = valid_mask & np.isfinite(stumpf_depth)
        if ref_mask.sum() > 10:
            pc_vals = pc1[ref_mask]
            d_vals = stumpf_depth[ref_mask]
            slope, intercept = np.polyfit(pc_vals, d_vals, 1)
            depth = slope * pc1 + intercept
        else:
            median_pc = float(np.nanmedian(pc1))
            std_pc = float(np.nanstd(pc1))
            depth = -20.0 * (pc1 - median_pc) / max(std_pc, 1e-6)
    else:
        median_pc = float(np.nanmedian(pc1))
        std_pc = float(np.nanstd(pc1))
        depth = -20.0 * (pc1 - median_pc) / max(std_pc, 1e-6)

    depth = np.where(valid_mask, depth, np.nan)
    depth = -np.abs(depth)
    depth = np.where((depth > 0) | (depth < -60), np.nan, depth)

    stats = _depth_stats(depth)
    stats["pca_explained_variance"] = [
        round(float(v), 4) for v in pca.explained_variance_ratio_
    ]
    log.info("PCA depth: mean=%.1f m, valid=%.0f%%, explained_var=%s",
             stats["mean"] or 0, stats["valid_pct"],
             stats["pca_explained_variance"])
    return depth, stats


# ---------------------------------------------------------------------------
# Enhanced indices
# ---------------------------------------------------------------------------

def compute_indices(bands):
    """Compute NDWI, NDI, depth-to-substrate ratio, water clarity index."""
    shape = _get_common_shape(bands)
    indices = {}

    def _get(name):
        if name in bands:
            return _resize_to(bands[name], shape)
        return None

    b02 = _get("B02")
    b03 = _get("B03")
    b05 = _get("B05")
    b08 = _get("B08")
    b8a = _get("B8A")

    if b03 is not None and b08 is not None:
        denom = b03 + b08
        ndwi = np.where(denom > 0, (b03 - b08) / denom, np.nan)
        indices["ndwi"] = ndwi
        log.info("NDWI: mean=%.3f", float(np.nanmean(ndwi)))

    if b03 is not None and b05 is not None:
        denom = b03 + b05
        ndi = np.where(denom > 0, (b03 - b05) / denom, np.nan)
        indices["ndi"] = ndi
        log.info("NDI: mean=%.3f", float(np.nanmean(ndi)))

    if b02 is not None and b05 is not None:
        dts = np.where(b05 > 0, b02 / b05, np.nan)
        indices["depth_substrate_ratio"] = dts
        log.info("Depth-to-substrate ratio: mean=%.3f", float(np.nanmean(dts)))

    if b02 is not None and b8a is not None:
        clarity = np.where(b8a > 0, b02 / b8a, np.nan)
        indices["clarity_index"] = clarity
        log.info("Clarity index: mean=%.3f", float(np.nanmean(clarity)))

    return indices


# ---------------------------------------------------------------------------
# Reef candidate detection (same algorithm as reef_bathy_module)
# ---------------------------------------------------------------------------

def detect_reef_candidates(depth, profile, output_dir, lat, lon,
                           depth_min=-50.0, depth_max=-1.0,
                           tri_pct=70.0, bpi_pct=60.0, min_area_px=4):
    """Detect reef candidates from depth map using TRI + BPI + depth mask."""
    if not HAS_SHAPELY:
        log.error("Reef detection requires shapely + geopandas")
        return None

    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    if valid.sum() == 0:
        log.warning("No pixels in depth range [%.0f, %.0f]", depth_min, depth_max)
        return None

    dem = np.where(np.isfinite(depth), depth, 0.0)
    transform = profile.get("transform")
    crs = profile.get("crs")

    pad = np.pad(dem, 1, mode="edge")
    centre = pad[1:-1, 1:-1]
    sq_sum = np.zeros_like(dem)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            nbr = pad[dr:dr + dem.shape[0], dc:dc + dem.shape[1]]
            sq_sum += (nbr - centre) ** 2
    tri = np.sqrt(sq_sum / 8.0)
    tri = np.where(np.isfinite(depth), tri, np.nan)

    mean_broad = uniform_filter(dem, size=15, mode="nearest")
    bpi = depth - mean_broad
    bpi = np.where(np.isfinite(depth), bpi, np.nan)

    tri_valid = tri[valid & np.isfinite(tri)]
    bpi_valid = bpi[valid & np.isfinite(bpi)]

    if len(tri_valid) == 0 or len(bpi_valid) == 0:
        log.warning("No valid TRI/BPI values in depth window")
        return None

    tri_thresh = np.percentile(tri_valid, tri_pct)
    bpi_thresh = np.percentile(bpi_valid, bpi_pct)

    rugose = np.isfinite(tri) & (tri >= tri_thresh)
    elevated = np.isfinite(bpi) & (bpi >= bpi_thresh)
    candidate_mask = valid & rugose & elevated

    log.info("Reef candidates: %d / %d depth pixels",
             int(candidate_mask.sum()), int(valid.sum()))

    if candidate_mask.sum() == 0:
        log.warning("No reef candidates found")
        return _empty_geojson(output_dir), 0

    labelled, n_feat = ndimage_label(candidate_mask)
    sizes = np.bincount(labelled.ravel())
    small = np.where(sizes < min_area_px)[0]
    remove = np.isin(labelled, small)
    filtered = np.where(remove, 0, candidate_mask.astype(np.uint8))

    geoms = []
    for geom_dict, val in rasterio_shapes(
        filtered.astype(np.uint8), mask=filtered, transform=transform
    ):
        if val == 1:
            geoms.append(shape(geom_dict))

    if not geoms:
        log.warning("No polygons after filtering")
        return _empty_geojson(output_dir), int(candidate_mask.sum())

    merged = [unary_union(geoms)] if len(geoms) > 50 else geoms
    epsg = crs.to_epsg() if crs else 4326
    is_projected = epsg and epsg != 4326

    def _calc_area_m2(g):
        if is_projected:
            return g.area
        return g.area * (111_320.0 ** 2)

    gdf = gpd.GeoDataFrame(
        {
            "method": ["multiband_pca"] * len(merged),
            "depth_min_m": [depth_min] * len(merged),
            "depth_max_m": [depth_max] * len(merged),
            "lat": [lat] * len(merged),
            "lon": [lon] * len(merged),
            "area_m2": [_calc_area_m2(g) for g in merged],
        },
        geometry=merged,
        crs=epsg,
    )

    out_path = os.path.join(output_dir, "reef_candidates_multiband.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    log.info("Reef candidates → %s (%d polygons, total area=%.0f m²)",
             out_path, len(gdf), gdf["area_m2"].sum())
    return out_path, int(candidate_mask.sum())


def _empty_geojson(output_dir):
    path = os.path.join(output_dir, "reef_candidates_multiband.geojson")
    fc = {"type": "FeatureCollection", "features": []}
    with open(path, "w") as f:
        json.dump(fc, f)
    return path


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def generate_comparison_figure(stumpf, lyzenga, pca, indices, profile, output_dir):
    """Generate 2x2 subplot comparison figure."""
    if not HAS_MPL:
        log.warning("matplotlib not available — skipping figure")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    vmin, vmax = -30, 0

    ax = axes[0, 0]
    im = ax.imshow(stumpf, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("Stumpf 2-band Depth (m)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    ax = axes[0, 1]
    im = ax.imshow(lyzenga, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("Lyzenga 6-band Depth (m)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    ax = axes[1, 0]
    im = ax.imshow(pca, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("PCA Depth (m)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    ax = axes[1, 1]
    ndwi = indices.get("ndwi")
    ndi = indices.get("ndi")
    depth_ref = stumpf
    if ndwi is not None and ndi is not None:
        r = np.where(np.isfinite(depth_ref),
                     np.clip((depth_ref - vmin) / (vmax - vmin), 0, 1), 0)
        g = np.where(np.isfinite(ndi),
                     np.clip((ndi + 1) / 2, 0, 1), 0)
        b_ch = np.where(np.isfinite(ndwi),
                        np.clip((ndwi + 1) / 2, 0, 1), 0)
        rgb = np.stack([r, g, b_ch], axis=-1)
        ax.imshow(rgb)
        ax.set_title("Enhanced RGB (R=depth, G=NDI, B=NDWI)")
    else:
        ax.text(0.5, 0.5, "Insufficient indices", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Enhanced RGB (unavailable)")

    for ax in axes.ravel():
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.suptitle("Multi-band SDB Comparison — Sentinel-2", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(output_dir, "multiband_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Comparison figure → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Write output GeoTIFFs
# ---------------------------------------------------------------------------

def write_depth_tif(arr, profile, path):
    prof = profile.copy()
    prof.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
    _safe_write_tif(path, prof, arr.astype(np.float32)[np.newaxis, ...])
    log.info("  → %s", path)


def write_index_tif(arr, profile, path):
    prof = profile.copy()
    prof.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
    _safe_write_tif(path, prof, arr.astype(np.float32)[np.newaxis, ...])
    log.info("  → %s", path)


# ---------------------------------------------------------------------------
# Select best depth method
# ---------------------------------------------------------------------------

def select_best_method(stumpf_stats, lyzenga_stats, pca_stats):
    """Select the method with the highest valid_pct and lowest std."""
    candidates = []
    for name, stats in [("stumpf", stumpf_stats),
                        ("lyzenga", lyzenga_stats),
                        ("pca", pca_stats)]:
        if stats and stats.get("valid_pct", 0) > 0:
            score = stats["valid_pct"] - 2.0 * stats.get("std", 999)
            candidates.append((name, score, stats))

    if not candidates:
        return "stumpf"

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    log.info("Best method: %s (score=%.1f)", best[0], best[1])
    return best[0]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(input_dir, output_dir, lat, lon, depth_min=-50.0, depth_max=-1.0):
    os.makedirs(output_dir, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    log_path = os.path.join(output_dir, "multiband_analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    log.info("=" * 60)
    log.info("MULTIBAND REEF ANALYSIS")
    log.info("  Input:  %s", input_dir)
    log.info("  Output: %s", output_dir)
    log.info("  Coords: lat=%.5f, lon=%.5f", lat, lon)
    log.info("=" * 60)

    bands, profile = load_bands(input_dir)

    stumpf_depth = stumpf_stats = None
    lyzenga_depth = lyzenga_stats = None
    pca_depth = pca_stats = None

    log.info("-" * 40)
    log.info("Method 1: Stumpf 2-band")
    result = compute_stumpf_depth(bands)
    if result:
        stumpf_depth, stumpf_stats = result
        write_depth_tif(stumpf_depth, profile,
                        os.path.join(output_dir, "stumpf_depth.tif"))

    log.info("-" * 40)
    log.info("Method 2: Lyzenga 6-band")
    result = compute_lyzenga_depth(bands)
    if result:
        lyzenga_depth, lyzenga_stats = result
        write_depth_tif(lyzenga_depth, profile,
                        os.path.join(output_dir, "lyzenga_depth.tif"))

    log.info("-" * 40)
    log.info("Method 3: PCA-based depth")
    if not HAS_SKLEARN:
        log.warning("scikit-learn not installed — PCA method unavailable")
    else:
        result = compute_pca_depth(bands)
        if result:
            pca_depth, pca_stats = result
            write_depth_tif(pca_depth, profile,
                            os.path.join(output_dir, "pca_depth.tif"))

    log.info("-" * 40)
    log.info("Computing enhanced indices")
    indices = compute_indices(bands)

    if "ndwi" in indices:
        write_index_tif(indices["ndwi"], profile,
                        os.path.join(output_dir, "ndwi.tif"))
    if "ndi" in indices:
        write_index_tif(indices["ndi"], profile,
                        os.path.join(output_dir, "ndi.tif"))
    if "clarity_index" in indices:
        write_index_tif(indices["clarity_index"], profile,
                        os.path.join(output_dir, "clarity_index.tif"))

    log.info("-" * 40)
    log.info("Generating comparison figure")
    ref_depth = stumpf_depth if stumpf_depth is not None else (lyzenga_depth if lyzenga_depth is not None else pca_depth)
    if ref_depth is not None:
        generate_comparison_figure(
            stumpf_depth if stumpf_depth is not None else ref_depth,
            lyzenga_depth if lyzenga_depth is not None else ref_depth,
            pca_depth if pca_depth is not None else ref_depth,
            indices, profile, output_dir,
        )

    log.info("-" * 40)
    log.info("Detecting reef candidates")
    best_name = select_best_method(stumpf_stats, lyzenga_stats, pca_stats)
    best_depth = {"stumpf": stumpf_depth,
                  "lyzenga": lyzenga_depth,
                  "pca": pca_depth}.get(best_name)

    reef_path = None
    reef_candidate_pixels = 0
    if best_depth is not None:
        result = detect_reef_candidates(
            best_depth, profile, output_dir, lat, lon,
            depth_min=depth_min, depth_max=depth_max,
        )
        if result is not None:
            reef_path, reef_candidate_pixels = result

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "lat": lat,
        "lon": lon,
        "bands_loaded": sorted(bands.keys()),
        "methods": {
            "stumpf_2band": stumpf_stats,
            "lyzenga_6band": lyzenga_stats,
            "pca_based": pca_stats,
        },
        "best_method": best_name,
        "indices": {
            name: _depth_stats(arr) if arr is not None else None
            for name, arr in indices.items()
        },
        "reef_candidates_geojson": reef_path,
        "reef_candidate_pixels": reef_candidate_pixels,
    }

    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Summary → %s", summary_path)

    _print_comparison_table(stumpf_stats, lyzenga_stats, pca_stats)

    return summary


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(stumpf_s, lyzenga_s, pca_s):
    print("\n" + "=" * 78)
    print("MULTIBAND SDB METHOD COMPARISON")
    print("=" * 78)
    header = f"{'Metric':<25} {'Stumpf 2-band':>15} {'Lyzenga 6-band':>15} {'PCA-based':>15}"
    print(header)
    print("-" * 78)

    metrics = [
        ("Mean depth (m)", "mean"),
        ("Std deviation (m)", "std"),
        ("Valid pixels (%)", "valid_pct"),
        ("Min depth (m)", "min"),
        ("Max depth (m)", "max"),
    ]

    for label, key in metrics:
        vals = []
        for s in [stumpf_s, lyzenga_s, pca_s]:
            v = s.get(key) if s else None
            vals.append(f"{v:>15.2f}" if v is not None else f"{'N/A':>15}")
        print(f"{label:<25} {vals[0]} {vals[1]} {vals[2]}")

    if pca_s and "pca_explained_variance" in pca_s:
        ev = pca_s["pca_explained_variance"]
        print(f"\n  PCA explained variance: {ev}")

    best = select_best_method(stumpf_s, lyzenga_s, pca_s)
    print(f"\n  Best method (lowest noise): {best}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Multi-band Sentinel-2 SDB and reef detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory with downloaded band GeoTIFFs (B01.tif … B12.tif)")
    p.add_argument("--output-dir", default="outputs/multiband_analysis",
                   help="Output directory for results")
    p.add_argument("--lat", type=float, required=True,
                   help="Latitude of AOI centre")
    p.add_argument("--lon", type=float, required=True,
                   help="Longitude of AOI centre")
    p.add_argument("--depth-min", type=float, default=-50.0,
                   help="Minimum depth for reef detection (negative)")
    p.add_argument("--depth-max", type=float, default=-1.0,
                   help="Maximum depth for reef detection (negative)")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_analysis(args.input_dir, args.output_dir, args.lat, args.lon,
                 depth_min=args.depth_min, depth_max=args.depth_max)
