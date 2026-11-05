#!/usr/bin/env python3
"""
calibrate_multiband_emodnet.py
===============================
Calibrate Stumpf 2-band, Lyzenga 6-band, and PCA depth methods
against EMODnet bathymetry for the Santa Eulalia reef site.

Steps:
  1. Read EMODnet bathymetry and reproject to S2 grid (10m, EPSG:32629)
  2. Read all S2 bands and sample EMODnet depth at S2 pixel locations
  3. Calibrate Stumpf (m0, m1) via scipy.optimize.minimize RMSE
  4. Calibrate Lyzenga via OLS regression against EMODnet
  5. PCA on water-penetrating bands, correlate PC1 with EMODnet
  6. Produce calibrated depth maps for all 3 methods
  7. Detect reef candidates from best method (depth -18 to -6)
  8. Validate reef candidates
  9. Generate 4-panel comparison figure
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes as rasterio_shapes
from scipy.ndimage import uniform_filter, label as ndimage_label
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATHY_PATH = "outputs/santa_eulalia_12m/bathy_emodnet_20260529.tif"
BAND_DIR = "outputs/santa_eulalia_multiband"
OUT_DIR = "outputs/santa_eulalia_multiband_calibrated"

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B11", "B12"]
WATER_PENETRATING = ["B01", "B02", "B03", "B04", "B05"]
LYZENGA_BANDS = ["B01", "B02", "B03", "B04", "B05", "B8A"]

REEF_DEPTH_MIN = -18.0
REEF_DEPTH_MAX = -6.0

REFLECTANCE_THRESHOLD = 2.0
STUMPF_N = 1000.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("calibrate_emodnet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def write_tif(arr, profile, path):
    prof = profile.copy()
    prof.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32)[np.newaxis, ...])


def normalize_reflectance(arr):
    if np.nanmax(arr) > REFLECTANCE_THRESHOLD:
        arr = arr / 10000.0
    return arr


def load_bands():
    import glob as _glob
    bands = {}
    profile = None
    for name in BAND_NAMES:
        s2_matches = sorted(_glob.glob(os.path.join(BAND_DIR, f"S2_{name}_*.tif")))
        candidates = s2_matches + [os.path.join(BAND_DIR, f"{name}.tif")]
        for path in candidates:
            if os.path.exists(path):
                arr, prof = read_raster(path)
                arr = normalize_reflectance(arr)
                bands[name] = arr
                if profile is None:
                    profile = prof
                log.info("Loaded %s: shape=%s, range=[%.4f, %.4f]",
                         name, arr.shape, np.nanmin(arr), np.nanmax(arr))
                break
    return bands, profile


def get_common_shape(bands):
    from collections import Counter
    shapes = set(b.shape for b in bands.values())
    if len(shapes) == 1:
        return list(shapes)[0]
    return Counter(b.shape for b in bands.values()).most_common(1)[0][0]


def resize_to(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    from scipy.ndimage import zoom
    factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, factors, order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1: Reproject EMODnet to S2 grid
# ---------------------------------------------------------------------------
def reproject_bathy_to_s2(bathy_profile, s2_profile):
    """Reproject EMODnet bathymetry to match S2 grid."""
    bathy_arr, bathy_prof = read_raster(BATHY_PATH)

    dst_shape = (s2_profile["height"], s2_profile["width"])
    dst_transform = s2_profile["transform"]
    dst_crs = s2_profile["crs"]

    reprojected = np.full(dst_shape, np.nan, dtype=np.float32)

    reproject(
        source=bathy_arr,
        destination=reprojected,
        src_transform=bathy_profile["transform"],
        src_crs=bathy_profile["crs"],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    out_profile = s2_profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    log.info("Reprojected EMODnet to S2 grid: shape=%s, range=[%.1f, %.1f]",
             reprojected.shape, np.nanmin(reprojected), np.nanmax(reprojected))
    return reprojected, out_profile


# ---------------------------------------------------------------------------
# Step 2: Sample calibration points
# ---------------------------------------------------------------------------
def sample_calibration_points(emodnet, bands, n_samples=10000):
    """Sample EMODnet depth at S2 pixel locations for calibration."""
    shape = get_common_shape(bands)
    b02 = resize_to(bands["B02"], shape)
    b03 = resize_to(bands["B03"], shape)
    emod_resized = resize_to(emodnet, shape)

    valid = (np.isfinite(emod_resized) &
             (b02 > 0) & (b03 > 0) &
             np.isfinite(b02) & np.isfinite(b03) &
             (emod_resized < 0) & (emod_resized > -100))

    n_valid = int(valid.sum())
    if n_valid < 50:
        log.error("Too few valid calibration points: %d", n_valid)
        return None, None, None

    idxs = np.where(valid)
    chosen = np.random.choice(len(idxs[0]), min(n_samples, n_valid), replace=False)
    rows = idxs[0][chosen]
    cols = idxs[1][chosen]

    depths = emod_resized[rows, cols]
    log.info("Sampled %d calibration points, depth range=[%.1f, %.1f]",
             len(depths), np.min(depths), np.max(depths))
    return rows, cols, depths


# ---------------------------------------------------------------------------
# Step 3: Calibrate Stumpf
# ---------------------------------------------------------------------------
def stumpf_ratio(b02, b03, n=STUMPF_N):
    with np.errstate(divide="ignore", invalid="ignore"):
        log_b = np.log(n * b02)
        log_g = np.log(n * b03)
        safe = (log_g > 0) & np.isfinite(log_b) & np.isfinite(log_g)
        ratio = np.where(safe, log_b / log_g, np.nan)
    return ratio, safe


def calibrate_stumpf(bands, emodnet, rows, cols, depths):
    """Optimize Stumpf m0, m1 to minimize RMSE vs EMODnet."""
    shape = get_common_shape(bands)
    b02 = resize_to(bands["B02"], shape)
    b03 = resize_to(bands["B03"], shape)

    ratio, safe = stumpf_ratio(b02, b03)
    ratio_samples = ratio[rows, cols]
    safe_samples = safe[rows, cols]

    mask = safe_samples & np.isfinite(ratio_samples)
    r = ratio_samples[mask]
    d = depths[mask]

    log.info("Stumpf calibration: %d valid ratio samples", len(r))

    def objective(params):
        m0, m1 = params
        pred = m1 * r + m0
        pred = -np.abs(pred)
        return np.sqrt(np.mean((pred - d) ** 2))

    result = minimize(objective, x0=[-16.0, 20.0], method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    m0_opt, m1_opt = result.x
    rmse_opt = result.fun

    log.info("Stumpf calibrated: m0=%.3f, m1=%.3f, RMSE=%.2f m", m0_opt, m1_opt, rmse_opt)

    ratio_full, safe_full = stumpf_ratio(b02, b03)
    depth = m1_opt * ratio_full + m0_opt
    depth = -np.abs(depth)
    depth = np.where(safe_full & (depth > -60) & (depth < 0), depth, np.nan)

    return depth, {"m0": round(float(m0_opt), 4), "m1": round(float(m1_opt), 4),
                    "rmse_emodnet": round(float(rmse_opt), 4)}


# ---------------------------------------------------------------------------
# Step 4: Calibrate Lyzenga
# ---------------------------------------------------------------------------
def calibrate_lyzenga(bands, emodnet, rows, cols, depths):
    """Lyzenga 6-band OLS regression against EMODnet depths."""
    available = [b for b in LYZENGA_BANDS if b in bands]
    shape = get_common_shape(bands)

    log_bands = {}
    for name in available:
        arr = resize_to(bands[name], shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.log(np.where(arr > 0, arr, np.nan))
        log_bands[name] = lb

    valid_mask = np.ones(shape, dtype=bool)
    for name in available:
        valid_mask &= np.isfinite(log_bands[name]) & (resize_to(bands[name], shape) > 0)

    X_sample = np.column_stack([log_bands[name][rows, cols] for name in available])
    valid_sample = np.all(np.isfinite(X_sample), axis=1)
    X_sample = X_sample[valid_sample]
    d_sample = depths[valid_sample]

    model = LinearRegression()
    model.fit(X_sample, d_sample)
    pred_sample = model.predict(X_sample)
    rmse = np.sqrt(np.mean((pred_sample - d_sample) ** 2))

    log.info("Lyzenga calibrated: RMSE=%.2f m, coeffs=%s, intercept=%.3f",
             rmse, [round(float(c), 4) for c in model.coef_], model.intercept_)

    X_full = np.column_stack([log_bands[name].ravel() for name in available])
    valid_flat = valid_mask.ravel()
    X_flat = X_full[valid_flat]
    depth_flat = np.full(valid_flat.shape, np.nan, dtype=np.float32)
    depth_flat[valid_flat] = model.predict(X_flat)
    depth = depth_flat.reshape(shape)

    depth = -np.abs(depth)
    depth = np.where((depth > -60) & (depth < 0), depth, np.nan)

    coeffs = {"intercept": round(float(model.intercept_), 4),
              "coefficients": {name: round(float(c), 4)
                               for name, c in zip(available, model.coef_)},
              "rmse_emodnet": round(float(rmse), 4)}
    return depth, coeffs


# ---------------------------------------------------------------------------
# Step 5: PCA depth correlated with EMODnet
# ---------------------------------------------------------------------------
def compute_pca_depth(bands, emodnet, rows, cols, depths):
    """PCA on water-penetrating bands, scale PC1 against EMODnet."""
    available = [b for b in WATER_PENETRATING if b in bands]
    shape = get_common_shape(bands)

    stack = []
    for name in available:
        arr = resize_to(bands[name], shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.log(np.where(arr > 0, arr, np.nan))
        stack.append(la)

    stack = np.stack(stack, axis=-1)
    band_arrs = np.array([resize_to(bands[n], shape) for n in available])
    valid_mask = np.all(np.isfinite(stack), axis=-1) & np.all(band_arrs > 0, axis=0)

    n_valid = int(valid_mask.sum())
    if n_valid < 50:
        log.error("Too few valid PCA pixels: %d", n_valid)
        return None, None

    pixels = stack[valid_mask]
    pca = PCA(n_components=min(2, len(available)))
    components = pca.fit_transform(pixels)

    pc1 = np.full(shape, np.nan, dtype=np.float32)
    pc1[valid_mask] = components[:, 0]

    pc1_samples = pc1[rows, cols]
    valid_sample = np.isfinite(pc1_samples)
    slope, intercept = np.polyfit(pc1_samples[valid_sample], depths[valid_sample], 1)
    depth = slope * pc1 + intercept
    depth = np.where(valid_mask, depth, np.nan)
    depth = -np.abs(depth)
    depth = np.where((depth > -60) & (depth < 0), depth, np.nan)

    pred = slope * pc1_samples[valid_sample] + intercept
    rmse = np.sqrt(np.mean((pred - depths[valid_sample]) ** 2))

    log.info("PCA depth: explained_var=%s, slope=%.3f, intercept=%.3f, RMSE=%.2f m",
             [round(float(v), 4) for v in pca.explained_variance_ratio_],
             slope, intercept, rmse)

    stats = {"explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
             "slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
             "rmse_emodnet": round(float(rmse), 4)}
    return depth, stats


# ---------------------------------------------------------------------------
# Step 6-7: Reef detection and validation
# ---------------------------------------------------------------------------
def detect_reef_candidates(depth, profile, depth_min, depth_max,
                           tri_pct=70.0, bpi_pct=60.0, min_area_px=4):
    """Detect reef candidates from calibrated depth map."""
    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    if valid.sum() == 0:
        log.warning("No pixels in depth range [%.0f, %.0f]", depth_min, depth_max)
        return None, 0

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
        return None, 0

    tri_thresh = np.percentile(tri_valid, tri_pct)
    bpi_thresh = np.percentile(bpi_valid, bpi_pct)

    rugose = np.isfinite(tri) & (tri >= tri_thresh)
    elevated = np.isfinite(bpi) & (bpi >= bpi_thresh)
    candidate_mask = valid & rugose & elevated

    n_candidates = int(candidate_mask.sum())
    log.info("Reef candidates: %d / %d depth pixels", n_candidates, int(valid.sum()))

    if n_candidates == 0:
        return None, 0

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
        return None, 0

    merged = [unary_union(geoms)] if len(geoms) > 50 else geoms
    gdf = gpd.GeoDataFrame(
        {
            "method": ["calibrated"] * len(merged),
            "depth_min_m": [depth_min] * len(merged),
            "depth_max_m": [depth_max] * len(merged),
            "area_m2": [g.area for g in merged],
        },
        geometry=merged,
        crs=crs,
    )
    return gdf, n_candidates


def validate_reef_candidates(gdf, depth_map, profile):
    """Validate reef candidates with depth statistics."""
    if gdf is None or len(gdf) == 0:
        return gdf

    from rasterio.mask import mask as rio_mask

    stats_list = []
    for idx, row in gdf.iterrows():
        try:
            geom = [row.geometry]
            out_image, _ = rio_mask(
                rasterio.open(os.path.join(OUT_DIR, "depth_calibrated_best.tif")),
                geom, crop=True, filled=False
            )
            vals = out_image[0]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                stats_list.append({
                    "mean_depth": round(float(np.mean(valid)), 2),
                    "std_depth": round(float(np.std(valid)), 2),
                    "min_depth": round(float(np.min(valid)), 2),
                    "max_depth": round(float(np.max(valid)), 2),
                    "n_pixels": len(valid),
                })
            else:
                stats_list.append({})
        except Exception:
            stats_list.append({})

    stats_df = gpd.GeoDataFrame(stats_list)
    for col in stats_df.columns:
        if col != "geometry":
            gdf[col] = stats_df[col].values

    return gdf


# ---------------------------------------------------------------------------
# Step 8: Comparison figure
# ---------------------------------------------------------------------------
def generate_figure(emodnet, stumpf, lyzenga, pca_depth, reef_gdf, best_name,
                    stumpf_stats, lyzenga_stats, pca_stats, s2_profile):
    """Generate 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))

    vmin, vmax = -25, 0

    ax = axes[0, 0]
    im = ax.imshow(emodnet, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("EMODnet Reference Depth (m)", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    ax = axes[0, 1]
    im = ax.imshow(stumpf, cmap="viridis", vmin=vmin, vmax=vmax)
    rmse_s = stumpf_stats.get("rmse_emodnet", "?")
    ax.set_title(f"Calibrated Stumpf Depth (m) [RMSE={rmse_s}]", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    ax = axes[1, 0]
    im = ax.imshow(lyzenga, cmap="viridis", vmin=vmin, vmax=vmax)
    rmse_l = lyzenga_stats.get("rmse_emodnet", "?")
    ax.set_title(f"Calibrated Lyzenga Depth (m) [RMSE={rmse_l}]", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    ax = axes[1, 1]
    best_map = {"stumpf": stumpf, "lyzenga": lyzenga, "pca": pca_depth}.get(best_name, stumpf)
    im = ax.imshow(best_map, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Depth (m)")

    if reef_gdf is not None and len(reef_gdf) > 0:
        transform = s2_profile["transform"]
        for geom in reef_gdf.geometry:
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                cols_coords = [(c - transform.c) / transform.a for c in x]
                rows_coords = [(r - transform.f) / transform.e for r in y]
                ax.plot(cols_coords, rows_coords, "r-", linewidth=1.5, alpha=0.8)
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    x, y = part.exterior.xy
                    cols_coords = [(c - transform.c) / transform.a for c in x]
                    rows_coords = [(r - transform.f) / transform.e for r in y]
                    ax.plot(cols_coords, rows_coords, "r-", linewidth=1.5, alpha=0.8)
    n_reef = len(reef_gdf) if reef_gdf is not None else 0
    ax.set_title(f"Reef Candidates ({best_name}) — {n_reef} polygons", fontsize=12)

    for ax in axes.ravel():
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.suptitle(
        "Calibrated SDB vs EMODnet — Santa Eulalia (Target ~12m depth)",
        fontsize=15, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "calibrated_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("EMODnet-Calibrated Multi-band Reef Analysis")
    log.info("=" * 60)

    # Step 1: Load data
    bathy_arr, bathy_profile = read_raster(BATHY_PATH)
    bands, s2_profile = load_bands()

    # Step 2: Reproject EMODnet to S2 grid
    emodnet, emod_profile = reproject_bathy_to_s2(bathy_profile, s2_profile)
    write_tif(emodnet, emod_profile, os.path.join(OUT_DIR, "emodnet_reprojected.tif"))

    # Step 3: Sample calibration points
    rows, cols, depths = sample_calibration_points(emodnet, bands)
    if rows is None:
        log.error("Calibration aborted — no valid points")
        return

    # Step 4: Calibrate Stumpf
    log.info("-" * 40)
    log.info("Calibrating Stumpf 2-band method...")
    stumpf_depth, stumpf_stats = calibrate_stumpf(bands, emodnet, rows, cols, depths)
    write_tif(stumpf_depth, s2_profile, os.path.join(OUT_DIR, "stumpf_calibrated.tif"))

    # Step 5: Calibrate Lyzenga
    log.info("-" * 40)
    log.info("Calibrating Lyzenga 6-band method...")
    lyzenga_depth, lyzenga_stats = calibrate_lyzenga(bands, emodnet, rows, cols, depths)
    write_tif(lyzenga_depth, s2_profile, os.path.join(OUT_DIR, "lyzenga_calibrated.tif"))

    # Step 6: PCA depth
    log.info("-" * 40)
    log.info("Computing PCA depth correlated with EMODnet...")
    pca_depth, pca_stats = compute_pca_depth(bands, emodnet, rows, cols, depths)
    if pca_depth is not None:
        write_tif(pca_depth, s2_profile, os.path.join(OUT_DIR, "pca_calibrated.tif"))

    # Step 7: Select best method
    methods = {
        "stumpf": (stumpf_depth, stumpf_stats),
        "lyzenga": (lyzenga_depth, lyzenga_stats),
    }
    if pca_depth is not None:
        methods["pca"] = (pca_depth, pca_stats)

    best_name = min(methods, key=lambda k: methods[k][1].get("rmse_emodnet", 9999))
    best_depth, best_stats = methods[best_name]
    log.info("Best method: %s (RMSE=%.2f m)", best_name, best_stats["rmse_emodnet"])

    write_tif(best_depth, s2_profile, os.path.join(OUT_DIR, "depth_calibrated_best.tif"))

    # Step 8: Detect reef candidates
    log.info("-" * 40)
    log.info("Detecting reef candidates [%.0f, %.0f] m...", REEF_DEPTH_MIN, REEF_DEPTH_MAX)
    reef_gdf, n_candidates = detect_reef_candidates(
        best_depth, s2_profile, REEF_DEPTH_MIN, REEF_DEPTH_MAX
    )

    reef_path = os.path.join(OUT_DIR, "reef_candidates_calibrated.geojson")
    if reef_gdf is not None and len(reef_gdf) > 0:
        reef_gdf.to_file(reef_path, driver="GeoJSON")
        log.info("Reef candidates → %s (%d polygons)", reef_path, len(reef_gdf))
    else:
        fc = {"type": "FeatureCollection", "features": []}
        with open(reef_path, "w") as f:
            json.dump(fc, f)
        log.info("No reef candidates found")

    # Step 9: Validate
    log.info("-" * 40)
    log.info("Validating reef candidates...")
    if reef_gdf is not None and len(reef_gdf) > 0:
        try:
            reef_gdf = validate_reef_candidates(reef_gdf, best_depth, s2_profile)
            val_path = os.path.join(OUT_DIR, "reef_candidates_validated.geojson")
            reef_gdf.to_file(val_path, driver="GeoJSON")
            log.info("Validated reef candidates → %s", val_path)
        except Exception as e:
            log.warning("Validation partial: %s", e)

    # Step 10: Generate figure
    log.info("-" * 40)
    log.info("Generating comparison figure...")
    fig_path = generate_figure(
        emodnet, stumpf_depth, lyzenga_depth, pca_depth,
        reef_gdf, best_name, stumpf_stats, lyzenga_stats, pca_stats, s2_profile
    )

    # Step 11: Summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bathy_source": BATHY_PATH,
        "band_dir": BAND_DIR,
        "target_depth_m": 12,
        "stumpf_calibrated": stumpf_stats,
        "lyzenga_calibrated": lyzenga_stats,
        "pca_calibrated": pca_stats,
        "best_method": best_name,
        "best_rmse": best_stats["rmse_emodnet"],
        "reef_depth_range": [REEF_DEPTH_MIN, REEF_DEPTH_MAX],
        "n_calibration_points": len(depths),
        "n_reef_candidates": n_candidates,
    }
    summary_path = os.path.join(OUT_DIR, "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print results
    print("\n" + "=" * 70)
    print("EMODnet-CALIBRATED SDB RESULTS — Santa Eulalia")
    print("=" * 70)
    print(f"{'Method':<25} {'RMSE vs EMODnet':>18} {'Key params'}")
    print("-" * 70)
    print(f"{'Stumpf 2-band':<25} {stumpf_stats['rmse_emodnet']:>15.2f} m"
          f"   m0={stumpf_stats['m0']}, m1={stumpf_stats['m1']}")
    print(f"{'Lyzenga 6-band':<25} {lyzenga_stats['rmse_emodnet']:>15.2f} m"
          f"   intercept={lyzenga_stats['intercept']}")
    if pca_stats:
        print(f"{'PCA depth':<25} {pca_stats['rmse_emodnet']:>15.2f} m"
              f"   slope={pca_stats['slope']}, intercept={pca_stats['intercept']}")
    print("-" * 70)
    print(f"Best method: {best_name} (RMSE={best_stats['rmse_emodnet']:.2f} m)")
    print(f"Calibration points: {len(depths)}")
    print(f"Reef candidates: {n_candidates} pixels in [{REEF_DEPTH_MIN}, {REEF_DEPTH_MAX}] m")
    if reef_gdf is not None:
        print(f"Reef polygons: {len(reef_gdf)}")
    print(f"\nOutputs → {OUT_DIR}/")
    print(f"Comparison figure: {fig_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
