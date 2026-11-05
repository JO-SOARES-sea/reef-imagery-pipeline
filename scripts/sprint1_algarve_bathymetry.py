#!/usr/bin/env python3
"""
sprint1_algarve_bathymetry.py
═══════════════════════════════════════════════════════════════════════════════
SPRINT 1 — Bathymetric Map of Algarve Central Coast
            Faro (-7.93°W) → Armação de Pêra (-8.35°W)

PHASE A: Sentinel-2 Stumpf SDB (no login needed)
  1. Find best Sentinel-2 dates over the area (cloud<5%, multi-pass)
  2. Read B02 (blue) + B03 (green) clipped to coastal bbox
  3. Apply Stumpf log-ratio bathymetry per scene
  4. Median composite over best dates → clean bathy map
  5. Mask land (NDWI) + deep water (saturation)
  6. Output GeoTIFF + visual

PHASE B (next): Calibrate m0/m1 with ICESat-2 ATL03 (requires NASA EarthData)
"""
import warnings; warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — Faro → Armação de Pêra
# ═══════════════════════════════════════════════════════════════════════════

# Coastal bbox (WGS84)
BBOX_WGS84 = {
    "west":  -8.38,   # west of Armação de Pêra
    "east":  -7.90,   # east of Faro (Praia de Faro)
    "south":  36.99,  # ~2km offshore (includes 20m+ isobath)
    "north":  37.12,  # onshore coast
}

YEARS = 5              # search window
MAX_CLOUD = 5          # % — strict
TOP_N_DATES = 8        # composite from top N dates
OUT_RES_M = 10         # output grid resolution (m)

# Stumpf parameters — Algarve oligotrophic waters initial calibration
# Calibrated to reach ~25m in clear water (Kd~0.045)
# These will be refined in Phase B with ICESat-2 ground truth
STUMPF_M0 = -28.0     # offset (m)
STUMPF_M1 = 32.0      # scale  (m)
STUMPF_N  = 1500.0    # log scaling factor

# Output paths
OUT_DIR = Path(os.path.dirname(__file__)) / "outputs" / "sprint1_bathy"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TIF = OUT_DIR / "algarve_central_bathy_10m_v1.tif"
OUT_CSV = OUT_DIR / "scenes_used.csv"

print("="*80)
print("  SPRINT 1 — Algarve Central Coast Bathymetry (Phase A: Sentinel-2)")
print(f"  BBox: {BBOX_WGS84['west']}°W → {BBOX_WGS84['east']}°W, "
      f"{BBOX_WGS84['south']}°N → {BBOX_WGS84['north']}°N")
print("  Faro ↔ Armação de Pêra (~40km × 14km)")
print(f"  Output: {OUT_TIF.name}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Search Sentinel-2 best clear scenes
# ═══════════════════════════════════════════════════════════════════════════

print("\n[1] Searching Sentinel-2 L2A via Planetary Computer STAC...")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

end_date = datetime.now()
start_date = datetime(end_date.year - YEARS, 1, 1)

# Use bbox (faster than polygon intersect on STAC backend)
bbox = [BBOX_WGS84["west"], BBOX_WGS84["south"], BBOX_WGS84["east"], BBOX_WGS84["north"]]

def _search_year(year, max_retries=3):
    """Search a single year window — splits large queries to avoid timeouts."""
    import time
    y_start = f"{year}-01-01"
    y_end   = f"{year}-12-31"
    # Prefer clear-water months only to keep result set small
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{y_start}/{y_end}",
                query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
                limit=100,
            )
            return list(search.items())
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    ⚠ {year}: giving up ({type(e).__name__})")
                return []
            time.sleep(3)
    return []

def _search_window(start, end, max_retries=3):
    import time
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start}/{end}",
                query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
                limit=50,
            )
            return list(search.items())
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    ⚠ {start} window: giving up ({type(e).__name__})")
                return []
            time.sleep(3)
    return []

items = []
# Search only clear-water months Sep/Oct/Jan/Dec for last YEARS
for yr in range(end_date.year - YEARS + 1, end_date.year + 1):
    for month_pair in [("09-01", "10-31"), ("01-01", "01-31"), ("12-01", "12-31")]:
        s, e = f"{yr}-{month_pair[0]}", f"{yr}-{month_pair[1]}"
        print(f"    {s} → {e}...", end="", flush=True)
        win_items = _search_window(s, e)
        print(f" {len(win_items)}")
        items.extend(win_items)
print(f"    Found {len(items)} scenes with <{MAX_CLOUD}% cloud cover")

# Build dataframe with key metadata
scene_data = []
for item in items:
    props = item.properties
    if props.get("s2:nodata_pixel_percentage", 100) > 25:
        continue
    scene_data.append({
        "date_str": item.datetime.strftime("%Y-%m-%d"),
        "date": pd.Timestamp(item.datetime.date()),
        "tile":   props.get("s2:mgrs_tile", "?"),
        "cloud":  props.get("eo:cloud_cover", 100),
        "sun_el": props.get("view:sun_elevation", 45),
        "month":  item.datetime.month,
        "item":   item,
    })

df = pd.DataFrame(scene_data)
print(f"    After nodata filter: {len(df)} scenes")
print(f"    Tiles covered: {df['tile'].unique().tolist()}")

# Prefer clear-water months (Sep/Oct/Jan/Dec) + low cloud + high sun
df["clear_water_bonus"] = df["month"].isin([1, 9, 10, 12]).astype(int) * 0.3
df["quality_score"] = (
    df["clear_water_bonus"] +
    (1 - df["cloud"] / MAX_CLOUD) * 0.5 +
    (df["sun_el"] / 90) * 0.2
)

# Top N candidates — diversify per tile to cover whole bbox
df_unique = df.sort_values("quality_score", ascending=False).drop_duplicates("date_str")
parts = []
per_tile = max(1, TOP_N_DATES // df_unique["tile"].nunique())
for tile, grp in df_unique.groupby("tile"):
    parts.append(grp.head(per_tile + 2))
df_top = pd.concat(parts).sort_values("quality_score", ascending=False).head(TOP_N_DATES * 2)
print(f"\n    Diversified across {df_top['tile'].nunique()} tile(s): "
      f"{df_top['tile'].value_counts().to_dict()}")
print(f"\n    TOP {TOP_N_DATES} scenes selected:")
print(f"    {'Date':<12} {'Tile':<8} {'Cloud':>6} {'Sun':>5} {'Month':>5} {'Score':>6}")
print("    " + "-"*48)
for _, r in df_top.iterrows():
    print(f"    {r['date_str']:<12} {r['tile']:<8} {r['cloud']:>5.1f}% "
          f"{r['sun_el']:>4.1f}° {r['month']:>5} {r['quality_score']:>5.3f}")

# Save scene list
df_top.drop(columns=["item"]).to_csv(OUT_CSV, index=False)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Determine output grid (UTM 29N for Algarve)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n[2] Setting up output grid (UTM 29N, {OUT_RES_M}m)...")

# Sentinel-2 tiles over Algarve use EPSG:32629 (UTM 29N)
DST_CRS = "EPSG:32629"
to_utm = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)

# Convert bbox corners to UTM
xs, ys = [], []
for lon in [BBOX_WGS84["west"], BBOX_WGS84["east"]]:
    for lat in [BBOX_WGS84["south"], BBOX_WGS84["north"]]:
        x, y = to_utm.transform(lon, lat)
        xs.append(x); ys.append(y)

minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)
# Snap to OUT_RES_M grid
minx = np.floor(minx / OUT_RES_M) * OUT_RES_M
maxx = np.ceil(maxx / OUT_RES_M) * OUT_RES_M
miny = np.floor(miny / OUT_RES_M) * OUT_RES_M
maxy = np.ceil(maxy / OUT_RES_M) * OUT_RES_M

width  = int((maxx - minx) / OUT_RES_M)
height = int((maxy - miny) / OUT_RES_M)
transform = from_origin(minx, maxy, OUT_RES_M, OUT_RES_M)

print(f"    Grid: {width} × {height} pixels @ {OUT_RES_M}m")
print(f"    UTM bounds: [{minx:.0f}, {miny:.0f}] → [{maxx:.0f}, {maxy:.0f}]")
print(f"    Area: {(maxx-minx)/1000:.1f} × {(maxy-miny)/1000:.1f} km")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: For each scene, read B02/B03 windowed, compute Stumpf SDB
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3] Reading bands + computing Stumpf SDB per scene...")

def stumpf_sdb(b02, b03, m0=STUMPF_M0, m1=STUMPF_M1, n=STUMPF_N):
    """Stumpf et al. 2003: depth = m1 * ln(n*B02)/ln(n*B03) + m0"""
    eps = 1e-6
    valid = (b02 > eps) & (b03 > eps) & (b02 < 0.20) & (b03 < 0.20)
    ln_b02 = np.log(n * np.where(valid, b02, eps) + eps)
    ln_b03 = np.log(n * np.where(valid, b03, eps) + eps)
    ratio = ln_b02 / (ln_b03 + eps)
    depth = m1 * ratio + m0
    depth = np.where(valid, depth, np.nan)
    return depth.astype(np.float32)

def ndwi_water_mask(b03, b08):
    """McFeeters NDWI: water > 0.3, land < 0"""
    eps = 1e-6
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    return ndwi > 0.05  # permissive water mask

env = rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_USE_HEAD="NO",
)

depth_stack = []
valid_stack = []

with env:
    for idx, row in df_top.iterrows():
        item = row["item"]
        date_str = row["date_str"]
        sys.stdout.write(f"\r    [{len(depth_stack)+1:>2}/{len(df_top)}] {date_str}  ")
        sys.stdout.flush()

        try:
            b02_href = item.assets["B02"].href
            b03_href = item.assets["B03"].href
            b08_href = item.assets["B08"].href

            with rasterio.open(b02_href) as src:
                src_crs = src.crs
                # If different UTM zone, skip this tile (rare for our area)
                if src_crs.to_epsg() != 32629:
                    print(f"\n    ⚠ {date_str}: tile in CRS {src_crs} (skipped)")
                    continue
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                b02 = src.read(1, window=win, boundless=True, fill_value=0,
                               out_shape=(height, width)).astype(np.float32) / 10000.0
            with rasterio.open(b03_href) as src:
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                b03 = src.read(1, window=win, boundless=True, fill_value=0,
                               out_shape=(height, width)).astype(np.float32) / 10000.0
            with rasterio.open(b08_href) as src:
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                b08 = src.read(1, window=win, boundless=True, fill_value=0,
                               out_shape=(height, width)).astype(np.float32) / 10000.0

            # Sunglint correction (subtract NIR over water)
            # b08 represents NIR; over clear water it's ~0, sunglint adds offset
            water_mask = ndwi_water_mask(b03, b08)
            if water_mask.sum() < 100:
                continue
            glint = np.where(water_mask & (b08 > 0), b08, 0)
            b02_corr = np.clip(b02 - 0.85 * glint, 0, 1)
            b03_corr = np.clip(b03 - 0.95 * glint, 0, 1)

            # Stumpf depth
            depth = stumpf_sdb(b02_corr, b03_corr)
            # Mask: water only, valid depth range (0-35m)
            depth = np.where(water_mask & (depth > -1) & (depth < 35), depth, np.nan)
            depth = np.clip(depth, 0, 35)
            depth = np.where(water_mask, depth, np.nan)

            depth_stack.append(depth)
            valid_stack.append(~np.isnan(depth))

        except Exception as e:
            print(f"\n    ✗ {date_str}: {e}")
            continue

print(f"\r    Processed {len(depth_stack)}/{len(df_top)} scenes successfully.       ")

if not depth_stack:
    print("  ❌ No valid scenes processed!")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Median composite + uncertainty
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4] Computing median composite + uncertainty...")

depth_arr = np.stack(depth_stack, axis=0)  # (N, H, W)
valid_arr = np.stack(valid_stack, axis=0)
n_obs = valid_arr.sum(axis=0).astype(np.int16)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    depth_median = np.nanmedian(depth_arr, axis=0).astype(np.float32)
    depth_std    = np.nanstd(depth_arr, axis=0).astype(np.float32)

# Mask cells with too few observations
min_obs = max(2, len(depth_stack) // 3)
depth_median = np.where(n_obs >= min_obs, depth_median, np.nan)

# Stats
valid_px = ~np.isnan(depth_median)
print(f"    Cells with depth: {valid_px.sum():,} / {valid_px.size:,} "
      f"({100*valid_px.sum()/valid_px.size:.1f}%)")
if valid_px.sum() > 0:
    print(f"    Depth range:  {np.nanmin(depth_median):.1f} → {np.nanmax(depth_median):.1f} m")
    print(f"    Mean depth:   {np.nanmean(depth_median):.1f} m")
    print(f"    Std (between dates):   {np.nanmean(depth_std):.2f} m")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Save as multi-band GeoTIFF
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n[5] Writing GeoTIFF: {OUT_TIF}")

profile = {
    "driver":    "GTiff",
    "height":    height,
    "width":     width,
    "count":     3,
    "dtype":     "float32",
    "crs":       DST_CRS,
    "transform": transform,
    "compress":  "deflate",
    "predictor": 2,
    "tiled":     True,
    "blockxsize": 256,
    "blockysize": 256,
    "nodata":    np.nan,
}

with rasterio.open(OUT_TIF, "w", **profile) as dst:
    dst.write(depth_median, 1)
    dst.set_band_description(1, "depth_median_m")
    dst.write(depth_std, 2)
    dst.set_band_description(2, "depth_std_m")
    dst.write(n_obs.astype(np.float32), 3)
    dst.set_band_description(3, "n_observations")
    dst.update_tags(
        site="Algarve Central (Faro → Armação de Pêra)",
        method="Stumpf log-ratio SDB",
        m0=str(STUMPF_M0), m1=str(STUMPF_M1), n_param=str(STUMPF_N),
        n_scenes_input=str(len(df_top)),
        n_scenes_used=str(len(depth_stack)),
        date_range=f"{df_top['date_str'].min()} to {df_top['date_str'].max()}",
        calibration_status="UNCALIBRATED — needs ICESat-2 (Phase B)",
        generated=datetime.now().isoformat(),
    )

size_mb = OUT_TIF.stat().st_size / 1e6
print(f"    ✓ Saved {size_mb:.1f} MB")
print("    Bands: 1=depth_median(m), 2=depth_std(m), 3=n_observations")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Sample at dive sites for sanity check
# ═══════════════════════════════════════════════════════════════════════════

print("\n[6] Sanity check at known dive sites:")

SITES = {
    "Pedra do Alto":     {"lat": 37.05815, "lon": -8.20982, "expected": 16},  # 37°03.489'N 008°12.589'W
    "Pedra Sta Eulália": {"lat": 37.068978, "lon": -8.210328, "expected": 12},
}

with rasterio.open(OUT_TIF) as src:
    depth_band = src.read(1)
    std_band = src.read(2)
    nobs_band = src.read(3)
    for name, cfg in SITES.items():
        x, y = to_utm.transform(cfg["lon"], cfg["lat"])
        try:
            row, col = src.index(x, y)
            # Sample a 5x5 window around the dive site (50m radius)
            r0, r1 = max(0, row-2), min(src.height, row+3)
            c0, c1 = max(0, col-2), min(src.width, col+3)
            window = depth_band[r0:r1, c0:c1]
            window_std = std_band[r0:r1, c0:c1]
            window_nobs = nobs_band[r0:r1, c0:c1]
            exp = cfg["expected"]
            valid = ~np.isnan(window)
            if valid.any():
                d_med = float(np.nanmedian(window))
                d_min = float(np.nanmin(window))
                d_max = float(np.nanmax(window))
                std_med = float(np.nanmedian(window_std[valid]))
                nobs_med = int(np.nanmedian(window_nobs[valid]))
                diff = d_med - exp
                marker = "✓" if abs(diff) < 5 else "⚠"
                print(f"    {marker} {name:<22} "
                      f"median={d_med:5.1f}m (range {d_min:.1f}–{d_max:.1f})  "
                      f"expected={exp}m  diff={diff:+5.1f}m  std={std_med:.1f}  n={nobs_med}")
            else:
                print(f"    ✗ {name:<22} NaN in 5x5 window (likely too deep for Stumpf SDB)")
        except Exception as e:
            print(f"    ✗ {name}: {e}")

# ── Generate preview PNG ──
print("\n[7] Generating preview image...")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Map 1: Depth median
    ax = axes[0]
    cmap = plt.cm.viridis_r
    cmap.set_bad("white", alpha=0)
    im1 = ax.imshow(
        depth_band, cmap=cmap, vmin=0, vmax=25,
        extent=[minx, maxx, miny, maxy], origin="upper"
    )
    ax.set_title(f"Algarve Central Bathymetry — Median of {len(depth_stack)} Sentinel-2 scenes (Stumpf SDB)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("UTM East (m)"); ax.set_ylabel("UTM North (m)")
    cb1 = plt.colorbar(im1, ax=ax, label="Depth (m)", fraction=0.025, pad=0.02)

    # Mark dive sites
    for name, cfg in SITES.items():
        x, y = to_utm.transform(cfg["lon"], cfg["lat"])
        ax.plot(x, y, "rx", markersize=12, mew=2)
        ax.annotate(name, (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=9, color="red", fontweight="bold")

    # Map 2: Number of observations
    ax = axes[1]
    im2 = ax.imshow(
        np.where(nobs_band > 0, nobs_band, np.nan),
        cmap="plasma", vmin=1, vmax=len(depth_stack),
        extent=[minx, maxx, miny, maxy], origin="upper"
    )
    ax.set_title("Coverage — Number of valid scenes per pixel", fontsize=11)
    ax.set_xlabel("UTM East (m)"); ax.set_ylabel("UTM North (m)")
    cb2 = plt.colorbar(im2, ax=ax, label="N scenes", fraction=0.025, pad=0.02)

    plt.tight_layout()
    preview_path = OUT_DIR / "preview.png"
    plt.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    ✓ {preview_path}")
except Exception as e:
    print(f"    ⚠ preview failed: {e}")

print(f"\n{'='*80}")
print("  ✓ PHASE A COMPLETE")
print(f"{'='*80}")
print(f"""
  Generated: {OUT_TIF}
  
  ⚠  CALIBRAÇÃO INICIAL — assenta em Stumpf m0={STUMPF_M0}, m1={STUMPF_M1}
     Valores absolutos podem ter erro sistemático ±3-5m.
     A morfologia/contornos são fiáveis.
  
  PRÓXIMO PASSO (Phase B):
    1. Regista NASA EarthData: https://urs.earthdata.nasa.gov/users/new
    2. Run: phase_b_calibrate_icesat2.py
       → Faz download ATL03 das ~50 datas Set/Out/Jan/Dez
       → Extrai fotões batimétricos (signal_conf ≥ 3)
       → Mínimos quadrados para refinar m0, m1
       → Gera v2 com erro ±0.5-1m
  
  PARA VISUALIZAR AGORA:
    Abrir em QGIS/ArcGIS, ou:
    python -c "
    import rasterio
    from rasterio.plot import show
    import matplotlib.pyplot as plt
    with rasterio.open('{OUT_TIF}') as src:
        fig, ax = plt.subplots(figsize=(14,5))
        show(src.read(1), transform=src.transform, ax=ax,
             cmap='Blues_r', vmin=0, vmax=25)
        ax.set_title('Algarve Central Bathymetry (m)')
        plt.savefig('{OUT_DIR}/preview.png', dpi=120)
    "
""")
