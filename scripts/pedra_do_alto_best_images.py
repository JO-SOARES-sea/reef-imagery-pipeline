#!/usr/bin/env python3
"""
pedra_do_alto_best_images.py
═══════════════════════════════════════════════════════════════════════════════
Top 10 historical visibility days at Pedra do Alto (16m depth)
→ Filter for clear sky + calm water (good satellite imagery conditions)
→ Fetch Sentinel-2 via Planetary Computer STAC
→ Calculate per-image quality metrics:
    - SNR (signal-to-noise)
    - benthic_contrast (Laplacian/Sobel)
    - fft_cleanliness (surface calmness)
    - edge_entropy (structural information)
    - kd_mean (water clarity from band ratio)
→ Rank by composite Benthic Visibility Index (BVI)
"""
import sys, os
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client

# ── Reuse existing physics functions ──
from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio

# ═══════════════════════════════════════════════════════════════════════════
# SITE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
SITE_LAT = 37.05815       # 37°03.489'N — GPS confirmed 2026-06-22
SITE_LON = -8.20982       # 008°12.589'W — GPS confirmed 2026-06-22
DEPTH = 16                # meters
BUFFER_M = 500            # pixel window around coordinate
YEARS = 8                 # search window
MAX_CLOUD = 5             # % — strict for clear sky
MAX_WAVE_HEURISTIC = True # use wave/wind filtering

print("="*80)
print("  PEDRA DO ALTO — Top Visibility Days + Sentinel-2 Image Quality")
print(f"  {SITE_LAT:.5f}°N, {abs(SITE_LON):.5f}°W | Depth: {DEPTH}m")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load satellite visibility history → find top-10 clear days
# ═══════════════════════════════════════════════════════════════════════════

print("\n[1] Loading satellite visibility history...")

# Check if we have pre-computed training data
training_csv = os.path.join(os.path.dirname(__file__), "..", "UnderWater_Visibility",
                            "data", "models", "pedra_do_alto_training_data.csv")
if os.path.exists(training_csv):
    df_hist = pd.read_csv(training_csv, parse_dates=["date"])
    print(f"    Loaded {len(df_hist)} days from training data")
else:
    # Fallback: load directly from Copernicus
    print("    Training data not found, loading from Copernicus Marine...")
    import copernicusmarine
    BUFFER = 0.02
    ds_kd = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D",
        variables=["KD490"],
        minimum_longitude=SITE_LON-BUFFER, maximum_longitude=SITE_LON+BUFFER,
        minimum_latitude=SITE_LAT-BUFFER, maximum_latitude=SITE_LAT+BUFFER,
        start_datetime="2017-01-01", end_datetime="2025-12-31",
    )
    kd = ds_kd["KD490"].mean(dim=["latitude","longitude"], skipna=True)
    df_hist = kd.to_dataframe().reset_index()
    df_hist.columns = ["date","kd490"]
    df_hist = df_hist.dropna()
    df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.normalize()
    df_hist["secchi"] = 1.0 / df_hist["kd490"]

# Compute depth visibility
def vis_at_depth(secchi, depth=DEPTH):
    if secchi <= 0: return 0
    kd = 1.0 / secchi
    return 1.5 * secchi * np.sqrt(np.exp(-kd * depth))

def light_at_depth(secchi, depth=DEPTH):
    if secchi <= 0: return 0
    return np.exp(-depth / secchi) * 100

if "vis_depth" not in df_hist.columns:
    if "secchi" not in df_hist.columns:
        df_hist["secchi"] = 1.0 / df_hist["kd490"]
    df_hist["vis_depth"] = df_hist["secchi"].apply(vis_at_depth)
    df_hist["light_pct"] = df_hist["secchi"].apply(light_at_depth)

# Filter for conditions suitable for satellite imaging:
# 1. High visibility (top days)
# 2. We'll cross-check cloud cover from meteo data if available
if "wave_height_mean" in df_hist.columns:
    # Use calm sea filter
    calm_mask = (df_hist["wave_height_mean"] <= 0.8) & (df_hist["vis_depth"] > 5)
    df_calm = df_hist[calm_mask].nlargest(30, "vis_depth")
    print(f"    Filtered: {len(df_calm)} calm + clear days (wave ≤ 0.8m, vis > 5m)")
else:
    df_calm = df_hist.nlargest(30, "vis_depth" if "vis_depth" in df_hist.columns else "secchi")
    print("    Top 30 visibility days selected")

# Print top-10 from satellite data
vis_col = "vis_depth" if "vis_depth" in df_hist.columns else "secchi"
top10 = df_calm.nlargest(10, vis_col)
print("\n  TOP 10 VISIBILITY DAYS (satellite Kd490):")
print(f"  {'#':>3} {'Date':<12} {'Secchi':>7} {'@16m':>7} {'Light%':>7}", end="")
if "wave_height_mean" in df_hist.columns:
    print(f" {'Wave':>6}", end="")
print()
print("  " + "-"*55)
for i, (_, r) in enumerate(top10.iterrows(), 1):
    secchi = r.get("secchi", 1.0/r.get("kd490", 0.1))
    vis = r.get("vis_depth", vis_at_depth(secchi))
    light = r.get("light_pct", light_at_depth(secchi))
    line = f"  {i:>3}. {r['date'].strftime('%Y-%m-%d')}  {secchi:>5.1f}m {vis:>5.1f}m {light:>5.1f}%"
    if "wave_height_mean" in df_hist.columns:
        line += f" {r['wave_height_mean']:>5.2f}m"
    print(line)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Search Sentinel-2 STAC for those dates + extra candidates
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2] Searching Sentinel-2 via Planetary Computer STAC...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace
)

end_date = datetime.now()
start_date = datetime(end_date.year - YEARS, 1, 1)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [SITE_LON, SITE_LAT]},
    datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
    query={"eo:cloud_cover": {"lt": MAX_CLOUD}}
)

items = list(search.items())
print(f"    Found {len(items)} scenes with <{MAX_CLOUD}% cloud cover")

# Build STAC dataframe
stac_data = []
for item in items:
    props = item.properties
    if props.get("s2:nodata_pixel_percentage", 100) > 20:
        continue
    stac_data.append({
        "date_str": item.datetime.strftime("%Y-%m-%d"),
        "date": pd.Timestamp(item.datetime.date()),
        "cloud_cover": props.get("eo:cloud_cover", 100),
        "sun_elevation": props.get("view:sun_elevation", 45),
        "item": item,
    })

df_stac = pd.DataFrame(stac_data)
if df_stac.empty:
    print("  ❌ No viable Sentinel-2 scenes found!")
    sys.exit(1)

# Cross-reference with visibility data
df_stac = df_stac.merge(
    df_hist[["date", "secchi", "vis_depth", "light_pct"]].drop_duplicates("date"),
    on="date", how="left"
)

# Sort by visibility (prefer days we know are clear)
df_stac["vis_depth"] = df_stac["vis_depth"].fillna(0)
df_stac = df_stac.sort_values("vis_depth", ascending=False).drop_duplicates("date_str")
candidates = df_stac.head(25)  # top 25 candidates to evaluate

print(f"    Candidates with visibility data: {candidates['vis_depth'].gt(0).sum()}")
print(f"    Evaluating top {len(candidates)} scenes...")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: VSI streaming — read B02/B03/B08, compute quality metrics
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3] Computing image quality metrics via VSI streaming...")

results = []
env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

for idx, row in candidates.iterrows():
    item = row["item"]
    date_str = row["date_str"]
    sys.stdout.write(f"\r    Processing {date_str}...                ")
    sys.stdout.flush()

    b02_href = item.assets["B02"].href
    b03_href = item.assets["B03"].href
    b08_href = item.assets["B08"].href if "B08" in item.assets else None

    with env:
        try:
            with rasterio.open(b02_href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(SITE_LON, SITE_LAT)
                window = from_bounds(x - BUFFER_M, y - BUFFER_M,
                                     x + BUFFER_M, y + BUFFER_M, src.transform)
                b02_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(b03_href) as src:
                b03_arr = src.read(1, window=window).astype(np.float32)
            b08_arr = None
            if b08_href:
                with rasterio.open(b08_href) as src:
                    b08_arr = src.read(1, window=window).astype(np.float32)
        except Exception:
            continue

    # Convert to BOA reflectance
    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03 = np.clip(b03_arr / 10000.0, 0, 1.5)

    if b02.max() == 0 or np.all(np.isnan(b02)):
        continue

    # Cloud mask — LOCAL cloud check with water-safe detection
    from src.enhancer import water_safe_cloud_mask
    b08 = np.clip(b08_arr / 10000.0, 0, 1.5) if b08_arr is not None else None
    local_cloud_pct = water_safe_cloud_mask(b02, b08)
    # Only reject if LOCAL cloud at GPS is high (not tile-level STAC cloud)
    if local_cloud_pct > 50:
        continue

    # --- Sunglint correction ---
    p95_b02 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
    p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
    b02_corr = np.clip(b02 - 0.8 * p95_b02 * 0.05, 0, 1.0)
    b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)

    # ── METRIC 1: SNR ──
    snr_map = make_snr_map(b02_corr, window=5)
    snr_mean = float(np.nanmean(snr_map))

    # ── METRIC 2: Benthic contrast (Laplacian + Sobel) ──
    macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    laplacian = cv2.Laplacian(macro, cv2.CV_32F)
    laplacian_var = float(np.var(laplacian))

    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = float(np.mean(grad_mag))

    benthic_contrast = laplacian_var * 1e6 + sobel_mean * 100

    # ── METRIC 3: FFT cleanliness ──
    f_transform = np.fft.fft2(b02_corr)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    h, w = b02_corr.shape
    cy, cx = h // 2, w // 2

    r_low = 5
    yy, xx = np.ogrid[:h, :w]
    mask_low = (xx - cx)**2 + (yy - cy)**2 <= r_low**2
    r_high = 15
    mask_high = (xx - cx)**2 + (yy - cy)**2 >= r_high**2

    low_power = np.sum(power[mask_low])
    high_power = np.sum(power[mask_high])
    fft_cleanliness = low_power / (high_power + 1e-12)

    # ── METRIC 4: Edge entropy ──
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) /
                     (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0

    # ── METRIC 5: Kd mean (band-ratio estimate) ──
    kd_prior = 0.045  # Algarve clear water baseline
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, kd_prior)

    # Raw signal level (for glint/fog detection)
    raw_mean = float(np.mean(b02[b02 > 0])) if np.any(b02 > 0) else 0
    p1 = np.percentile(b02[b02 > 0], 1) if np.any(b02 > 0) else 0
    p99 = np.percentile(b02[b02 > 0], 99) if np.any(b02 > 0) else 0
    contrast_ratio = p99 / p1 if p1 > 0 else 1.0

    results.append({
        "date": date_str,
        "cloud": row["cloud_cover"],
        "secchi": row.get("secchi", np.nan),
        "vis_16m": row.get("vis_depth", np.nan),
        "light_pct": row.get("light_pct", np.nan),
        "snr": snr_mean,
        "benthic_contrast": benthic_contrast,
        "fft_cleanliness": fft_cleanliness,
        "edge_entropy": edge_entropy,
        "kd_mean": kd_est,
        "contrast_ratio": contrast_ratio,
        "raw_mean": raw_mean,
    })

print(f"\r    Processed {len(results)} scenes successfully.         ")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Compute Benthic Visibility Index (BVI) and rank
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4] Computing Benthic Visibility Index...")

df_res = pd.DataFrame(results)

if df_res.empty:
    print("  ❌ No scenes could be processed!")
    sys.exit(1)

# Normalize each metric to [0, 1]
def norm(s):
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-12: return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

df_res["n_snr"]        = norm(df_res["snr"])
df_res["n_contrast"]   = norm(df_res["benthic_contrast"])
df_res["n_clean"]      = norm(np.log10(df_res["fft_cleanliness"].clip(lower=1)))
df_res["n_entropy"]    = norm(df_res["edge_entropy"])
df_res["n_kd"]         = norm(1.0 / df_res["kd_mean"])  # lower Kd = clearer = better

# Penalize bad signal range (sunglint or fog)
signal_ok = ((df_res["raw_mean"] >= 0.04) & (df_res["raw_mean"] <= 0.15)).astype(float)
signal_ok = signal_ok.replace(0, 0.2)  # don't zero out, just penalize

# Benthic Visibility Index (weighted composite)
df_res["BVI"] = (
    0.25 * df_res["n_clean"] +       # calm water (no ripple noise)
    0.25 * df_res["n_kd"] +           # clear water (low attenuation)
    0.20 * df_res["n_contrast"] +     # benthic structure visible
    0.15 * df_res["n_entropy"] +      # structural complexity
    0.15 * df_res["n_snr"]            # signal quality
) * signal_ok

df_res = df_res.sort_values("BVI", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("  🏆 RANKING — Índice de Visibilidade Bentônica (BVI)")
print(f"  Pedra do Alto | {SITE_LAT:.5f}°N, {abs(SITE_LON):.5f}°W | 16m depth")
print(f"{'='*80}")

print(f"\n  {'#':>2}  {'Date':<12} {'BVI':>5} {'Cloud':>6} {'Secchi':>7} {'@16m':>6} "
      f"{'SNR':>6} {'Contrast':>9} {'Clean':>9} {'Entropy':>8} {'Kd':>7} {'Signal':>7}")
print("  " + "-"*105)

for i, r in df_res.iterrows():
    secchi_str = f"{r['secchi']:.1f}m" if pd.notna(r['secchi']) else "  n/a"
    vis_str    = f"{r['vis_16m']:.1f}m" if pd.notna(r['vis_16m']) else " n/a"

    # Star rating
    if r["BVI"] >= 0.7:
        star = "⭐⭐⭐"
    elif r["BVI"] >= 0.5:
        star = "⭐⭐"
    elif r["BVI"] >= 0.3:
        star = "⭐"
    else:
        star = ""

    print(f"  {i+1:>2}. {r['date']:<12} {r['BVI']:.3f} {r['cloud']:>5.1f}% "
          f"{secchi_str:>6} {vis_str:>5} "
          f"{r['snr']:>5.1f} {r['benthic_contrast']:>8.1f} "
          f"{r['fft_cleanliness']:>8.0f} {r['edge_entropy']:>7.3f} "
          f"{r['kd_mean']:>6.4f} {r['raw_mean']:>6.3f}  {star}")

# Best day summary
best = df_res.iloc[0]
print(f"\n  ⭐ MELHOR DATA PARA IMAGEM BENTÔNICA: {best['date']}")
print(f"     BVI Score:           {best['BVI']:.3f}")
print(f"     Secchi:              {best['secchi']:.1f}m" if pd.notna(best['secchi']) else "     Secchi:              n/a")
print(f"     Vis @16m:            {best['vis_16m']:.1f}m" if pd.notna(best['vis_16m']) else "     Vis @16m:            n/a")
print(f"     Kd estimado:         {best['kd_mean']:.4f}")
print(f"     SNR:                 {best['snr']:.1f}")
print(f"     FFT Cleanliness:     {best['fft_cleanliness']:.0f}")
print(f"     Benthic Contrast:    {best['benthic_contrast']:.1f}")
print(f"     Edge Entropy:        {best['edge_entropy']:.3f}")

# Top 3 recommended dates
print("\n  📋 TOP 3 DATAS RECOMENDADAS:")
for i in range(min(3, len(df_res))):
    r = df_res.iloc[i]
    print(f"     {i+1}. {r['date']} — BVI={r['BVI']:.3f} | Kd={r['kd_mean']:.4f} | Clean={r['fft_cleanliness']:.0f}")

print(f"\n{'='*80}")
