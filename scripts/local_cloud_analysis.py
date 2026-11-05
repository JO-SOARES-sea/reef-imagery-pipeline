#!/usr/bin/env python3
"""
local_cloud_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Validação local das 5 melhores datas do hybrid_stac_physical_orchestrator.

Para cada data候选者, baixa um window pequeno (250m × 250m) à volta do GPS
e verifica a cobertura real de nuvens local — não a média do tile STAC.

Uso:
  python local_cloud_analysis.py --lat 37.05815 --lon -8.20982 --depth 16
  python local_cloud_analysis.py --lat 37.05815 --lon -8.20982 \
    --dates 2025-09-15 2025-09-25 2024-09-30 2025-09-02 2022-09-26
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
import logging
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("LocalCloud")

# ── Defaults ────────────────────────────────────────────────────────────────────
SITE_LAT = 37.05815   # 37°03.489'N — GPS confirmed 2026-06-22
SITE_LON = -8.20982   # 008°12.589'W — GPS confirmed 2026-06-22
DEPTH = 16.0
BUFFER_M = 250          # 250m radius = 500m × 500m window
TOP_K = 5

# Best dates from hybrid orchestrator (hardcoded for speed)
DEFAULT_DATES = [
    "2025-09-15",
    "2025-09-25",
    "2024-09-30",
    "2025-09-02",
    "2022-09-26",
]


def search_s2_by_date(lat, lon, date_str):
    """Find a single Sentinel-2 L2A scene for a specific date."""
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    target_date = pd.to_datetime(date_str).date()
    next_day = target_date + pd.Timedelta(days=1)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{target_date.isoformat()}/{next_day.isoformat()}",
        query={"eo:cloud_cover": {"lt": 50}}  # lenient — we filter locally
    )

    items = list(search.items())
    if not items:
        return None

    # Pick least cloudy
    return min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))


def analyze_local_window(item, lat, lon, buffer_m=250):
    """
    Download and analyze a 500m×500m window around lat/lon.
    Returns dict with local cloud/coverage metrics.
    """
    assets_needed = ["B02", "B03", "B08"]  # B09 not available as COG

    b02_href = item.assets["B02"].href
    b03_href = item.assets["B03"].href
    b08_href = item.assets["B08"].href

    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

    results = {"date": item.datetime.strftime("%Y-%m-%d")}

    with env:
        # ── B02 blue band ──────────────────────────────────────────────────────
        with rasterio.open(b02_href) as src:
            tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = tf.transform(lon, lat)
            window = from_bounds(
                x - buffer_m, y - buffer_m,
                x + buffer_m, y + buffer_m,
                src.transform
            )
            b02_arr = src.read(1, window=window).astype(np.float32)
            rio_profile = src.profile
            rio_crs = src.crs

        # ── B03 green band ─────────────────────────────────────────────────────
        with rasterio.open(b03_href) as src:
            b03_arr = src.read(1, window=window).astype(np.float32)

        # ── B08 NIR band ──────────────────────────────────────────────────────
        with rasterio.open(b08_href) as src:
            b08_arr = src.read(1, window=window).astype(np.float32)

        # ── B09 cirrus band (not available as COG in this STAC) ────────────────
        b09_arr = None

    # L2A DN → reflectance
    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03 = np.clip(b03_arr / 10000.0, 0, 1.5)
    b08 = np.clip(b08_arr / 10000.0, 0, 1.5)
    b09 = np.clip(b09_arr / 10000.0, 0, 1.5) if b09_arr is not None else None

    # ── Check nodata ──────────────────────────────────────────────────────────
    if b02.max() == 0 or np.all(np.isnan(b02)):
        log.warning("All-zero or NaN window — likely nodata region")
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # CLOUD DETECTION (local, not STAC metadata)
    # ══════════════════════════════════════════════════════════════════════════

    # Method 1: Bright pixel on B02 — most reliable for cloud vs water
    # Clouds: B02 > 0.12 (bright white/grey)
    # Clear water: B02 < 0.08 even in shallow reef areas
    cloud_mask_b02 = b02 > 0.12
    cloud_pct_b02 = float(cloud_mask_b02.mean())

    # Method 2: B08 (NIR) excess as secondary check (water-safe)
    # Deep clear water: B08 very low; shallow reef with sand: B08 moderate
    # Clouds: B08 high + B02 also high (bright in all bands)
    cloud_pct_b08_water_safe = float(((b08 > 0.12) & (b02 > 0.10)).mean())

    # Combined local cloud cover — use B02 + B08 water-safe as check
    local_cloud_pct = max(cloud_pct_b02, cloud_pct_b08_water_safe)

    results["local_cloud_pct"] = local_cloud_pct * 100
    results["cloud_pct_b02"] = cloud_pct_b02 * 100
    results["cloud_pct_nir"] = cloud_pct_b08_water_safe * 100
    results["stac_cloud_cover"] = item.properties.get("eo:cloud_cover", -1)

    # ── Water quality metrics ──────────────────────────────────────────────────
    # Sunglint correction
    p95_b02 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
    p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
    b02_corr = np.clip(b02 - 0.8 * p95_b02 * 0.05, 0, 1.0)
    b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)

    # SNR local
    snr_map = make_snr_map(b02_corr, window=5)
    snr_mean = float(np.nanmean(snr_map))
    results["snr_mean"] = snr_mean

    # Kd local (band ratio)
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, 0.045)
    results["kd_mean"] = float(kd_est)

    # FFT calmness
    f_transform = np.fft.fft2(b02_corr)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    h, w = b02_corr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask_low = (xx - cx)**2 + (yy - cy)**2 <= 5**2
    mask_high = (xx - cx)**2 + (yy - cy)**2 >= 15**2
    low_power = np.sum(power[mask_low])
    high_power = np.sum(power[mask_high])
    fft_cleanliness = float(low_power / (high_power + 1e-12))
    results["fft_cleanliness"] = fft_cleanliness

    # Edge entropy (benthic structure)
    b02_blur = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    sobelx = cv2.Sobel(b02_blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(b02_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) /
                     (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0
    results["edge_entropy"] = edge_entropy

    # Signal level (for sunglint/fog detection)
    raw_mean = float(np.mean(b02[b02 > 0])) if np.any(b02 > 0) else 0
    results["raw_mean"] = raw_mean
    results["b02_mean"] = float(np.nanmean(b02_corr))
    results["b03_mean"] = float(np.nanmean(b03_corr))

    # ── Water-leaving reflectance sanity ──────────────────────────────────────
    # In clear water, B02 should be > B03? No — at depth, B02 attenuates faster
    # B02/B03 ratio tells us about in-water scattering
    valid = (b02 > 0) & (b03 > 0)
    if np.any(valid):
        ratio = b02[valid] / b03[valid]
        results["b02_b03_ratio_mean"] = float(np.nanmean(ratio))
    else:
        results["b02_b03_ratio_mean"] = np.nan

    return results


def run(lat, lon, depth, dates, buffer_m, top_k):
    log.info("=" * 72)
    log.info("  LOCAL CLOUD ANALYSIS — Pedra do Alto validation")
    log.info("  GPS: %.5f°N %.5f°W | Buffer: ±%dm | Depth: %.1fm",
             lat, abs(lon), buffer_m, depth)
    log.info("  Dates to validate: %s", dates)
    log.info("=" * 72)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    all_results = []

    for date_str in dates:
        log.info("\n[%s] Searching STAC...", date_str)

        item = search_s2_by_date(lat, lon, date_str)
        if item is None:
            log.warning("  No scene found for %s", date_str)
            continue

        stac_cloud = item.properties.get("eo:cloud_cover", -1)
        log.info("  STAC cloud cover: %.1f%% | Scene ID: %s",
                 stac_cloud, item.id[-30:])

        log.info("  Downloading %dm × %dm window via VSI...", buffer_m * 2, buffer_m * 2)
        phys = analyze_local_window(item, lat, lon, buffer_m)

        if phys is None:
            log.warning("  Window analysis failed for %s", date_str)
            continue

        log.info("  ✓ Local cloud: %.1f%%  (B02=%.1f%% | NIR=%.1f%%)",
                 phys["local_cloud_pct"],
                 phys["cloud_pct_b02"],
                 phys["cloud_pct_nir"])
        log.info("  ✓ SNR=%.1f | Kd=%.4f | Clean=%.0f | Entropy=%.3f",
                 phys["snr_mean"], phys["kd_mean"],
                 phys["fft_cleanliness"], phys["edge_entropy"])
        log.info("  ✓ Signal: %.3f | B02/B03 ratio: %.3f",
                 phys["raw_mean"], phys["b02_b03_ratio_mean"])

        phys["stac_cloud_cover"] = stac_cloud
        all_results.append(phys)

    if not all_results:
        log.error("No scenes could be analyzed!")
        return

    df = pd.DataFrame(all_results)

    # ── Filter: reject scenes with > 5% local cloud cover ─────────────────────
    df["local_cloud_ok"] = df["local_cloud_pct"] <= 5.0
    df_passed = df[df["local_cloud_ok"]].copy()
    df_rejected = df[~df["local_cloud_ok"]].copy()

    # ── Score calculation (same as hybrid but using LOCAL cloud) ────────────────
    def norm(s):
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-12:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    df_passed["n_clean"] = norm(np.log10(df_passed["fft_cleanliness"].clip(lower=1)))
    df_passed["n_kd"] = norm(1.0 / df_passed["kd_mean"])
    df_passed["n_entropy"] = norm(df_passed["edge_entropy"])
    df_passed["n_snr"] = norm(df_passed["snr_mean"])
    df_passed["n_cloud"] = norm(100 - df_passed["local_cloud_pct"])  # less cloud = better

    # Signal ok penalty
    signal_ok = ((df_passed["raw_mean"] >= 0.04) &
                 (df_passed["raw_mean"] <= 0.15)).astype(float)
    signal_ok = signal_ok.replace(0, 0.2)

    df_passed["BVI"] = (
        0.30 * df_passed["n_clean"] +
        0.25 * df_passed["n_kd"] +
        0.20 * df_passed["n_entropy"] +
        0.15 * df_passed["n_snr"] +
        0.10 * df_passed["n_cloud"]
    ) * signal_ok

    df_passed = df_passed.sort_values("BVI", ascending=False).reset_index(drop=True)

    # ── Output ──────────────────────────────────────────────────────────────────
    print("\n")
    print("╔" + "═" * 80 + "╗")
    print("║  LOCAL CLOUD ANALYSIS — BEST DATES FOR PEDRA DO ALTO (LOCAL WINDOW)  ║")
    print("╚" + "═" * 80 + "╝")

    print(f"\n  GPS: {lat:.5f}°N, {abs(lon):.5f}°W | Window: {buffer_m*2}m × {buffer_m*2}m")
    print(f"  {'#':>2}  {'Date':<12} {'Local☁':>7} {'STAC☁':>7} {'SNR':>6} {'Kd':>7} "
          f"{'Clean':>9} {'Entropy':>8} {'Signal':>7} {'BVI':>6}")
    print("  " + "-" * 80)

    for i, r in df_passed.iterrows():
        if r["BVI"] >= 0.7:
            star = " ⭐⭐⭐"
        elif r["BVI"] >= 0.5:
            star = " ⭐⭐"
        elif r["BVI"] >= 0.3:
            star = " ⭐"
        else:
            star = ""
        print(
            f"  {i+1:>2}. {r['date']:<12} {r['local_cloud_pct']:>5.1f}% "
            f"{r['stac_cloud_cover']:>5.1f}% {r['snr_mean']:>5.1f} "
            f"{r['kd_mean']:>6.4f} {r['fft_cleanliness']:>8.0f} "
            f"{r['edge_entropy']:>7.3f} {r['raw_mean']:>6.3f} {r['BVI']:>5.3f}{star}"
        )

    if len(df_rejected) > 0:
        print("\n  ── REJECTED (>5%% local cloud) ──")
        for i, r in df_rejected.iterrows():
            print(f"  ✗  {r['date']:<12} Local: {r['local_cloud_pct']:>5.1f}%  "
                  f"STAC: {r['stac_cloud_cover']:>5.1f}%  SNR: {r['snr_mean']:.1f}  "
                  f"Signal: {r['raw_mean']:.3f}")

    if df_passed.empty:
        print("\n  ⚠️  NO SCENES PASSED LOCAL CLOUD FILTER (local_cloud_pct > 5%)")
        print("  ── This means the target GPS point had clouds on ALL candidate dates.")
        print("  ── Possible causes:")
        print("     1. Actual cloud cover at the exact GPS location (local conditions)")
        print("     2. Sensor pixel containing shallow-water bright features (not clouds)")
        print("     3. The B02 > 0.12 threshold may be too aggressive for shallow reef")
        print("  ── Try: increase --buffer to sample larger area, or raise cloud threshold")
        print(f"\n  Top rejected scene: {df_rejected.iloc[0]['date']} "
              f"(local {df_rejected.iloc[0]['local_cloud_pct']:.1f}% | STAC {df_rejected.iloc[0]['stac_cloud_cover']:.1f}%)")
        return df_passed

    # Best recommendation
    best = df_passed.iloc[0]
    print(f"\n  ══ BEST LOCAL DATE: {best['date']} ══")
    print(f"     BVI: {best['BVI']:.3f} | Local cloud: {best['local_cloud_pct']:.1f}% "
          f"(STAC: {best['stac_cloud_cover']:.1f}%)")
    print(f"     SNR: {best['snr_mean']:.1f} | FFT Clean: {best['fft_cleanliness']:.0f} "
          f"| Kd: {best['kd_mean']:.4f}")
    print(f"     Signal: {best['raw_mean']:.3f} | B02/B03: {best['b02_b03_ratio_mean']:.3f}")
    print(f"\n     ⚠️  STAC vs Local cloud delta: "
          f"{best['stac_cloud_cover'] - best['local_cloud_pct']:+.1f}% "
          f"({'overestimated' if best['stac_cloud_cover'] > best['local_cloud_pct'] else 'underestimated'})")

    # Save CSV
    out_csv = f"local_cloud_pedra_alto_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
    df_passed.sort_values("BVI", ascending=False).to_csv(out_csv, index=False)
    log.info("Saved: %s", out_csv)

    return df_passed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Local cloud validation for top reef dates")
    p.add_argument("--lat", type=float, default=SITE_LAT)
    p.add_argument("--lon", type=float, default=SITE_LON)
    p.add_argument("--depth", type=float, default=DEPTH)
    p.add_argument("--buffer", type=int, default=BUFFER_M,
                   help="Buffer in meters around GPS point (default 250 = 500m window)")
    p.add_argument("--dates", nargs="+", default=DEFAULT_DATES,
                   help="Dates to analyze (default: top 5 from hybrid orchestrator)")
    args = p.parse_args()

    run(args.lat, args.lon, args.depth, args.dates, args.buffer, TOP_K)