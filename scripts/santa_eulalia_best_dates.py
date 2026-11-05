#!/usr/bin/env python3
"""
santa_eulalia_best_dates.py
══════════════════════════════════════════════════════════════════════════════════
BVI (Benthic Visibility Index) ranking for Pedra de Santa Eulália reef.

Key improvement over STAC-only ranking:
  → Checks LOCAL cloud cover at the GPS point (not tile-level STAC metadata)
  → Uses B02 + B08 water-safe cloud detection to avoid misclassifying
    shallow reef/sand as cloud
  → Only penalizes images that have confirmed cloud AT the reef location
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd, cv2
from datetime import datetime

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio
from src.ranking_model import predict_score

SITE_LAT = 37.069081
SITE_LON = -8.210242
DEPTH = 12
BUFFER_M = 500
YEARS = 8
MAX_CLOUD_STAC = 20
LOCAL_CLOUD_REJECT = 15.0
TOP_N = 25


def check_local_cloud(b02, b08=None):
    """
    Water-safe local cloud detection at GPS window.
    Returns local_cloud_pct (0-100).

    Key: shallow reef and sand have B02 ~0.06-0.12 but B08 is low.
    Clouds have BOTH B02 > 0.12 AND B08 high.
    This prevents false-positive cloud detection over clear shallow reef.
    """
    # Method 1: Bright pixels in B02 (threshold 0.18 — conservative for reef)
    cloud_b02 = b02 > 0.18

    # Method 2: Water-safe — both B02 AND B08 must be high
    if b08 is not None:
        cloud_b08_safe = (b08 > 0.12) & (b02 > 0.12)
    else:
        cloud_b08_safe = cloud_b02

    # Use the stricter of the two (less false positives over reef)
    local_cloud_pct = float(max(cloud_b02.mean(), cloud_b08_safe.mean())) * 100
    return local_cloud_pct


def compute_bvi_metrics(b02_corr, b03_corr):
    """Compute all BVI component metrics from corrected reflectance."""
    # SNR
    snr_map = make_snr_map(b02_corr, window=5)
    snr_mean = float(np.nanmean(snr_map))

    # Benthic contrast (Laplacian + Sobel)
    macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    laplacian = cv2.Laplacian(macro, cv2.CV_32F)
    laplacian_var = float(np.var(laplacian))
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = float(np.mean(grad_mag))
    benthic_contrast = laplacian_var * 1e6 + sobel_mean * 100

    # FFT cleanliness (surface calmness)
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

    # Edge entropy (structural information)
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) /
                     (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0

    # Kd (water clarity from band ratio)
    kd_prior = 0.045
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, kd_prior)

    return {
        "snr": snr_mean,
        "benthic_contrast": benthic_contrast,
        "fft_cleanliness": fft_cleanliness,
        "edge_entropy": edge_entropy,
        "kd_mean": kd_est,
    }


def norm(s):
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def main():
    parser = argparse.ArgumentParser(description="BVI ranking with local cloud check")
    parser.add_argument("--lat", type=float, default=SITE_LAT)
    parser.add_argument("--lon", type=float, default=SITE_LON)
    parser.add_argument("--depth", type=float, default=DEPTH)
    parser.add_argument("--buffer", type=int, default=BUFFER_M)
    parser.add_argument("--years", type=int, default=YEARS)
    parser.add_argument("--max-cloud-stac", type=float, default=MAX_CLOUD_STAC)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    lat, lon, depth = args.lat, args.lon, args.depth
    buffer_m = args.buffer

    print("=" * 80)
    print("  PEDRA DE SANTA EULALIA — BVI Ranking with Local Cloud Check")
    print(f"  {lat:.6f} N, {abs(lon):.6f} W | Depth: {depth}m | Buffer: +/-{buffer_m}m")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # ── STEP 1: Search Sentinel-2 via STAC ─────────────────────────────────────
    print("\n[1] Searching Sentinel-2 L2A via Planetary Computer STAC...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    end_date = datetime.now()
    start_date = datetime(end_date.year - args.years, 1, 1)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": args.max_cloud_stac}},
    )

    items = list(search.items())
    print(f"    Found {len(items)} scenes with <{args.max_cloud_stac}% STAC cloud cover")

    if not items:
        print("    No scenes found. Exiting.")
        sys.exit(1)

    # Build dataframe
    stac_data = []
    for item in items:
        props = item.properties
        if props.get("s2:nodata_pixel_percentage", 100) > 20:
            continue
        stac_data.append({
            "date_str": item.datetime.strftime("%Y-%m-%d"),
            "date": pd.Timestamp(item.datetime.date()),
            "cloud_stac": props.get("eo:cloud_cover", 100),
            "sun_elevation": props.get("view:sun_elevation", 45),
            "item": item,
        })

    df_stac = pd.DataFrame(stac_data)
    df_stac = df_stac.sort_values("cloud_stac").drop_duplicates("date_str", keep="first")
    print(f"    Unique dates: {len(df_stac)}")

    # Smart candidate selection: stratified monthly sampling
    # Instead of just top-N by STAC cloud (unreliable), pick the best N per month
    # to ensure seasonal diversity and catch dates where STAC cloud is moderate
    # but local cloud at GPS is zero
    df_stac["month"] = df_stac["date"].dt.month
    per_month = max(2, args.top_n // 12 + 1)
    candidates = (
        df_stac.sort_values("cloud_stac")
        .groupby("month", group_keys=False)
        .head(per_month)
        .head(args.top_n)
    )
    print(f"    Stratified selection: {len(candidates)} candidates across all months")

    # ── STEP 2: Download local window + compute BVI metrics ─────────────────────
    print("\n[2] Downloading local windows + computing BVI metrics...")
    print("    LOCAL cloud check: B02>0.18 OR (B02>0.12 AND B08>0.12) = cloud at GPS")

    results = []
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

    for idx, row in candidates.iterrows():
        item = row["item"]
        date_str = row["date_str"]
        sys.stdout.write(f"\r    Processing {date_str} ({len(results)+1}/{len(candidates)})...          ")
        sys.stdout.flush()

        try:
            with env:
                with rasterio.open(item.assets["B02"].href) as src:
                    tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = tf.transform(lon, lat)
                    window = from_bounds(
                        x - buffer_m, y - buffer_m,
                        x + buffer_m, y + buffer_m,
                        src.transform,
                    )
                    b02_arr = src.read(1, window=window).astype(np.float32)

                with rasterio.open(item.assets["B03"].href) as src:
                    b03_arr = src.read(1, window=window).astype(np.float32)

                b08_arr = None
                if "B08" in item.assets:
                    with rasterio.open(item.assets["B08"].href) as src:
                        b08_arr = src.read(1, window=window).astype(np.float32)
        except Exception:
            continue

        b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
        b03 = np.clip(b03_arr / 10000.0, 0, 1.5)
        b08 = np.clip(b08_arr / 10000.0, 0, 1.5) if b08_arr is not None else None

        if b02.max() == 0 or np.all(np.isnan(b02)):
            continue

        # ── LOCAL CLOUD CHECK (key improvement) ──────────────────────────────
        local_cloud_pct = check_local_cloud(b02, b08)

        # Sunglint correction
        p95_b02 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
        p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
        b02_corr = np.clip(b02 - 0.8 * p95_b02 * 0.05, 0, 1.0)
        b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)

        # BVI metrics
        metrics = compute_bvi_metrics(b02_corr, b03_corr)

        raw_mean = float(np.mean(b02[b02 > 0])) if np.any(b02 > 0) else 0
        dyn_range = float(np.max(b02_corr) - np.min(b02_corr)) if np.any(b02_corr) else 0.008

        # Call the canonical ML ranking model / expert B02 model
        features_for_ranking = {
            "benthic_contrast": metrics["benthic_contrast"],
            "snr": metrics["snr"],
            "fft_clean": metrics["fft_cleanliness"],
            "edge_entropy": metrics["edge_entropy"],
            "dyn_range": dyn_range,
            "signal": raw_mean,
            "cloud_cover": local_cloud_pct,
        }
        prediction = predict_score(features_for_ranking)

        results.append({
            "date": date_str,
            "cloud_stac": row["cloud_stac"],
            "local_cloud_pct": local_cloud_pct,
            "snr": metrics["snr"],
            "benthic_contrast": metrics["benthic_contrast"],
            "fft_cleanliness": metrics["fft_cleanliness"],
            "edge_entropy": metrics["edge_entropy"],
            "kd_mean": metrics["kd_mean"],
            "raw_mean": raw_mean,
            "dyn_range": dyn_range,
            "ml_score": prediction["score"],
            "ml_mode": prediction["mode"],
        })

    print(f"\r    Processed {len(results)} scenes successfully.                    ")

    if not results:
        print("    No scenes could be processed!")
        sys.exit(1)

    # ── STEP 3: BVI scoring with LOCAL cloud penalty ────────────────────────────
    print("\n[3] Computing BVI with local cloud penalty...")

    df = pd.DataFrame(results)

    # Mark local cloud status
    df["local_cloud_ok"] = df["local_cloud_pct"] <= LOCAL_CLOUD_REJECT

    n_ok = df["local_cloud_ok"].sum()
    n_rejected = (~df["local_cloud_ok"]).sum()
    print(f"    Passed local cloud filter ({LOCAL_CLOUD_REJECT}%): {n_ok}")
    print(f"    Rejected (cloud at GPS): {n_rejected}")

    # Split
    df_ok = df[df["local_cloud_ok"]].copy()
    df_bad = df[~df["local_cloud_ok"]].copy()

    if df_ok.empty:
        print("\n  ALL scenes have cloud at GPS point!")
        print("  Lowering threshold to 30% to show best available...")
        df_ok = df[df["local_cloud_pct"] <= 30].copy()
        if df_ok.empty:
            df_ok = df.copy()

    # Use canonical absolute ML BVI score directly
    df_ok["BVI"] = df_ok["ml_score"]

    df_ok = df_ok.sort_values("BVI", ascending=False).reset_index(drop=True)

    # ── STEP 4: Results ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  CANONICAL ML BVI RANKING — Pedra de Santa Eulalia")
    print(f"  {lat:.6f} N, {abs(lon):.6f} W | Depth: {depth}m")
    print(f"{'='*80}")

    print(f"\n  {'#':>2}  {'Date':<12} {'BVI':>5} {'Mode':<8} {'Local':>6} {'STAC':>6} {'SNR':>6} "
          f"{'Contrast':>9} {'Clean':>9} {'Entropy':>8} {'Kd':>7} {'Signal':>7}")
    print("  " + "-" * 104)

    for i, r in df_ok.iterrows():
        if r["BVI"] >= 0.7:
            star = " ***"
        elif r["BVI"] >= 0.5:
            star = " **"
        elif r["BVI"] >= 0.3:
            star = " *"
        else:
            star = ""

        print(
            f"  {i+1:>2}. {r['date']:<12} {r['BVI']:.3f} {r['ml_mode']:<8} {r['local_cloud_pct']:>5.1f}% "
            f"{r['cloud_stac']:>5.1f}% {r['snr']:>5.1f} "
            f"{r['benthic_contrast']:>8.1f} {r['fft_cleanliness']:>8.0f} "
            f"{r['edge_entropy']:>7.3f} {r['kd_mean']:>6.4f} {r['raw_mean']:>6.3f}{star}"
        )

    if len(df_bad) > 0:
        print(f"\n  REJECTED (cloud at GPS > {LOCAL_CLOUD_REJECT}%):")
        for _, r in df_bad.iterrows():
            delta = r["cloud_stac"] - r["local_cloud_pct"]
            print(f"  X  {r['date']:<12} Local: {r['local_cloud_pct']:>5.1f}%  "
                  f"STAC: {r['cloud_stac']:>5.1f}%  (delta: {delta:+.1f}%)  "
                  f"SNR: {r['snr']:.1f}  Kd: {r['kd_mean']:.4f}")

    # Best day
    if len(df_ok) > 0:
        best = df_ok.iloc[0]
        print(f"\n  BEST DATE: {best['date']}")
        print(f"     BVI:             {best['BVI']:.3f}")
        print(f"     Ranker Mode:     {best['ml_mode']}")
        print(f"     Local cloud:     {best['local_cloud_pct']:.1f}% (STAC: {best['cloud_stac']:.1f}%)")
        print(f"     SNR:             {best['snr']:.1f}")
        print(f"     Kd:              {best['kd_mean']:.4f}")
        print(f"     FFT Cleanliness: {best['fft_cleanliness']:.0f}")
        print(f"     Benthic Contrast:{best['benthic_contrast']:.1f}")
        print(f"     Edge Entropy:    {best['edge_entropy']:.3f}")
        print(f"     Signal:          {best['raw_mean']:.3f}")

    # Top 5
    print("\n  TOP 5 RECOMMENDED:")
    for i in range(min(5, len(df_ok))):
        r = df_ok.iloc[i]
        print(f"     {i+1}. {r['date']} — BVI={r['BVI']:.3f} | Mode={r['ml_mode']} | "
              f"LocalCloud={r['local_cloud_pct']:.1f}% | Kd={r['kd_mean']:.4f}")

    # Save CSV
    out_csv = f"local_cloud_pedra_alto_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
    df_ok.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    print(f"{'='*80}")

    return df_ok


if __name__ == "__main__":
    main()
