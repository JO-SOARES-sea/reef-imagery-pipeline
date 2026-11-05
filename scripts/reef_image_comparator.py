#!/usr/bin/env python3
"""
reef_image_comparator.py
Multi-date satellite image comparison tool for reef visibility assessment.

Downloads Sentinel-2 images for multiple dates, enhances them, detects reef
contours, and produces a comparative visualization showing which dates
provide the best underwater reef visibility.

Usage:
  python scripts/reef_image_comparator.py
  python scripts/reef_image_comparator.py --dates 2025-09-25 2023-03-15 2025-09-15
  python scripts/reef_image_comparator.py --lat 37.068978 --lon -8.210328 --buffer 500
"""
import sys, os, json, warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd, cv2
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
from skimage.restoration import denoise_nl_means, estimate_sigma

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio

SITE_LAT = 37.068978
SITE_LON = -8.210328
BUFFER_M = 500
OUT_DIR = "outputs/reef_comparator"


def download_bands(lat, lon, date_str, buffer_m=500):
    """Download B02, B03, B08 bands for a given date and location."""
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    target = pd.to_datetime(date_str).date()
    next_day = target + pd.Timedelta(days=1)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{target.isoformat()}/{next_day.isoformat()}",
        query={"eo:cloud_cover": {"lt": 50}},
    )
    items = list(search.items())
    if not items:
        return None
    item = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    stac_cloud = item.properties.get("eo:cloud_cover", -1)
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    result = {"date": date_str, "stac_cloud": stac_cloud, "item_id": item.id}
    try:
        with env:
            with rasterio.open(item.assets["B02"].href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                window = from_bounds(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m, src.transform)
                b02_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(item.assets["B03"].href) as src:
                b03_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(item.assets["B08"].href) as src:
                b08_arr = src.read(1, window=window).astype(np.float32)
    except Exception as e:
        print(f"    Error downloading {date_str}: {e}")
        return None
    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03 = np.clip(b03_arr / 10000.0, 0, 1.5)
    b08 = np.clip(b08_arr / 10000.0, 0, 1.5)
    if b02.max() == 0 or np.all(np.isnan(b02)):
        return None
    result["b02"] = b02
    result["b03"] = b03
    result["b08"] = b08
    return result


def check_local_cloud(b02, b08):
    """Water-safe local cloud detection."""
    cloud_b02 = float((b02 > 0.18).mean()) * 100
    cloud_b08_safe = float(((b08 > 0.12) & (b02 > 0.12)).mean()) * 100
    return max(cloud_b02, cloud_b08_safe)


def enhance_image(b02):
    """Enhance B02: sunglint removal -> NLM denoising -> CLAHE."""
    p95 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
    b02_corr = np.clip(b02 - 0.8 * p95 * 0.05, 0, 1.0)
    sigma_est = np.mean(estimate_sigma(b02_corr))
    b02_denoised = denoise_nl_means(b02_corr, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    b02_16 = np.clip(b02_denoised * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(4, 4))
    b02_clahe = clahe.apply(b02_16).astype(np.float32) / 65535.0
    b02_enhanced = b02_denoised * 0.5 + b02_clahe * 0.5
    return b02_corr, b02_denoised, b02_enhanced


def detect_reef_contours(b02_enhanced, b03_corr, b08):
    """Multi-method reef contour detection."""
    h, w = b02_enhanced.shape
    macro = cv2.GaussianBlur(b02_enhanced, (9, 9), 0)
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_norm = sobel_mag / (sobel_mag.max() + 1e-12)

    b02_fine = cv2.GaussianBlur(b02_enhanced, (5, 5), 0)
    laplacian = cv2.Laplacian(b02_fine, cv2.CV_32F)
    lap_abs = np.abs(laplacian)
    lap_norm = lap_abs / (lap_abs.max() + 1e-12)

    b02_uint8 = np.clip(b02_enhanced * 255 / (b02_enhanced.max() + 1e-12), 0, 255).astype(np.uint8)
    median_val = np.median(b02_uint8[b02_uint8 > 0]) if np.any(b02_uint8 > 0) else 128
    low = int(max(0, 0.5 * median_val))
    high = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(b02_uint8, low, high)
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    valid = (b02_enhanced > 0.01) & (b03_corr > 0.01)
    ratio_map = np.zeros_like(b02_enhanced)
    ratio_map[valid] = b02_enhanced[valid] / b03_corr[valid]

    b02_subsurface = np.clip(b02_enhanced - 0.6 * b08, 0, 1.0)
    sub_uint8 = np.clip(b02_subsurface * 255 / (b02_subsurface.max() + 1e-12), 0, 255).astype(np.uint8)
    sub_edges = cv2.Canny(sub_uint8, low, high)
    sub_dilated = cv2.dilate(sub_edges, kernel, iterations=1)

    combined = 0.4 * sobel_norm + 0.3 * lap_norm + 0.3 * (edges_dilated.astype(np.float32) / 255)
    combined = combined / (combined.max() + 1e-12)
    contour_mask = np.maximum(edges_dilated, sub_dilated)

    return {
        "sobel_norm": sobel_norm, "laplacian": lap_norm, "edges": edges_dilated,
        "subsurface_edges": sub_dilated, "contour_mask": contour_mask,
        "ratio_map": ratio_map, "combined_edge": combined,
    }


def compute_metrics(b02, b03, b08, b02_corr, b02_enhanced):
    """Compute all quality metrics for a single date."""
    snr_map = make_snr_map(b02_enhanced, window=5)
    snr_mean = float(np.nanmean(snr_map))
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03, 0.045)

    f_transform = np.fft.fft2(b02_corr)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    h, w = b02_corr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    fft_clean = float(np.sum(power[(xx - cx)**2 + (yy - cy)**2 <= 25]) /
                      (np.sum(power[(xx - cx)**2 + (yy - cy)**2 >= 225]) + 1e-12))

    macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0

    laplacian = cv2.Laplacian(macro, cv2.CV_32F)
    benthic_contrast = float(np.var(laplacian)) * 1e6 + float(np.mean(grad_mag)) * 100
    local_cloud = check_local_cloud(b02, b08)
    raw_mean = float(np.mean(b02[b02 > 0])) if np.any(b02 > 0) else 0

    # Sunglint-correct B03
    p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
    b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)

    # Ratio stats (B02/B03)
    valid = (b02_corr > 0.01) & (b03_corr > 0.01)
    if np.any(valid):
        ratio_vals = b02_corr[valid] / b03_corr[valid]
        ratio_mean = float(np.mean(ratio_vals))
        ratio_std = float(np.std(ratio_vals))
    else:
        ratio_mean = 1.0
        ratio_std = 0.0

    # Dynamic range
    p1 = float(np.percentile(b02_corr[b02_corr > 0], 1)) if np.any(b02_corr > 0) else 0
    p99 = float(np.percentile(b02_corr[b02_corr > 0], 99)) if np.any(b02_corr > 0) else 0
    dyn_range = p99 - p1

    # Subsurface signal variation (NIR-subtracted)
    b02_sub = np.clip(b02_corr - 0.6 * b08, 0, 1.0)
    subsurf_std = float(np.std(b02_sub[b02_sub > 0])) if np.any(b02_sub > 0) else 0

    return {
        "snr": snr_mean, "kd": kd_est, "fft_clean": fft_clean,
        "edge_entropy": edge_entropy, "benthic_contrast": benthic_contrast,
        "local_cloud": local_cloud, "signal": raw_mean,
        "ratio_mean": ratio_mean, "ratio_std": ratio_std,
        "dyn_range": dyn_range, "subsurf_std": subsurf_std,
    }


def compute_bvi(metrics_list):
    """Compute BVI using trained weights (loaded from models/bvi_weights.json)."""
    weights_path = Path("models/bvi_weights.json")
    if weights_path.exists():
        with open(weights_path) as f:
            data = json.load(f)
        weights = data["weights"]
    else:
        weights = {
            "fft_clean": 0.35, "edge_entropy": 0.25, "dyn_range": 0.20,
            "snr": 0.10, "benthic_contrast": 0.05, "signal": 0.05,
        }

    df = pd.DataFrame(metrics_list)
    feature_cols = ["benthic_contrast", "snr", "fft_clean", "edge_entropy", "dyn_range", "signal"]

    # Min-max normalize to [0,1] (robust to outliers unlike z-score)
    for col in feature_cols:
        mn, mx = df[col].min(), df[col].max()
        if mx - mn > 1e-12:
            df[f"n_{col}"] = (df[col] - mn) / (mx - mn)
        else:
            df[f"n_{col}"] = 0.5

    # Weighted sum
    weight_vec = np.array([weights.get(f, 0) for f in feature_cols])
    norm_cols = [f"n_{f}" for f in feature_cols]
    X_norm = df[norm_cols].values
    scores = X_norm @ weight_vec

    # Normalize to [0,1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-12:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = np.full_like(scores, 0.5)

    return scores_norm.tolist()


def create_comparison_figure(all_data, lat, lon, out_path):
    """Multi-panel comparison figure with reef contour overlays."""
    n_dates = len(all_data)
    if n_dates == 0:
        return

    all_data = sorted(all_data, key=lambda d: d["metrics"].get("bvi", 0), reverse=True)

    reef_cmap = LinearSegmentedColormap.from_list(
        "reef", ["#000022", "#001155", "#003388", "#0066aa", "#00aacc", "#44ddaa", "#aaffaa"])
    edge_cmap = LinearSegmentedColormap.from_list(
        "edge", ["#000000", "#001133", "#0044aa", "#00aaff", "#44ffcc", "#ffffff"])

    fig = plt.figure(figsize=(22, 5.5 * n_dates + 2), facecolor="#0a0a1a")
    fig.suptitle(
        f"REEF IMAGE COMPARATOR — Pedra de Santa Eulalia\n"
        f"{lat:.6f}N, {abs(lon):.6f}W | {n_dates} dates | "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=14, fontweight="bold", color="white", y=0.99)

    outer_gs = gridspec.GridSpec(n_dates, 1, hspace=0.35, top=0.93, bottom=0.03, left=0.04, right=0.97)

    for idx, data in enumerate(all_data):
        m = data["metrics"]
        contours = data["contours"]
        b02_enhanced = data["b02_enhanced"]
        ratio_map = contours["ratio_map"]
        combined_edge = contours["combined_edge"]

        bvi = m.get("bvi", 0)
        star = "***" if bvi >= 0.7 else "**" if bvi >= 0.5 else "*" if bvi >= 0.3 else ""
        local_cloud = m["local_cloud"]

        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[idx], wspace=0.15)

        p2, p98 = np.percentile(b02_enhanced[b02_enhanced > 0], [2, 98]) if np.any(b02_enhanced > 0) else (0, 1)
        img_stretch = np.clip((b02_enhanced - p2) / (p98 - p2 + 1e-12), 0, 1)

        ax1 = fig.add_subplot(inner_gs[0])
        ax1.imshow(img_stretch, cmap=reef_cmap, interpolation="bilinear")
        ax1.set_title(f"#{idx+1} {data['date']} Enhanced B02\nBVI={bvi:.3f} {star} | Cloud={local_cloud:.0f}%",
                       fontsize=10, color="white", fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(inner_gs[1])
        ax2.imshow(img_stretch * 0.4, cmap="gray", interpolation="bilinear")
        contour_rgb = np.zeros((*contours["contour_mask"].shape, 4), dtype=np.float32)
        sub_edges = contours["subsurface_edges"]
        contour_rgb[sub_edges > 0] = [0, 1, 0.8, 0.9]
        surf_edges = contours["edges"]
        contour_rgb[(surf_edges > 0) & (sub_edges == 0)] = [1, 0.9, 0, 0.7]
        ax2.imshow(contour_rgb, interpolation="nearest")
        ax2.set_title("Reef Contours\ncyan=reef | yellow=structure", fontsize=9, color="white")
        ax2.axis("off")

        ax3 = fig.add_subplot(inner_gs[2])
        ratio_display = np.ma.masked_where(ratio_map == 0, ratio_map)
        r2, r98 = np.percentile(ratio_map[ratio_map > 0], [2, 98]) if np.any(ratio_map > 0) else (0.5, 1.5)
        im3 = ax3.imshow(ratio_display, cmap="RdYlBu_r", vmin=r2, vmax=r98, interpolation="bilinear")
        plt.colorbar(im3, ax=ax3, shrink=0.7, label="B02/B03 ratio")
        ax3.set_title(f"Depth Proxy (B02/B03)\nKd={m['kd']:.4f} | SNR={m['snr']:.0f}", fontsize=9, color="white")
        ax3.axis("off")

        ax4 = fig.add_subplot(inner_gs[3])
        ax4.imshow(combined_edge, cmap=edge_cmap, interpolation="bilinear")
        ax4.set_title(f"Edge Detection\nEntropy={m['edge_entropy']:.2f} | FFT={m['fft_clean']:.0f}",
                       fontsize=9, color="white")
        ax4.axis("off")

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")

    # BVI ranking bar chart
    dates_sorted = sorted(all_data, key=lambda d: d["metrics"].get("bvi", 0))
    fig2, ax = plt.subplots(figsize=(12, max(4, n_dates * 0.8)), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    y_labels = [d["date"] for d in dates_sorted]
    bvis = [d["metrics"]["bvi"] for d in dates_sorted]
    clouds = [d["metrics"]["local_cloud"] for d in dates_sorted]
    colors = ["#44ff88" if c < 5 else "#ffaa00" if c < 15 else "#ff4444" for c in clouds]
    bars = ax.barh(y_labels, bvis, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    for bar, bvi_val, cloud in zip(bars, bvis, clouds):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"BVI={bvi_val:.3f} (cloud={cloud:.0f}%)", va="center", ha="left", fontsize=9, color="white")
    ax.set_xlabel("Benthic Visibility Index (BVI)", color="white", fontsize=11)
    ax.set_title("Reef Image Quality Ranking\nGreen=clear | Orange=marginal | Red=cloudy at GPS",
                  color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlim(0, 1.0)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    summary_path = out_path.replace(".png", "_ranking.png")
    fig2.savefig(summary_path, dpi=150, facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close(fig2)
    print(f"    Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Reef image comparator with contour detection")
    parser.add_argument("--lat", type=float, default=SITE_LAT)
    parser.add_argument("--lon", type=float, default=SITE_LON)
    parser.add_argument("--buffer", type=int, default=BUFFER_M)
    parser.add_argument("--dates", nargs="+", default=[
        "2023-03-15", "2025-03-29", "2023-09-01", "2025-09-15",
        "2026-02-22", "2025-10-05", "2025-09-25", "2024-09-30",
    ])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    lat, lon, buffer_m = args.lat, args.lon, args.buffer
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  REEF IMAGE COMPARATOR — Multi-Date Analysis with Contour Detection")
    print(f"  {lat:.6f}N, {abs(lon):.6f}W | Buffer: +/-{buffer_m}m")
    print(f"  Dates: {args.dates}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    all_data = []
    for i, date_str in enumerate(args.dates):
        print(f"\n[{i+1}/{len(args.dates)}] Processing {date_str}...")
        raw = download_bands(lat, lon, date_str, buffer_m)
        if raw is None:
            print("    SKIP - no scene found")
            continue
        b02, b03, b08 = raw["b02"], raw["b03"], raw["b08"]
        local_cloud = check_local_cloud(b02, b08)
        print(f"    STAC cloud: {raw['stac_cloud']:.1f}% | Local cloud at GPS: {local_cloud:.1f}%")
        if local_cloud > 50:
            print(f"    SKIP - too cloudy at GPS ({local_cloud:.1f}%)")
            continue

        b02_corr, b02_denoised, b02_enhanced = enhance_image(b02)
        p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
        b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)
        contours = detect_reef_contours(b02_enhanced, b03_corr, b08)
        metrics = compute_metrics(b02, b03, b08, b02_corr, b02_enhanced)

        print(f"    SNR={metrics['snr']:.1f} | Kd={metrics['kd']:.4f} | FFT={metrics['fft_clean']:.0f} | Entropy={metrics['edge_entropy']:.2f}")

        all_data.append({
            "date": date_str, "b02": b02, "b03": b03, "b08": b08,
            "b02_corr": b02_corr, "b02_enhanced": b02_enhanced,
            "contours": contours, "metrics": metrics, "stac_cloud": raw["stac_cloud"],
        })

    if not all_data:
        print("\nNo valid scenes found!")
        sys.exit(1)

    bvis = compute_bvi([d["metrics"] for d in all_data])
    for d, bvi in zip(all_data, bvis):
        d["metrics"]["bvi"] = bvi
    all_data.sort(key=lambda d: d["metrics"]["bvi"], reverse=True)

    print(f"\n{'='*80}")
    print("  BVI RANKING")
    print(f"{'='*80}")
    print(f"\n  {'#':>2}  {'Date':<12} {'BVI':>5} {'Local':>6} {'STAC':>6} {'SNR':>6} {'Kd':>7} {'FFT':>8} {'Entropy':>8}")
    print("  " + "-" * 75)
    for i, d in enumerate(all_data):
        m = d["metrics"]
        star = "***" if m["bvi"] >= 0.7 else "**" if m["bvi"] >= 0.5 else "*" if m["bvi"] >= 0.3 else ""
        print(f"  {i+1:>2}. {d['date']:<12} {m['bvi']:.3f} {m['local_cloud']:>5.1f}% {d['stac_cloud']:>5.1f}% "
              f"{m['snr']:>5.1f} {m['kd']:>6.4f} {m['fft_clean']:>7.0f} {m['edge_entropy']:>7.2f}  {star}")

    print("\n  Generating comparison figure...")
    out_path = str(out_dir / "reef_comparison.png")
    create_comparison_figure(all_data, lat, lon, out_path)

    print(f"\n{'='*80}")
    print(f"  COMPLETE: {len(all_data)} dates compared")
    print(f"  Output: {out_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
