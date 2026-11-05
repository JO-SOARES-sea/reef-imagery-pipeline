#!/usr/bin/env python3
"""
zimbral_best_visibility.py
══════════════════════════════════════════════════════════════════════════════
Historical best-visibility analysis for Zimbral Reef (Algarve coast).
Searches 2018-2025 for the clearest Sentinel-2 scene per year, scores each
with the pipeline BVI model, and ranks all years by underwater reef visibility.

What it does:
  1. Queries Element84 STAC for the lowest cloud-cover summer scene per year
     (extends to spring/autumn if summer yields nothing clean enough)
  2. Downloads B02/B03/B04/B11 bands via COG VSI reads (no full-scene download)
  3. Applies the full correction chain:
       SZA normalisation → Hedley glint (SWIR B11) → NL-Means denoising
       → FFT wave filter → CLAHE → Stumpf B02/B03 ratio
  4. Computes BVI features and scores each scene with the trained RF model
  5. Saves per-year depth-proxy PNGs + a composite 3×3 grid
  6. Saves a ranked CSV and a trend chart
  7. Writes the best year + date back to dashboard_layers.json as
     "zimbral_best_visibility" so the dashboard can display it

Usage:
    cd /path/to/reef_imagery_pipeline
    source .venv/bin/activate
    python scripts/zimbral_best_visibility.py

Outputs (in outputs/zimbral_temporal/):
    zimbral_<YYYY>_depth_proxy.png   — one image per year
    zimbral_temporal_grid.png        — 3×3 comparison grid
    zimbral_trend.png                — BVI trend line 2018-2025
    zimbral_visibility_ranked.csv    — ranked table, best year first
    zimbral_best_<YYYY>_<MMDD>_*.png — full render of the single best scene
"""

import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pystac_client import Client

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import planetary_computer as pc
    _PC_AVAILABLE = True
except ImportError:
    _PC_AVAILABLE = False

from ranking_model import predict_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Site definition ───────────────────────────────────────────────────────────
ZIMBRAL = {
    "lat": 36.964,
    "lon": -7.936,
    "depth_m": 12,
    "label": "Zimbral Reef",
    "buffer_m": 700,   # 1.4 km × 1.4 km patch
}

YEARS        = list(range(2018, 2026))
OUT_DIR      = _PROJECT_ROOT / "outputs" / "zimbral_temporal"
CACHE_DIR    = OUT_DIR / "_cache"
DPI          = 300
DASHBOARD_JSON = _PROJECT_ROOT / "dashboard" / "dashboard_layers.json"

# ── Colormaps ─────────────────────────────────────────────────────────────────
blue_depth_cmap = LinearSegmentedColormap.from_list(
    "blue_depth", ["#000814", "#001d3d", "#023e8a", "#0077b6", "#0096c7", "#48cae4", "#ade8f4"]
)
green_reef_cmap = LinearSegmentedColormap.from_list(
    "green_reef", ["#001100", "#002d08", "#004d1b", "#007a33", "#22b15b", "#6ce08c", "#baffd0"]
)

# ── STAC helpers ─────────────────────────────────────────────────────────────

STAC_PROVIDERS = [
    {
        "name": "Element84",
        "url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "modifier": None,
    },
]
if _PC_AVAILABLE:
    STAC_PROVIDERS.insert(0, {
        "name": "Planetary Computer",
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        "modifier": pc.sign_inplace,
    })


def _stac_search(lat, lon, date_range_str, cloud_lt=5.0):
    """Try all STAC providers; return (items, provider_dict) or ([], None)."""
    for provider in STAC_PROVIDERS:
        kwargs = {}
        if provider["modifier"]:
            kwargs["modifier"] = provider["modifier"]
        for attempt in range(1, 4):
            try:
                cat = Client.open(provider["url"], **kwargs)
                search = cat.search(
                    collections=[provider["collection"]],
                    intersects={"type": "Point", "coordinates": [lon, lat]},
                    datetime=date_range_str,
                    query={"eo:cloud_cover": {"lt": cloud_lt}},
                )
                items = list(search.items())
                if items:
                    return items, provider
            except Exception as exc:
                wait = 2 ** attempt
                log.warning("STAC attempt %d/3 (%s): %s", attempt, provider["name"], exc)
                if attempt < 3:
                    time.sleep(wait)
    return [], None


def find_best_scene_for_year(year: int, lat: float, lon: float):
    """
    Return (date_str, item, provider) for the clearest Sentinel-2 scene
    in the summer window, progressively relaxing constraints if needed.
    """
    windows = [
        (f"{year}-07-01/{year}-09-30", 5.0),
        (f"{year}-07-01/{year}-09-30", 20.0),
        (f"{year}-06-01/{year}-10-31", 30.0),
        (f"{year}-04-01/{year}-10-31", 40.0),
    ]
    for date_range, cloud_lt in windows:
        items, provider = _stac_search(lat, lon, date_range, cloud_lt)
        if items:
            best = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 99.0))[0]
            date_str = best.datetime.strftime("%Y-%m-%d")
            cc = best.properties.get("eo:cloud_cover", "?")
            log.info("  Year %d → %s  clouds=%.1f%%  [%s]", year, date_str, float(cc), provider["name"])
            return date_str, best, provider

    # Hard fallback — known clear dates near Zimbral (south Algarve)
    fallbacks = {
        2018: "2018-08-03", 2019: "2019-08-18", 2020: "2020-09-06",
        2021: "2021-08-22", 2022: "2022-09-02", 2023: "2023-08-28",
        2024: "2024-09-15", 2025: "2025-09-25",
    }
    date_str = fallbacks.get(year, f"{year}-08-15")
    log.warning("  Year %d: STAC failed — using fallback date %s", year, date_str)
    return date_str, None, None


# ── Image processing helpers ──────────────────────────────────────────────────

def _get_window(src, lon, lat, buffer_m):
    tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = tf.transform(lon, lat)
    inv = ~src.transform
    c0, r1 = inv * (x - buffer_m, y - buffer_m)
    c1, r0 = inv * (x + buffer_m, y + buffer_m)
    return Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))


def _asset_href(item, *names):
    for name in names:
        if name in item.assets:
            return item.assets[name].href
    raise KeyError(f"None of {names} in item assets: {list(item.assets.keys())}")


def _read_band_cog(href, window, env):
    with env:
        with rasterio.open(href) as src:
            arr = src.read(1, window=window).astype(np.float32)
    return np.clip(arr / 10_000.0, 0.0, 1.5)


def _sza_correct(arr, sza_deg):
    return arr / math.cos(math.radians(max(float(sza_deg), 5.0)))


def _hedley_glint(b_vis, b_swir):
    _pos = b_swir[b_swir > 0]
    b11_min = float(np.percentile(_pos, 1)) if _pos.size > 0 else 0.0
    glint = np.clip(b_swir - b11_min, 0.0, 1.0)
    return np.clip(b_vis - 0.9 * glint, 0.0, 1.0).astype(np.float32)


def _fft_wave_filter(arr):
    from numpy.fft import fft2, ifft2, fftshift
    F = fftshift(fft2(arr.astype(np.float64)))
    rows, cols = F.shape
    mask = np.ones((rows, cols), np.float64)
    cx = rows // 2
    half_r = max(1, rows // 80)
    mask[cx - half_r : cx + half_r, :] = 0.3
    mask[cx, cols // 2] = 1.0
    filtered = np.real(ifft2(fftshift(F * mask))).astype(np.float32)
    return np.clip(filtered, 0.0, None)


def _denoise(arr):
    try:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        sigma = float(np.mean(estimate_sigma(arr)))
        arr_dn = denoise_nl_means(arr, h=1.2 * sigma, fast_mode=True,
                                   patch_size=7, patch_distance=11)
        return cv2.GaussianBlur(arr_dn.astype(np.float32), (3, 3), sigmaX=0.8)
    except Exception:
        return arr


def _clahe(arr, clip_limit=0.5, tile=8):
    arr16 = np.clip(arr * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    enh = clahe.apply(arr16).astype(np.float32) / 65535.0
    return arr * 0.8 + enh * 0.2


def _stumpf_ratio(b02, b03, n=1000.0):
    s02 = np.clip(b02, 0.001, 1.5)
    s03 = np.clip(b03, 0.001, 1.5)
    return np.log(n * s02) / np.log(n * s03)


def _water_stretch(arr):
    valid = arr[arr > 0]
    if valid.size == 0:
        return arr
    p2, p98 = np.percentile(valid, [2, 98])
    stretched = np.clip((arr - p2) / (p98 - p2 + 1e-12), 0.0, 1.0)
    return np.power(stretched, 0.7)


# ── BVI feature extraction ────────────────────────────────────────────────────

def _bvi_features(b02_enh, ratio, cloud_pct, sza_deg):
    """
    Extract the 6-feature vector expected by predict_score.
    Scales must match training data in train_feature_ranker.py:
      snr          86–138  (mean/std of b02)
      fft_clean   704–17221 (absolute low-freq FFT energy sum)
      edge_entropy 1.98–7.29 (Shannon entropy of Sobel gradient histogram)
      dyn_range    0.005–0.014 (p98-p2 of Stumpf ratio, not raw b02)
      signal       0.118–0.141 (mean b02)
      benthic_contrast 0.1–1.1 (std of Stumpf ratio in central crop)
    """
    valid = b02_enh[b02_enh > 0]
    # snr: mean/std of b02 — training range 86-138
    snr_val = float(np.mean(valid) / (np.std(valid) + 1e-9)) if valid.size > 0 else 0.0

    # benthic_contrast: std of Stumpf ratio in central reef zone
    h, w = ratio.shape
    cy, cx = h // 2, w // 2
    crop = ratio[cy - 20 : cy + 20, cx - 20 : cx + 20]
    benthic_contrast = float(np.std(crop)) if crop.size > 0 else 0.0

    # fft_clean: absolute low-frequency FFT energy sum — training range 704-17221
    from numpy.fft import fft2, fftshift
    F = np.abs(fftshift(fft2(b02_enh.astype(np.float64))))
    rows, cols = F.shape
    r_mask = min(rows, cols) // 8
    cy_f, cx_f = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    low_mask = (Y - cy_f) ** 2 + (X - cx_f) ** 2 <= r_mask ** 2
    fft_clean = float(F[low_mask].sum())  # absolute, not fractional

    # edge_entropy: Shannon entropy of Sobel gradient magnitude histogram
    # Training range 1.98-7.29 — must use Sobel, not Canny
    sobel_x = cv2.Sobel(b02_enh.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(b02_enh.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    hist, _ = np.histogram(grad_mag.ravel(), bins=256, range=(0, grad_mag.max() + 1e-9))
    hist = hist.astype(np.float64) + 1e-12
    hist /= hist.sum()
    edge_entropy = float(-np.sum(hist * np.log(hist)))

    # dyn_range: proportional spread of centred Stumpf ratio, clamped to training range [0.005, 0.014]
    # Training range was 0.005-0.014 (2.6% feature importance — low impact).
    # Scale raw spread linearly into that range to avoid out-of-distribution drift.
    ratio_centred = np.clip(ratio - 0.9, 0.0, 0.5)
    rc_valid = ratio_centred[np.isfinite(ratio_centred) & (ratio_centred > 0)]
    raw_spread = float(np.percentile(rc_valid, 98) - np.percentile(rc_valid, 2)) if rc_valid.size > 0 else 0.15
    dyn_range = float(np.clip(raw_spread * 0.008 / 0.15, 0.005, 0.014))

    signal = float(np.mean(valid)) if valid.size > 0 else 0.0

    return {
        "benthic_contrast": benthic_contrast,
        "snr": snr_val,
        "fft_clean": fft_clean,
        "edge_entropy": edge_entropy,
        "dyn_range": dyn_range,
        "signal": signal,
        "cloud_cover": float(cloud_pct) if cloud_pct != "?" else 5.0,
    }


# ── Per-year processing ───────────────────────────────────────────────────────

def process_year(year: int, site: dict, out_dir: Path, cache_dir: Path, skip_render=True):
    """
    Download, correct, and score one year's best scene for the given site.
    Returns a result dict or None on failure.
    """
    lat, lon = site["lat"], site["lon"]
    buffer_m = site["buffer_m"]

    date_str, item, provider = find_best_scene_for_year(year, lat, lon)
    if item is None:
        log.warning("Year %d: no STAC item — skipping.", year)
        return None

    # Cache check — skip heavy processing if arrays already exist
    cache_ratio = cache_dir / f"zimbral_{year}_ratio.npy"
    cache_b02   = cache_dir / f"zimbral_{year}_b02_enh.npy"
    cache_meta  = cache_dir / f"zimbral_{year}_meta.json"

    if cache_ratio.exists() and cache_b02.exists() and cache_meta.exists():
        log.info("  Year %d: loading cached arrays.", year)
        ratio    = np.load(cache_ratio)
        b02_enh  = np.load(cache_b02)
        with open(cache_meta) as f:
            meta = json.load(f)
        cloud_pct = meta["cloud_pct"]
        sza_deg   = meta["sza_deg"]
    else:
        env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

        cloud_pct = item.properties.get("eo:cloud_cover", 5.0)
        sza_deg   = item.properties.get("s2:mean_solar_zenith", 30.0)

        try:
            b02_href = _asset_href(item, "B02", "blue")
            with env:
                with rasterio.open(b02_href) as src:
                    window = _get_window(src, lon, lat, buffer_m)
            b02 = _read_band_cog(b02_href, window, env)

            b03_href = _asset_href(item, "B03", "green")
            b03 = _read_band_cog(b03_href, window, env)

            b11_href = _asset_href(item, "B11", "swir16")
            with env:
                with rasterio.open(b11_href) as src:
                    win_b11 = _get_window(src, lon, lat, buffer_m)
            b11_raw = _read_band_cog(b11_href, win_b11, env)
            b11 = cv2.resize(b11_raw, (b02.shape[1], b02.shape[0]), interpolation=cv2.INTER_LINEAR)

        except Exception as exc:
            log.error("  Year %d: band download failed: %s", year, exc)
            return None

        # Correction chain
        b02_sza  = _sza_correct(b02, sza_deg)
        b03_sza  = _sza_correct(b03, sza_deg)
        b02_g    = _hedley_glint(b02_sza, b11)
        b03_g    = _hedley_glint(b03_sza, b11)
        b02_dn   = _denoise(b02_g)
        b03_dn   = _denoise(b03_g)
        b02_fft  = _fft_wave_filter(b02_dn)
        b02_enh  = _clahe(b02_fft)
        b03_enh  = _clahe(b03_dn)
        ratio    = _stumpf_ratio(b02_enh, b03_enh)

        # Persist cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_ratio, ratio)
        np.save(cache_b02, b02_enh)
        with open(cache_meta, "w") as f:
            json.dump({"cloud_pct": float(cloud_pct), "sza_deg": float(sza_deg),
                       "date": date_str}, f)

    # Score with BVI model
    features = _bvi_features(b02_enh, ratio, cloud_pct, sza_deg)
    score_result = predict_score(features)
    bvi_score = score_result["score"]

    # Normalised ratio for display
    ratio_norm = _water_stretch(np.clip(ratio - 0.9, 0, 0.5))

    # Optional PNG rendering
    if not skip_render:
        slug = f"zimbral_{year}_{date_str.replace('-', '')}"
        _save_img(ratio_norm, f"Zimbral · Depth Proxy (Stumpf) · {date_str}",
                  out_dir / f"{slug}_depth_proxy.png", blue_depth_cmap)
        _save_img(_water_stretch(b02_enh), f"Zimbral · B02 Enhanced · {date_str}",
                  out_dir / f"{slug}_blue_enhanced.png", blue_depth_cmap)

    return {
        "year": year,
        "date": date_str,
        "cloud_pct": float(cloud_pct),
        "sza_deg": float(sza_deg),
        "bvi_score": round(bvi_score, 4),
        "mode": score_result.get("mode", "?"),
        "ratio": ratio,
        "ratio_norm": ratio_norm,
        "b02_enh": b02_enh,
    }


def _save_img(arr, title, path, cmap, dpi=DPI):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#001108")
    ax.imshow(arr, cmap=cmap, interpolation="bilinear", vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9, pad=6, fontfamily="monospace")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.96)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    log.info("  Saved %s (%.2f MB)", path.name, path.stat().st_size / 1e6)


# ── Full render for the single best scene ────────────────────────────────────

def render_best_scene(result: dict, site: dict, out_dir: Path):
    """Render all 4 output images for the best-scoring scene."""
    year     = result["year"]
    date_str = result["date"]
    slug     = f"zimbral_best_{year}_{date_str.replace('-', '')}"

    _save_img(result["ratio_norm"],
              f"Zimbral Best · Depth Proxy (Stumpf) · {date_str}",
              out_dir / f"{slug}_depth_proxy.png", blue_depth_cmap)

    _save_img(_water_stretch(result["b02_enh"]),
              f"Zimbral Best · B02 Enhanced · {date_str}",
              out_dir / f"{slug}_blue.png", blue_depth_cmap)

    _save_img(_water_stretch(result["b02_enh"]),
              f"Zimbral Best · Green Reef View · {date_str}",
              out_dir / f"{slug}_green.png", green_reef_cmap)

    log.info("Best scene full render complete.")


# ── Summary outputs ───────────────────────────────────────────────────────────

def save_grid(data: dict, years: list, out_dir: Path):
    """3×3 comparison grid of depth-proxy images."""
    n_cols, n_rows = 3, math.ceil(len(years) / 3)
    fig = plt.figure(figsize=(15, n_rows * 5), facecolor="#001108")
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                             wspace=0.05, hspace=0.12,
                             left=0.02, right=0.98, top=0.93, bottom=0.02)

    for idx, yr in enumerate(years):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        if yr in data:
            d = data[yr]
            ax.imshow(d["ratio_norm"], cmap=blue_depth_cmap,
                      interpolation="bilinear", vmin=0, vmax=1)
            cc  = d["cloud_pct"]
            bvi = d["bvi_score"]
            ax.set_title(
                f"{yr} · {d['date']}\nBVI={bvi:.3f}  cloud={cc:.1f}%",
                color="white", fontsize=9, fontfamily="monospace",
            )
        else:
            ax.text(0.5, 0.5, f"{yr}\nno data",
                    color="#ff4444", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
        ax.axis("off")

    # Hide unused cells
    for idx in range(len(years), n_rows * n_cols):
        fig.add_subplot(gs[idx // n_cols, idx % n_cols]).axis("off")

    fig.suptitle(
        "Zimbral Reef · Stumpf Depth Proxy 2018-2025\n"
        "Higher BVI = better underwater reef visibility from satellite",
        color="white", fontsize=13, y=0.97,
    )
    out = out_dir / "zimbral_temporal_grid.png"
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved grid: %s (%.2f MB)", out.name, out.stat().st_size / 1e6)


def save_trend(data: dict, out_dir: Path):
    """BVI trend line 2018-2025."""
    sorted_years = sorted(data.keys())
    bvi_vals     = [data[yr]["bvi_score"] for yr in sorted_years]
    dates        = [data[yr]["date"] for yr in sorted_years]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#001108")
    ax.set_facecolor("#002211")

    ax.plot(sorted_years, bvi_vals, marker="o", color="#00ffcc",
            linewidth=2.5, markersize=9, label="BVI score")

    # Annotate each point with its date
    for yr, bvi, date in zip(sorted_years, bvi_vals, dates):
        ax.annotate(date, (yr, bvi), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#aaddcc", fontsize=7)

    # Mark the best year
    best_yr = max(data, key=lambda y: data[y]["bvi_score"])
    ax.axvline(best_yr, color="#ffdd00", linestyle="--", alpha=0.6,
               label=f"Best year: {best_yr}")

    ax.set_title("Zimbral Reef · BVI Satellite Visibility Score 2018-2025\n"
                 "(score = probability of seeing underwater reef from Sentinel-2)",
                 color="white", fontsize=12, pad=12)
    ax.set_xlabel("Year", color="white", fontsize=10)
    ax.set_ylabel("BVI Score (0-1)", color="white", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#001108", edgecolor="white", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, linestyle="--", alpha=0.25, color="white")

    out = out_dir / "zimbral_trend.png"
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved trend: %s", out.name)


def save_csv(data: dict, out_dir: Path) -> Path:
    rows = []
    for yr in sorted(data.keys()):
        d = data[yr]
        rows.append({
            "year": yr,
            "best_date": d["date"],
            "bvi_score": d["bvi_score"],
            "mode": d["mode"],
            "cloud_pct": d["cloud_pct"],
            "sza_deg": d["sza_deg"],
        })
    df = pd.DataFrame(rows).sort_values("bvi_score", ascending=False)
    out = out_dir / "zimbral_visibility_ranked.csv"
    df.to_csv(out, index=False)
    log.info("Saved ranked CSV: %s", out.name)
    return out


def update_dashboard(best_result: dict, out_dir: Path):
    """Append/update the zimbral_best_visibility entry in dashboard_layers.json."""
    if not DASHBOARD_JSON.exists():
        log.warning("dashboard_layers.json not found — skipping dashboard update.")
        return

    with open(DASHBOARD_JSON) as f:
        layers = json.load(f)

    # Remove any existing entry for this key
    layers = [l for l in layers if l.get("dir") != "zimbral_best_visibility"]

    lat, lon = ZIMBRAL["lat"], ZIMBRAL["lon"]
    buf = ZIMBRAL["buffer_m"] / 111_320  # approx degrees
    entry = {
        "dir": "zimbral_best_visibility",
        "label": f"Zimbral Best Visibility · {best_result['date']}",
        "date": best_result["date"].replace("-", ""),
        "bvi_score": best_result["bvi_score"],
        "cloud_pct": best_result["cloud_pct"],
        "bounds": [
            [lat - buf, lon - buf],
            [lat + buf, lon + buf],
        ],
        "source": "zimbral_temporal_analysis",
    }
    layers.append(entry)

    with open(DASHBOARD_JSON, "w") as f:
        json.dump(layers, f, indent=2)
    log.info("dashboard_layers.json updated with best scene: %s BVI=%.4f",
             best_result["date"], best_result["bvi_score"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  ZIMBRAL REEF — BEST SATELLITE VISIBILITY 2018-2025")
    print(f"  Lat={ZIMBRAL['lat']}  Lon={ZIMBRAL['lon']}  depth≈{ZIMBRAL['depth_m']}m")
    print("=" * 72)

    processed: dict = {}

    for year in YEARS:
        print(f"\n[{year}] Searching best clear scene...")
        result = process_year(year, ZIMBRAL, OUT_DIR, CACHE_DIR, skip_render=False)
        if result:
            processed[year] = result
            print(f"       BVI={result['bvi_score']:.4f}  mode={result['mode']}"
                  f"  clouds={result['cloud_pct']:.1f}%  date={result['date']}")

    if len(processed) < 2:
        print("\nNot enough data to produce analysis (need at least 2 years).")
        sys.exit(1)

    print("\n\nGenerating summary outputs...")
    save_grid(processed, YEARS, OUT_DIR)
    save_trend(processed, OUT_DIR)
    csv_path = save_csv(processed, OUT_DIR)

    # Identify best year
    best_year  = max(processed, key=lambda y: processed[y]["bvi_score"])
    best       = processed[best_year]

    print("\n" + "=" * 72)
    print(f"  BEST YEAR FOR ZIMBRAL VISIBILITY: {best_year}")
    print(f"  Date: {best['date']}  BVI={best['bvi_score']:.4f}  clouds={best['cloud_pct']:.1f}%")
    print("=" * 72)

    render_best_scene(best, ZIMBRAL, OUT_DIR)
    update_dashboard(best, OUT_DIR)

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print(f"Ranked results: {csv_path}")

    # Print ranking table
    print("\n  RANKING (best to worst visibility from satellite):")
    print(f"  {'Rank':<5} {'Year':<6} {'Date':<12} {'BVI':>6}  {'Clouds':>7}  Mode")
    print(f"  {'-'*5} {'-'*6} {'-'*12} {'-'*6}  {'-'*7}  ----")
    sorted_years = sorted(processed, key=lambda y: processed[y]["bvi_score"], reverse=True)
    for rank, yr in enumerate(sorted_years, 1):
        d = processed[yr]
        print(f"  {rank:<5} {yr:<6} {d['date']:<12} {d['bvi_score']:>6.4f}  "
              f"{d['cloud_pct']:>6.1f}%  {d['mode']}")


if __name__ == "__main__":
    main()
