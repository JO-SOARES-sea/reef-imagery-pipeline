#!/usr/bin/env python3
"""
generate_1200dpi_blue_images.py
===============================
Generates a single high-resolution (1200 DPI) enhanced B02 "blue image" (using reef_cmap)
for each of the 15 Algarve survey sites.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

SURVEY_SITES = [
    ("tavira_west",          37.1050, -7.6800, 12),
    ("ilha_de_tavira",       37.0950, -7.7200, 10),
    ("fuseta",               37.0600, -7.7600, 14),
    ("olhao_offshore",       37.0350, -7.8200, 12),
    ("faro_east",            37.0100, -7.8800, 15),
    ("praia_de_faro",        36.9800, -7.9400, 12),
    ("ancão_peninsula",      36.9650, -8.0000, 10),
    ("quarteira",            37.0650, -8.1000, 12),
    ("vilamoura",            37.0750, -8.1300, 14),
    ("olhos_de_agua",        37.0900, -8.1600, 12),
    ("pedra_sta_eulalia",    37.069081, -8.210242, 12),
    ("albufeira_reef",       37.0690, -8.2105, 16),
    ("galé",                 37.0560, -8.2296, 14),
    ("salgados",             37.0950, -8.3000, 12),
    ("armacao_de_pera",      37.0700, -8.3600, 10),
]

OUT_DIR = Path("../outputs/survey_1200dpi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Blue-cyan-green reef colormap
reef_cmap = LinearSegmentedColormap.from_list(
    "reef", ["#000022", "#001155", "#003388", "#0066aa", "#00aacc", "#44ddaa", "#aaffaa"]
)

def find_best_date(lat, lon):
    """Find the best cloud-free scene between 2023 and 2026."""
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime="2023-01-01/2026-05-30",
        query={"eo:cloud_cover": {"lt": 10}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        limit=5,
    )
    items = list(search.items())
    if not items:
        return None
    return items[0]

def download_and_enhance(item, lat, lon, buffer_m=500):
    """Download B02 and enhance it."""
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    try:
        with env:
            with rasterio.open(item.assets["B02"].href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                window = from_bounds(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m, src.transform)
                b02_arr = src.read(1, window=window).astype(np.float32)
    except Exception as e:
        print(f"      Error downloading B02: {e}")
        return None

    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    if b02.max() == 0 or np.all(np.isnan(b02)):
        return None

    # Enhance B02
    p95 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
    b02_corr = np.clip(b02 - 0.8 * p95 * 0.05, 0, 1.0)
    
    sigma_est = np.mean(estimate_sigma(b02_corr))
    b02_denoised = denoise_nl_means(b02_corr, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    
    b02_16 = np.clip(b02_denoised * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(4, 4))
    b02_clahe = clahe.apply(b02_16).astype(np.float32) / 65535.0
    
    b02_enhanced = b02_denoised * 0.5 + b02_clahe * 0.5
    return b02_enhanced

def main():
    print("=" * 80)
    print("  HIGH-RESOLUTION (1200 DPI) SURVEY IMAGES GENERATION")
    print(f"  Output directory: {OUT_DIR.resolve()}")
    print("=" * 80)

    for idx, (name, lat, lon, depth) in enumerate(SURVEY_SITES):
        print(f"\n[{idx+1}/{len(SURVEY_SITES)}] Processing site: {name} (lat={lat:.6f}, lon={lon:.6f})...")
        
        item = find_best_date(lat, lon)
        if not item:
            print("    SKIPPING: No cloud-free scene found.")
            continue
            
        date_str = item.datetime.strftime("%Y-%m-%d")
        print(f"    Best scene found for date: {date_str} (Cloud={item.properties.get('eo:cloud_cover'):.1f}%)")
        
        b02_enhanced = download_and_enhance(item, lat, lon)
        if b02_enhanced is None:
            print("    SKIPPING: Download or enhancement failed.")
            continue
            
        # Delineate clean stretch limits
        p2, p98 = np.percentile(b02_enhanced[b02_enhanced > 0], [2, 98]) if np.any(b02_enhanced > 0) else (0, 1)
        img_stretch = np.clip((b02_enhanced - p2) / (p98 - p2 + 1e-12), 0, 1)

        # Plot and save at 1200 DPI
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#000022")
        ax.imshow(img_stretch, cmap=reef_cmap, interpolation="bilinear")
        ax.axis("off")
        
        # Borderless save
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        out_path = OUT_DIR / f"{name}_{date_str}_blue_1200dpi.png"
        fig.savefig(out_path, dpi=1200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        
        print(f"    SUCCESS: Saved high-res image to: {out_path.name}")

    print("\n" + "=" * 80)
    print(f"  COMPLETE! All high-res blue images saved to: {OUT_DIR.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
