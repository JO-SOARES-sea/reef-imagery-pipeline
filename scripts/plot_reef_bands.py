"""
Plot B02 + B03 bands for reef analysis
2018-10-12 @ 37.0555N, 8.2296W — Secchi 23.6m
"""
import numpy as np, rasterio
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SITE_LAT, SITE_LON = 37.0555, -8.2296
BBOX_PAD = 0.08
OUT_DIR = Path("sentinel_images/20181010")

all_jp2 = list(OUT_DIR.glob("*.jp2"))
print(f"  Available files: {[f.name for f in all_jp2]}")
# T29SNB covers Algarve coast ~37°N — correct tile for this site
# T29SNC covers ~38-39°N (Lisboa/Setúbal) — wrong tile
b02_files = [f for f in all_jp2 if "B02" in f.name and "T29SNB" in f.name]
b03_files = [f for f in all_jp2 if "B03" in f.name and "T29SNB" in f.name]
tci_files = [f for f in all_jp2 if "TCI" in f.name and "T29SNB" in f.name]
# fallback priority: SNA then any
if not b02_files: b02_files = [f for f in all_jp2 if "B02" in f.name and "T29SNA" in f.name]
if not b03_files: b03_files = [f for f in all_jp2 if "B03" in f.name and "T29SNA" in f.name]
if not b02_files: b02_files = [f for f in all_jp2 if "B02" in f.name]
if not b03_files: b03_files = [f for f in all_jp2 if "B03" in f.name]
print(f"  B02: {[f.name for f in b02_files]}")
print(f"  B03: {[f.name for f in b03_files]}")
band_files = {}
if tci_files: band_files["TCI (True Colour)\nRGB visual"] = tci_files[-1]
if b02_files: band_files["B02 (Blue 490nm)\nMax. water penetration"] = b02_files[-1]
if b03_files: band_files["B03 (Green 560nm)\nMax. rock/sand contrast"] = b03_files[-1]

n = len(band_files)
fig, axes = plt.subplots(1, n, figsize=(8*n, 8))
fig.patch.set_facecolor('#0a0a1a')

for ax, (title, fpath) in zip(axes, band_files.items()):
    with rasterio.open(fpath) as src:
        print(f"  {fpath.name}: CRS={src.crs}")
        # Convert WGS84 bbox to image CRS
        wgs84 = CRS.from_epsg(4326)
        left, bottom, right, top = transform_bounds(
            wgs84, src.crs,
            SITE_LON - BBOX_PAD, SITE_LAT - BBOX_PAD,
            SITE_LON + BBOX_PAD, SITE_LAT + BBOX_PAD
        )
        print(f"  UTM bbox: {left:.0f},{bottom:.0f},{right:.0f},{top:.0f}")
        try:
            win = from_bounds(left, bottom, right, top, src.transform)
            data = src.read(1, window=win).astype(float)
            print(f"  Window shape: {data.shape}, min={data.min():.0f}, max={data.max():.0f}")
        except Exception as e:
            print(f"  Window error: {e} — reading full band")
            data = src.read(1).astype(float)

        if data.size == 0 or data.max() == 0:
            print("  Empty window — reading full image")
            data = src.read(1).astype(float)

    # Normalise
    valid = data[data > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [1, 99])
        data = np.clip((data - p2) / (p98 - p2 + 1e-9), 0, 1)
    else:
        data = np.zeros_like(data)

    if 'TCI' in title:
        with rasterio.open(fpath) as src2:
            left2, bottom2, right2, top2 = transform_bounds(CRS.from_epsg(4326), src2.crs, SITE_LON-BBOX_PAD, SITE_LAT-BBOX_PAD, SITE_LON+BBOX_PAD, SITE_LAT+BBOX_PAD)
            win2 = from_bounds(left2, bottom2, right2, top2, src2.transform)
            try:
                rgb = src2.read([1,2,3], window=win2).astype(float)
            except Exception:
                rgb = src2.read([1,2,3]).astype(float)
            rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9), 0, 1)
            ax.imshow(np.transpose(rgb, (1,2,0)), origin='upper')
    else:
        cmap = 'Blues_r' if 'B02' in title else 'Greens_r'
        ax.imshow(data, cmap=cmap, origin='upper', vmin=0, vmax=1)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    # Mark site centre approximately
    h, w = data.shape
    ax.plot(w//2, h//2, 'r+', markersize=20, markeredgewidth=2.5, label='Site')
    ax.set_xlabel("Secchi 23.6m | Kd490=0.0424 | @22m: B02 trans=39% B03 trans=37%",
                  color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='white')

plt.suptitle(
    "Sentinel-2 L2A — 2018-10-10 | Tile T29SNC\n"
    "37.0555°N, 8.2296°W @ 22m depth | Best water clarity in 10 years",
    color='white', fontsize=13, fontweight='bold'
)
plt.tight_layout()
out = OUT_DIR / "reef_band_analysis.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
print(f"\n  ✅ Saved: {out}")
plt.close()
