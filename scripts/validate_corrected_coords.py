#!/usr/bin/env python3
"""
validate_corrected_coords.py
═══════════════════════════════════════════════════════════════════════════════
Re-sample bathymetry map at CORRECTED dive site coordinates
and regenerate preview with accurate markers.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from pyproj import Transformer
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# CORRECTED COORDINATES
# ═══════════════════════════════════════════════════════════════════════════

SITES = {
    "Pedra do Alto (CORRECTED)": {"lat": 37.05815,  "lon": -8.20982,  "expected": 16},
    "Pedra do Alto (OLD)":       {"lat": 37.05895,  "lon": -8.20673,  "expected": 16},
    "Pedra Sta Eulália":         {"lat": 37.068978,  "lon": -8.210328,  "expected": 12},
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIF_PATH = _PROJECT_ROOT / "outputs" / "sprint1_bathy" / "algarve_central_bathy_10m_v1.tif"
PREVIEW_PATH = _PROJECT_ROOT / "outputs" / "sprint1_bathy" / "preview_corrected.png"

print("="*70)
print("  VALIDAÇÃO — Coordenadas corrigidas vs. antigas")
print("="*70)

with rasterio.open(TIF_PATH) as src:
    depth_band = src.read(1)
    std_band   = src.read(2)
    nobs_band  = src.read(3)
    transform  = src.transform
    crs        = src.crs
    bounds     = src.bounds

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    print(f"\n  Raster: {src.width}×{src.height} @ {src.res[0]}m")
    print(f"  CRS: {crs}")
    print(f"  Bounds: {bounds.left:.0f}, {bounds.bottom:.0f} → {bounds.right:.0f}, {bounds.top:.0f}")
    print()

    for name, cfg in SITES.items():
        x, y = to_utm.transform(cfg["lon"], cfg["lat"])
        row, col = src.index(x, y)

        # 5×5 window (50m radius)
        r0, r1 = max(0, row-2), min(src.height, row+3)
        c0, c1 = max(0, col-2), min(src.width, col+3)
        w_depth = depth_band[r0:r1, c0:c1]
        w_std   = std_band[r0:r1, c0:c1]
        w_nobs  = nobs_band[r0:r1, c0:c1]

        valid = ~np.isnan(w_depth)
        if valid.any():
            d_med = float(np.nanmedian(w_depth))
            d_min = float(np.nanmin(w_depth))
            d_max = float(np.nanmax(w_depth))
            std_med = float(np.nanmedian(w_std[valid]))
            nobs_med = int(np.nanmedian(w_nobs[valid]))
            diff = d_med - cfg["expected"]
            marker = "✓" if abs(diff) < 5 else "⚠"
            print(f"  {marker} {name:<30} "
                  f"median={d_med:5.1f}m ({d_min:.1f}–{d_max:.1f})  "
                  f"exp={cfg['expected']}m  diff={diff:+5.1f}m  "
                  f"std={std_med:.1f}  n={nobs_med}  "
                  f"pix=({row},{col})")
        else:
            print(f"  ✗ {name:<30} NaN in 5×5 window  pix=({row},{col})")

    # ═══════════════════════════════════════════════════════════════════
    # Regenerate preview with CORRECTED marker positions
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[+] Regenerating preview: {PREVIEW_PATH}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Map 1: depth
    ax = axes[0]
    cmap = plt.cm.viridis_r
    cmap.set_bad("white", alpha=0)
    im1 = ax.imshow(
        depth_band, cmap=cmap, vmin=0, vmax=25,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper"
    )
    ax.set_title(
        "Algarve Central Bathymetry — Median of 12 Sentinel-2 scenes (Stumpf SDB)\n"
        "Pedra do Alto CORRECTED: 37°03.489'N  008°12.589'W",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("UTM East (m)"); ax.set_ylabel("UTM North (m)")
    plt.colorbar(im1, ax=ax, label="Depth (m)", fraction=0.025, pad=0.02)

    # Markers — CORRECTED in red, OLD in gray
    for name, cfg in SITES.items():
        x, y = to_utm.transform(cfg["lon"], cfg["lat"])
        if "CORRECTED" in name:
            ax.plot(x, y, "rx", markersize=14, mew=2.5)
            ax.annotate("Pedra do Alto\n(CORRECTED)", (x, y),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9, color="red", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        elif "OLD" in name:
            ax.plot(x, y, "kx", markersize=10, mew=1.5, alpha=0.5)
            ax.annotate("Pedra do Alto (old coord)", (x, y),
                        xytext=(10, -15), textcoords="offset points",
                        fontsize=7, color="gray", alpha=0.6,
                        style="italic")
        else:
            ax.plot(x, y, "bx", markersize=12, mew=2)
            ax.annotate(name, (x, y), xytext=(8, 8), textcoords="offset points",
                        fontsize=9, color="blue", fontweight="bold")

    # Map 2: n_observations
    ax = axes[1]
    im2 = ax.imshow(
        np.where(nobs_band > 0, nobs_band, np.nan),
        cmap="plasma", vmin=1, vmax=int(np.nanmax(nobs_band)),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper"
    )
    ax.set_title("Coverage — Number of valid scenes per pixel", fontsize=11)
    ax.set_xlabel("UTM East (m)"); ax.set_ylabel("UTM North (m)")
    plt.colorbar(im2, ax=ax, label="N scenes", fraction=0.025, pad=0.02)

    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {PREVIEW_PATH}")

print("\n" + "="*70)
