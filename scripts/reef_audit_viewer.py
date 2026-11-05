#!/usr/bin/env python3
"""
reef_audit_viewer.py — Human-in-the-loop audit page for reef GPS + view verification.

Generates a self-contained HTML (embedded PNGs) that opens in Safari and shows,
for each candidate GPS coordinate and each Sentinel-2 scene:
  - true-colour crop with a crosshair at the GPS point
  - green-reflectance crop (the reef appears as a DARK patch)
The human auditor confirms: (1) does the crosshair land on the reef, and
(2) is the dark feature actually the reef.

Every pixel is real Sentinel-2 L2A BOA reflectance — nothing synthetic.

Usage
-----
    python scripts/reef_audit_viewer.py
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pystac_client import Client

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"

# Candidate GPS coordinates to audit (the human confirms which is correct)
GPS_CANDIDATES = [
    {"label": "A — detector coord (matches HTML filename)", "lon": -8.2102, "lat": 37.0691,
     "source": "reef_detector.py / reef_viewer HTML"},
    {"label": "B — dataset coord", "lon": -8.1695, "lat": 37.0643,
     "source": "build_reef_dataset.py"},
    {"label": "C — Pedra do Alto", "lon": -8.20982, "lat": 37.05815,
     "source": "GPS 37°03.489'N 008°12.589'W — confirmed 2026-06-22"},
]

SCENES = [
    {"id": "S2B_29SNB_20250925_0_L2A", "date": "2025-09-25",
     "note": "higher bottom contrast — best reef visibility"},
    {"id": "S2A_29SNB_20230901_0_L2A", "date": "2023-09-01",
     "note": "lowest glint / 'best BVI' label"},
]

OUT_HTML = Path("outputs/reef_detector/reef_audit.html")


def _crop(href: str, lon: float, lat: float, half_px: int = 75) -> tuple[np.ndarray, int, int]:
    """Read a (2*half_px) window centred on (lon,lat). Returns (array, cy, cx)."""
    with rasterio.open(href) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        r, c = src.index(x, y)
        r0, r1 = r - half_px, r + half_px
        c0, c1 = c - half_px, c + half_px
        win = Window.from_slices((r0, r1), (c0, c1))
        arr = src.read(1, window=win).astype(np.float32) / 10000.0
        return arr, half_px, half_px


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _resolve(scene_id: str) -> dict:
    cat = Client.open(_EARTH_SEARCH)
    it = list(cat.search(collections=["sentinel-2-l2a"], ids=[scene_id]).items())[0]
    return {k: v.href for k, v in it.assets.items()}


def build_panel(assets: dict, gps: dict, half_px: int, scale_m: int) -> str:
    """Two-up panel: true colour + green reflectance, crosshair at GPS."""
    blue, cy, cx = _crop(assets["blue"], gps["lon"], gps["lat"], half_px)
    green, _, _  = _crop(assets["green"], gps["lon"], gps["lat"], half_px)
    red, _, _    = _crop(assets["red"], gps["lon"], gps["lat"], half_px)

    rgb = np.dstack([red, green, blue])
    p99 = np.percentile(rgb, 99)
    rgb = np.clip(rgb / (p99 if p99 > 0 else 1), 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    extent_m = half_px * 10  # metres from centre

    ax[0].imshow(rgb)
    ax[0].set_title(f"True colour ({2*extent_m} m wide)", fontsize=9)
    ax[1].imshow(green, cmap="viridis")
    ax[1].set_title("Green reflectance — reef = DARK", fontsize=9)
    for a in ax:
        a.axhline(cy, color="cyan", lw=0.6, alpha=0.7)
        a.axvline(cx, color="cyan", lw=0.6, alpha=0.7)
        a.plot(cx, cy, "o", mfc="none", mec="cyan", ms=18, mew=2)
        a.axis("off")
    fig.suptitle(f"GPS {gps['lat']:.4f}°N  {abs(gps['lon']):.4f}°W   |   {gps['label']}",
                 fontsize=10, color="#1a3a5c")
    fig.tight_layout()
    return _fig_to_b64(fig)


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    cards = []

    for scene in SCENES:
        log.info("Resolving %s …", scene["id"])
        assets = _resolve(scene["id"])
        for gps in GPS_CANDIDATES:
            log.info("  Rendering %s @ %s", scene["date"], gps["label"][:20])
            try:
                b64 = build_panel(assets, gps, half_px=75, scale_m=1500)
                cards.append((scene, gps, b64))
            except Exception as exc:
                log.warning("  Failed: %s", exc)

    # Build HTML
    html = ["""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reef Audit — Sentinel-2 GPS &amp; View Verification</title>
<style>
  body { font-family:-apple-system,sans-serif; background:#0a1628; color:#e8edf5; padding:20px; max-width:1100px; margin:0 auto; }
  h1 { color:#7ec8e3; font-size:1.3rem; } h2 { color:#a8c5da; font-size:1rem; margin-top:24px; border-bottom:1px solid #1e3a5c; padding-bottom:6px;}
  .card { background:#12213a; border-radius:10px; padding:14px; margin:14px 0; }
  .card img { width:100%; border-radius:6px; }
  .meta { font-size:0.8rem; color:#6b7f94; margin:6px 0; }
  .q { background:#1a2744; border:1px solid #1e3a5c; border-radius:8px; padding:14px; margin:20px 0; font-size:0.9rem; line-height:1.6;}
  .q strong { color:#7ec8e3; }
  code { background:#0a1628; padding:1px 5px; border-radius:3px; color:#e3c87e; }
</style></head><body>
<h1>🔍 Reef Audit — verify GPS &amp; view against real Sentinel-2</h1>
<p class="meta">Every image is real Sentinel-2 L2A bottom-of-atmosphere reflectance. Cyan crosshair = the GPS coordinate. In the green-reflectance panel, reef/rock appears DARK against lighter sand.</p>
<div class="q">
<strong>As auditor, please confirm for each location:</strong><br>
1. Does the cyan crosshair land ON the dark reef structure?<br>
2. Is the dark feature actually the reef you dived (or sand/seagrass/artifact)?<br>
3. Which coordinate (A, B, or C) is the TRUE Pedra de Santa Eulália dive site?<br>
<em>I found a discrepancy: <code>reef_detector.py</code> uses A (37.0691N, 8.2102W) but <code>build_reef_dataset.py</code> uses B (37.0643N, 8.1695W) — ~5 km apart, same name.</em>
</div>
"""]

    last_date = None
    for scene, gps, b64 in cards:
        if scene["date"] != last_date:
            html.append(f'<h2>📅 {scene["date"]} — {scene["note"]}<br><span class="meta">{scene["id"]}</span></h2>')
            last_date = scene["date"]
        html.append(f'''<div class="card">
  <div class="meta"><strong>{gps["label"]}</strong> — source: {gps["source"]}</div>
  <img src="data:image/png;base64,{b64}">
</div>''')

    html.append("</body></html>")
    OUT_HTML.write_text("\n".join(html))
    log.info("Wrote audit page: %s", OUT_HTML)
    print(f"\n✓ Open in Safari:\n  file://{OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
