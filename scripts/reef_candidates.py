#!/usr/bin/env python3
"""
reef_candidates.py — Extract discrete reef candidates from a likelihood map.

Reads a reef_likelihood_*.tif produced by reef_detector.py, thresholds it at a
high percentile, extracts connected dark-anomaly patches, ranks them by mean
likelihood × size, flags which patch contains a GPS-confirmed reef, and writes:
  - reef_candidates_<date>.geojson  (patch polygons + scores)
  - candidates_<date>.png            (zoomed validation figure)

These are CANDIDATES: locations sharing the confirmed reef's optical signature.
Only the GPS-anchored patch is ground-truth; the rest require dive confirmation.

Usage
-----
    python scripts/reef_candidates.py \
        --likelihood outputs/reef_detector/reef_likelihood_2025-09-25.tif \
        --percentile 85 --min-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy as rio_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REEF_SITES = [
    # ── Western (dive-confirmed) ──────────────────────────────────────────────
    {"name": "Pedra_Sta_Eulalia", "lon": -8.2102,  "lat": 37.0691 },
    {"name": "Pedra_do_Alto",     "lon": -8.20982, "lat": 37.05815},  # 37°03.489'N 008°12.589'W
    # ── Eastern (fishing GPS, CarlosBarrinha.gpx — top reef_score clusters) ──
    {"name": "e_faro_reef_11m",   "lon": -8.02498, "lat": 37.00989},  # 12 visits, TRI=0.60
    {"name": "e_faro_reef_12m_a", "lon": -8.02339, "lat": 37.00917},  # 12 visits, TRI=0.65
    {"name": "e_faro_reef_12m_b", "lon": -8.02327, "lat": 37.00857},  # 10 visits, TRI=0.57
    {"name": "e_faro_16m_highTRI","lon": -7.99863, "lat": 36.98575},  # 5 visits,  TRI=0.99
    {"name": "e_olhao_12m",       "lon": -7.94072, "lat": 36.95948},  # 11 visits, TRI=0.47
    {"name": "e_tavira_13m",      "lon": -7.94326, "lat": 36.96016},  # 7 visits,  TRI=0.82
    {"name": "e_olhao_17m_a",     "lon": -7.97075, "lat": 36.96263},  # 12 visits, TRI=0.47
    {"name": "e_olhao_17m_b",     "lon": -7.97262, "lat": 36.96305},  # 12 visits, TRI=0.45
]


def extract_candidates(likelihood_path: Path, percentile: float, min_size: int,
                       smooth_sigma: float = 1.0) -> dict:
    from scipy.ndimage import label, find_objects, gaussian_filter

    with rasterio.open(likelihood_path) as src:
        like = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        transform = src.transform
        crs = src.crs
    like[like == nodata] = np.nan

    # Smooth to suppress single-pixel speckle while preserving compact reef patches.
    # NaN-aware Gaussian: normalise by the smoothed validity mask.
    if smooth_sigma > 0:
        valid = np.isfinite(like).astype(np.float32)
        filled = np.where(np.isfinite(like), like, 0.0)
        num = gaussian_filter(filled, sigma=smooth_sigma, mode="constant")
        den = gaussian_filter(valid,  sigma=smooth_sigma, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            like = np.where(den > 0.2, num / den, np.nan).astype(np.float32)

    finite = like[np.isfinite(like)]
    thresh = float(np.percentile(finite, percentile))
    log.info("Threshold @ %.0fth percentile = %.4f (over %d valid px)", percentile, thresh, finite.size)

    binary = np.isfinite(like) & (like >= thresh)
    labels, n = label(binary)
    log.info("Connected components above threshold: %d", n)

    # Which component holds each GPS reef?
    # Search within a 150 m radius (≈15 px at 10 m/px) to tolerate GPS uncertainty
    # and minor smoothing artefacts that can push the exact pixel just below threshold.
    # Sites outside the raster extent are skipped — they cannot be matched without
    # risk of grabbing a component at the tile edge (false positive).
    gps_labels = {}
    h, w = like.shape
    res_m = abs(transform.a) * 111_320  # degrees/px × metres/degree (approx)
    radius_px = max(1, int(round(150 / res_m)))
    EDGE_MARGIN = radius_px  # skip if GPS is within radius_px of the tile boundary
    for site in REEF_SITES:
        r0, c0 = rowcol(transform, site["lon"], site["lat"])
        # Skip sites that fall outside (or too close to the edge of) this raster
        if not (EDGE_MARGIN <= r0 < h - EDGE_MARGIN and EDGE_MARGIN <= c0 < w - EDGE_MARGIN):
            log.debug("Skipping %s — outside raster extent", site["name"])
            continue
        best_lab = int(labels[r0, c0])
        if best_lab == 0:  # GPS pixel is background — search neighbourhood
            r_lo, r_hi = max(0, r0 - radius_px), min(h, r0 + radius_px + 1)
            c_lo, c_hi = max(0, c0 - radius_px), min(w, c0 + radius_px + 1)
            window = labels[r_lo:r_hi, c_lo:c_hi]
            win_like = like[r_lo:r_hi, c_lo:c_hi]
            candidates_in_win = np.unique(window[window > 0])
            if len(candidates_in_win) > 0:
                # Pick the component with the highest mean likelihood in the window
                best_lab = int(max(candidates_in_win,
                                   key=lambda lab: float(np.nanmean(win_like[window == lab]))))
        if best_lab > 0:
            gps_labels[best_lab] = site["name"]

    slices = find_objects(labels)
    candidates = []
    for lab in range(1, n + 1):
        comp = labels == lab
        size = int(comp.sum())
        if size < min_size:
            continue
        mean_like = float(np.nanmean(like[comp]))
        max_like  = float(np.nanmax(like[comp]))
        rows, cols = np.where(comp)
        cr, cc = rows.mean(), cols.mean()
        lon, lat = rio_xy(transform, cr, cc)
        # Bounding box in geo coords
        sl = slices[lab - 1]
        ny0, nx0 = sl[0].start, sl[1].start
        ny1, nx1 = sl[0].stop,  sl[1].stop
        lon0, lat0 = rio_xy(transform, ny1, nx0)
        lon1, lat1 = rio_xy(transform, ny0, nx1)
        candidates.append({
            "id":          lab,
            "centroid":    [round(float(lon), 5), round(float(lat), 5)],
            "size_px":     size,
            "area_ha":     round(size * 100 / 1e4, 2),   # 10 m px → 100 m²
            "mean_like":   round(mean_like, 4),
            "max_like":    round(max_like, 4),
            "score":       round(mean_like * np.log1p(size), 4),
            "gps_confirmed_reef": gps_labels.get(lab),
            "bbox":        [round(lon0,5), round(lat0,5), round(lon1,5), round(lat1,5)],
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank

    return {
        "candidates": candidates,
        "threshold":  thresh,
        "percentile": percentile,
        "n_total":    n,
        "n_kept":     len(candidates),
        "like":       like,
        "transform":  transform,
        "crs":        crs,
        "gps_labels": gps_labels,
    }


def write_geojson(result: dict, out_path: Path) -> None:
    features = []
    for c in result["candidates"]:
        lon0, lat0, lon1, lat1 = c["bbox"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0],
                ]],
            },
            "properties": {k: v for k, v in c.items() if k != "bbox"},
        })
    fc = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    log.info("Wrote %d candidate polygons → %s", len(features), out_path)


def write_figure(result: dict, date: str, out_path: Path) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        log.warning("matplotlib unavailable — skipping figure")
        return

    like = result["like"]
    transform = result["transform"]
    h, w = like.shape

    # Find the GPS reef pixel to zoom around
    site = REEF_SITES[0]
    r, c = rowcol(transform, site["lon"], site["lat"])
    zoom = 70
    r0, r1 = max(0, r - zoom), min(h, r + zoom)
    c0, c1 = max(0, c - zoom), min(w, c + zoom)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Full scene likelihood with top candidates
    im0 = ax[0].imshow(np.ma.masked_invalid(like), cmap="inferno", vmin=0, vmax=1)
    ax[0].set_title(f"{date} — reef likelihood (full shelf)")
    for cand in result["candidates"][:15]:
        lon, lat = cand["centroid"]
        rr, ccol = rowcol(transform, lon, lat)
        color = "cyan" if cand["gps_confirmed_reef"] else "lime"
        ax[0].plot(ccol, rr, "o", mfc="none", mec=color, ms=8, mew=1.5)
    ax[0].axis("off")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # Zoomed on GPS reef
    im1 = ax[1].imshow(np.ma.masked_invalid(like[r0:r1, c0:c1]), cmap="inferno", vmin=0, vmax=1)
    ax[1].set_title("Zoom: Pedra Sta Eulália (GPS ◇)")
    ax[1].plot(c - c0, r - r0, "co", mfc="none", ms=18, mew=2.5)
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract reef candidates from a likelihood map")
    ap.add_argument("--likelihood", required=True, type=Path)
    ap.add_argument("--percentile", type=float, default=85.0)
    ap.add_argument("--min-size",   type=int,   default=4, help="Min patch size in pixels")
    ap.add_argument("--smooth",     type=float, default=1.0, help="Gaussian smoothing sigma (px)")
    args = ap.parse_args()

    date = args.likelihood.stem.replace("reef_likelihood_", "")
    result = extract_candidates(args.likelihood, args.percentile, args.min_size, args.smooth)

    out_dir = args.likelihood.parent
    write_geojson(result, out_dir / f"reef_candidates_{date}.geojson")
    write_figure(result, date, out_dir / f"candidates_{date}.png")

    # Report
    print(f"\n{'='*70}\nREEF CANDIDATES — {date}\n{'='*70}")
    print(f"  Threshold: {args.percentile:.0f}th percentile = {result['threshold']:.4f}")
    print(f"  Patches:   {result['n_kept']} kept (≥{args.min_size}px) of {result['n_total']} total\n")
    print(f"  {'Rank':<5}{'Centroid':<26}{'Area':<10}{'MeanLike':<10}{'GPS?'}")
    for c in result["candidates"][:15]:
        gps = c["gps_confirmed_reef"] or ""
        cen = f"{c['centroid'][1]:.4f}N {abs(c['centroid'][0]):.4f}W"
        flag = "★ CONFIRMED" if gps else ""
        print(f"  #{c['rank']:<4}{cen:<26}{c['area_ha']:>5} ha  {c['mean_like']:<10}{flag}")

    # Where does each confirmed GPS reef rank?
    confirmed = [c for c in result["candidates"] if c["gps_confirmed_reef"]]
    if confirmed:
        for cc in confirmed:
            print(f"\n  ✓ GPS reef '{cc['gps_confirmed_reef']}' detected at rank "
                  f"#{cc['rank']} of {result['n_kept']} candidates "
                  f"({cc['area_ha']} ha, mean likelihood {cc['mean_like']})")
    else:
        print("\n  ✗ No GPS reef in any extracted patch — try lowering --percentile")


if __name__ == "__main__":
    main()
