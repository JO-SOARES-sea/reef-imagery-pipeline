#!/usr/bin/env python3
"""
filter_olex_0_20m.py — Filter Olex reef marks and bathymetry to the 0–20 m
satellite-visible window.

Point 1 of the Olex × Sentinel-2 × Apple-GPU integration.

Reads the two vector layers already in the repo:
  - data/vector/olex_algarve_reefs.geojson      (11,345 confirmed reef marks)
  - data/vector/olex_sn1028_bathymetry.geojson  (103,689 10 m-grid soundings)

For every reef mark it assigns the depth of the nearest 10 m bathymetry cell
(UTM 29N KDTree) and keeps only marks with 0 < depth <= 20 m — the optical
limit of Sentinel-2 B02/B03 substrate discrimination.

Bathymetry points are filtered to the same 0–20 m window and are split into
a depth-class histogram used by the patch builder for depth-matched negatives.

Outputs (outputs/olex_0_20m/):
  - olex_reefs_0_20m.geojson
  - olex_bathymetry_0_20m.geojson
  - olex_filter_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "vector"
OUT_DIR = PROJECT_ROOT / "outputs" / "olex_0_20m"

DEFAULT_REEFS = DATA_DIR / "olex_algarve_reefs.geojson"
DEFAULT_BATHY = DATA_DIR / "olex_sn1028_bathymetry.geojson"

TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)

MIN_DEPTH_M = 0.0
MAX_DEPTH_M = 20.0


def _load_points(path: Path) -> list[dict]:
    with open(path) as f:
        gj = json.load(f)
    feats = gj.get("features", gj) if isinstance(gj, dict) else gj
    pts = []
    for feat in feats:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        x, y = TO_UTM.transform(lon, lat)
        pts.append({"lon": lon, "lat": lat, "x": x, "y": y, "properties": props})
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reefs", type=Path, default=DEFAULT_REEFS)
    parser.add_argument("--bathy", type=Path, default=DEFAULT_BATHY)
    parser.add_argument("--min-depth", type=float, default=MIN_DEPTH_M)
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    bathy = _load_points(args.bathy)
    bathy_xy = np.array([[p["x"], p["y"]] for p in bathy])
    bathy_depth = np.array([float(p["properties"].get("depth_m", np.nan)) for p in bathy])
    tree = cKDTree(bathy_xy)

    # ── Bathymetry: keep 0–20 m ─────────────────────────────────────────────
    bathy_kept = [p for p, d in zip(bathy, bathy_depth) if args.min_depth < d <= args.max_depth]
    log.info("Bathymetry: %d / %d points inside %.1f–%.1f m",
             len(bathy_kept), len(bathy), args.min_depth, args.max_depth)

    # ── Reefs: nearest sounding depth, keep 0–20 m ─────────────────────────
    reefs = _load_points(args.reefs)
    if not reefs:
        log.error("No reef points loaded from %s", args.reefs)
        raise SystemExit(1)
    reef_xy = np.array([[p["x"], p["y"]] for p in reefs])
    dist, idx = tree.query(reef_xy, k=1)
    kept, dropped = [], 0
    depth_classes = {}
    for j, (p, d_m) in enumerate(zip(reefs, bathy_depth[idx])):
        if not np.isfinite(d_m):
            dropped += 1
            continue
        p["properties"] = dict(p["properties"])
        p["properties"]["olex_depth_m"] = round(float(d_m), 2)
        p["properties"]["nearest_bathy_dist_m"] = round(float(dist[j]), 2)
        if args.min_depth < d_m <= args.max_depth:
            kept.append(p)
            k = int(np.floor(float(d_m) * 2)) / 2  # 0.5 m class
            depth_classes[str(k)] = depth_classes.get(str(k), 0) + 1
        else:
            dropped += 1

    log.info("Reefs: %d kept in %.1f–%.1f m, %d dropped (outside window / no depth)",
             len(kept), args.min_depth, args.max_depth, dropped)

    # ── Write outputs ───────────────────────────────────────────────────────
    def _write_geojson(points: list[dict], path: Path) -> None:
        feats = [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
             "properties": p["properties"]}
            for p in points
        ]
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
        log.info("Wrote %s (%d features)", path, len(feats))

    reefs_out = args.out / "olex_reefs_0_20m.geojson"
    bathy_out = args.out / "olex_bathymetry_0_20m.geojson"
    _write_geojson(kept, reefs_out)
    _write_geojson(bathy_kept, bathy_out)

    summary = {
        "n_reefs_in": len(reefs),
        "n_reefs_0_20m": len(kept),
        "n_reefs_dropped": dropped,
        "n_bathy_in": len(bathy),
        "n_bathy_0_20m": len(bathy_kept),
        "depth_window_m": [args.min_depth, args.max_depth],
        "reef_depth_0_5m_classes": dict(sorted(depth_classes.items(), key=lambda kv: float(kv[0]))),
    }
    with open(args.out / "olex_filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary: %s", args.out / "olex_filter_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
