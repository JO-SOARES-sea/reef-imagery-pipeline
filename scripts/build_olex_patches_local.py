#!/usr/bin/env python3
"""
build_olex_patches_local.py — Extract Sentinel-2 patches over Olex reef marks
from a LOCAL multi-band composite (offline; no STAC download).

Point 2 of the Olex × Sentinel-2 × Apple-GPU integration.

Inputs
------
  --reefs    filtered reef marks (olex_reefs_0_20m.geojson)
  --bathy    filtered bathymetry (olex_bathymetry_0_20m.geojson) — used for
             DEPTH-MATCHED sand negatives (relief_m < 0.5, |Δdepth| <= 1 m)
  --raster   multi-band composite (band order B02,B03,B04,B08), e.g.
             data/raw/pedra_do_valado_composite.tif

Outputs
-------
  {out}/train/{reef,sand}/*.npy   (4, patch_px, patch_px) float32
  {out}/val/{reef,sand}/*.npy
  {out}/patch_manifest.json

Validation note: this builder produces a first-pass dataset for the Apple-GPU
training smoke run. Production evaluation must still use the spatially grouped
/ leave-one-zone-out protocol, not random splits of this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = PROJECT_ROOT / "outputs" / "olex_0_20m" / "patches"
DEFAULT_RASTER = PROJECT_ROOT / "data" / "raw" / "pedra_do_valado_composite.tif"
DEFAULT_REEFS = PROJECT_ROOT / "outputs" / "olex_0_20m" / "olex_reefs_0_20m.geojson"
DEFAULT_BATHY = PROJECT_ROOT / "outputs" / "olex_0_20m" / "olex_bathymetry_0_20m.geojson"

PATCH_PX = 128
SAND_RELIEF_MAX_M = 0.5
DEPTH_MATCH_TOL_M = 1.0
VAL_FRACTION = 0.2


def _load_points(path: Path) -> list[dict]:
    gj = json.load(open(path))
    feats = gj.get("features", gj) if isinstance(gj, dict) else gj
    pts = []
    for feat in feats:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        pts.append({"lon": lon, "lat": lat, "properties": props})
    return pts


def _read_patch(src, to_raster, lon: float, lat: float, half: int) -> np.ndarray | None:
    """Return (4, 2*half, 2*half) float32 reflectance patch, or None if it
    falls outside the raster."""
    try:
        x, y = to_raster.transform(lon, lat)  # raster CRS (e.g. EPSG:32629)
        row, col = src.index(x, y)
    except Exception:
        return None
    # The patch centre must fall inside the raster; only the window EDGES may
    # extend beyond it (handled by zero-padding below).
    if not (0 <= col < src.width and 0 <= row < src.height):
        return None
    w = Window(col - half, row - half, 2 * half, 2 * half)
    # Read with zero-padding so edge points are kept, not dropped. 0.0 is the
    # water/nodata value in the composite, so padded pixels are inert.
    arr = src.read(window=w, boundless=True, fill_value=0.0)
    if arr.shape[0] < 4:
        return None
    arr = arr[:4].astype(np.float32)
    # Reflectance composite: keep 0–1 scale; NaN/negative -> 0 (water/nodata)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reefs", type=Path, default=DEFAULT_REEFS)
    parser.add_argument("--bathy", type=Path, default=DEFAULT_BATHY)
    parser.add_argument("--raster", type=Path, default=DEFAULT_RASTER)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--patch-px", type=int, default=PATCH_PX)
    parser.add_argument("--val-frac", type=float, default=VAL_FRACTION)
    parser.add_argument("--sand-mult", type=int, default=3, help="sand negatives per reef (depth-matched)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    half = args.patch_px // 2
    if args.patch_px % 2:
        raise SystemExit("--patch-px must be even")

    reefs = _load_points(args.reefs)
    bathy = _load_points(args.bathy)
    log.info("Loaded %d reefs, %d bathymetry points", len(reefs), len(bathy))

    # Sand pool: low relief, depth-matched per reef.
    sand_pool = [
        p for p in bathy
        if float(p["properties"].get("relief_m", 1e9) or 1e9) < SAND_RELIEF_MAX_M
    ]

    with rasterio.open(args.raster) as src:
        from pyproj import Transformer
        to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        reef_ok, reef_fail = 0, 0
        used_sand: set[tuple[float, float]] = set()
        patches: list[tuple[np.ndarray, int, str]] = []  # (arr, label, key)
        for r in reefs:
            arr = _read_patch(src, to_raster, r["lon"], r["lat"], half)
            if arr is None:
                reef_fail += 1
                continue
            depth = float(r["properties"].get("olex_depth_m", np.nan))
            patches.append((arr, 1, f"reef_{r['lon']:.5f}_{r['lat']:.5f}"))
            # Depth-matched sand: same 0.5 m depth class first (project rule),
            # then widen to ±DEPTH_MATCH_TOL_M. Up to --sand-mult NEGATIVES,
            # sampled without replacement so no sand patch leaks between
            # train and val or inflates the label count.
            sand_by_dist = sorted(
                sand_pool,
                key=lambda p: abs(float(p["properties"].get("depth_m", 1e9)) - depth),
            )
            n_sampled = 0
            for s in sand_by_dist:
                if n_sampled >= args.sand_mult:
                    break
                key = (round(s["lon"], 6), round(s["lat"], 6))
                if key in used_sand:
                    continue
                s_depth = float(s["properties"].get("depth_m", 1e9))
                same_class = int(np.floor(s_depth * 2)) == int(np.floor(depth * 2))
                within_tol = abs(s_depth - depth) <= DEPTH_MATCH_TOL_M
                if not (same_class or within_tol):
                    continue
                arr_s = _read_patch(src, to_raster, s["lon"], s["lat"], half)
                if arr_s is None:
                    continue
                patches.append((arr_s, 0, f"sand_{s['lon']:.5f}_{s['lat']:.5f}"))
                used_sand.add(key)
                n_sampled += 1
            reef_ok += 1

    log.info("Reefs with patches: %d  outside raster: %d", reef_ok, reef_fail)
    n_pos = sum(1 for _, l, _ in patches if l == 1)
    n_neg = sum(1 for _, l, _ in patches if l == 0)
    log.info("Total patches: %d (reef=%d sand=%d)", len(patches), n_pos, n_neg)

    if len(patches) == 0:
        raise SystemExit("No patches extracted — is the raster georeferenced (EPSG:32629)?")

    # Stratified train/val split (first-pass only; see production note above).
    rng.shuffle(patches)
    counts = {"train_reef": 0, "train_sand": 0, "val_reef": 0, "val_sand": 0}
    n_val_pos = max(1, int(n_pos * args.val_frac))
    n_val_neg = max(1, int(n_neg * args.val_frac))
    val_pos = val_neg = 0

    for arr, label, key in patches:
        cls = "reef" if label == 1 else "sand"
        if label == 1 and val_pos < n_val_pos:
            split = "val"; val_pos += 1
        elif label == 0 and val_neg < n_val_neg:
            split = "val"; val_neg += 1
        else:
            split = "train"
        out_dir = args.out / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"patch_{key}.npy", arr)
        counts[f"{split}_{cls}"] += 1

    with open(args.out / "patch_manifest.json", "w") as f:
        json.dump({"patch_px": args.patch_px, "counts": counts,
                   "raster": str(args.raster), "depth_match_tol_m": DEPTH_MATCH_TOL_M,
                   "sand_relief_max_m": SAND_RELIEF_MAX_M}, f, indent=2)
    log.info("Patch manifest: %s", args.out / "patch_manifest.json")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
