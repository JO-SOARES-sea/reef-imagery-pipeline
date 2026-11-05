#!/usr/bin/env python3
"""
crossref_fishing_spots_mini.py — Test: do fishing spots land on dark pixels?

Hypothesis:  if fishermens' fishing spots are a proxy for underwater reefs,
then in a clear Sentinel-2 scene the B04 reflectance at the fishing-spot GPS
should be LOWER than at random ocean points (because reefs are darker than
sand at 10 m depth in clear water).

Method
------
1. Open a user-specified Sentinel-2 L2A scene (default: 2025-09-25, which
   the user identified as the visually cleanest scene in 2025).
2. Read the B04 (red, 10 m) band over the Algarve bbox.
3. Sample the B04 reflectance at every fishing spot from
   ``outputs/reef_habitat_mini/fishing_spots/fishing_spots_filtered.csv``.
4. Sample the same number of B04 values at *random* ocean points inside the
   bbox and at the same depth window (matched control).
5. Compare the two distributions.  A significant negative shift in the
   fishing-spot distribution supports the reef-proxy hypothesis.

Inputs
------
- ``--scene-id``  STAC item id (default: 2025-09-25, the user-confirmed best scene)
- ``--spots``     fishing spots CSV from extract_fishing_spots_mini.py
- ``--depth``     EMODnet depth raster (used to mask out land for random control)

Outputs (under ``--out-dir``, default ``outputs/reef_habitat_mini/crossref_2025_09_25/``)
----------------------------------------------------------------------------------------
- ``crossref_summary.csv``      — per-spot B04 reflectance + control random samples
- ``crossref_summary.md``       — human-readable hypothesis test report
- ``crossref_overlay.png``      — B04 raster + fishing spots (red) + random control (blue)
- ``crossref_log.txt``          — run provenance

Example
-------
    python scripts/crossref_fishing_spots_mini.py \\
        --scene-id S2B_29SNB_20250925_0_L2A \\
        --spots outputs/reef_habitat_mini/fishing_spots/fishing_spots_filtered.csv \\
        --depth outputs/reef_habitat/emodnet/depth_algarve.tif

Notes
-----
* SAFE MODE: only creates new files in ``outputs/reef_habitat_mini/``.
  The baseline ``outputs/reef_habitat/`` and the user's GPX are untouched.
* The user must run ``extract_fishing_spots_mini.py`` first to produce
  ``fishing_spots_filtered.csv``.  This script does not re-derive it.
* Statistical test: Welch's t-test on B04 means (fishing vs control).
  A negative t-statistic with p < 0.05 supports the hypothesis.
* Visual sanity-check: ``crossref_overlay.png`` shows the B04 raster
  with the two point sets coloured differently.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import rowcol

try:
    from pystac_client import Client
    _HAS_PYSTAC = True
except Exception:
    _HAS_PYSTAC = False

import requests
try:
    from pyproj import Transformer as _PyprojTransformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("crossref_fishing_spots_mini")

_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"
_DN_SCALE = 10000.0


# ── STAC scene fetch (importable; small helper) ───────────────────────────────
def _get_scene_assets(scene_id: str) -> dict:
    """Fetch a Sentinel-2 L2A scene by id and return its asset URLs."""
    if _HAS_PYSTAC:
        try:
            cat = Client.open(_EARTH_SEARCH)
            item = cat.get_collection(_S2_COLLECTION).get_item(scene_id)
            if item is not None:
                return {k: v.href for k, v in item.assets.items()}
        except Exception as exc:
            log.warning("pystac-client get_item failed (%s) — falling back to raw API", exc)
    # Raw fallback
    url = f"{_EARTH_SEARCH}/collections/{_S2_COLLECTION}/items/{scene_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return {k: v["href"] for k, v in resp.json()["assets"].items()}


def _resolve_b04_href(assets: dict) -> Optional[str]:
    aliases = {"red", "b04", "b4"}
    for key, href in assets.items():
        if key.lower() in aliases:
            return href
    return None


# ── B04 sampling at a list of (lon, lat) ─────────────────────────────────────
def _sample_b04(
    b04_href: str,
    points: list[tuple[float, float]],
) -> list[Optional[float]]:
    """Return a list of B04 reflectance values (or None for out-of-raster)."""
    out: list[Optional[float]] = []
    with rasterio.open(b04_href) as src:
        epsg = src.crs.to_epsg() if src.crs is not None else None
        if epsg is not None and epsg != 4326 and _HAS_PYPROJ:
            xf = _PyprojTransformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        for lon, lat in points:
            try:
                if epsg is not None and epsg != 4326 and _HAS_PYPROJ:
                    cx, cy = xf.transform(lon, lat)
                else:
                    cx, cy = lon, lat
                r, c = rowcol(src.transform, cx, cy)
                if not (0 <= r < src.height and 0 <= c < src.width):
                    out.append(None)
                    continue
                v = float(src.read(1, window=((r, r+1), (c, c+1)))[0, 0])
                if v <= 0 or v == src.nodata:
                    out.append(None)
                else:
                    out.append(v / _DN_SCALE)
            except Exception:
                out.append(None)
    return out


# ── Random ocean control (matched depth) ──────────────────────────────────────
def _build_random_control(
    depth_path: Path,
    bbox: tuple[float, float, float, float],
    n: int,
    min_depth_m: float,
    max_depth_m: float,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Sample n random ocean points in bbox AND in depth window."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.emodnet_bathymetry import sample_depth_at_point  # type: ignore
    lon_min, lat_min, lon_max, lat_max = bbox
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        lon = float(rng.uniform(lon_min, lon_max))
        lat = float(rng.uniform(lat_min, lat_max))
        d = sample_depth_at_point(depth_path, lon, lat)
        if d is None:
            attempts += 1
            continue
        if -max_depth_m <= d <= -min_depth_m:
            out.append((lon, lat))
        attempts += 1
    log.info("Random control: %d points after %d attempts", len(out), attempts)
    return out


# ── Statistical test (no scipy dependency) ────────────────────────────────────
def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (t_statistic, two-sided p_value) using a normal approximation.

    Uses Welch's t formula and a normal-distribution p-value (valid for large
    n; with n=1617 and n≈1617, the normal approximation is excellent).
    """
    na, nb = len(a), len(b)
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    # Two-sided p-value from |t| under standard normal
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))
    return float(t), float(p)


# ── Plot (optional; matplotlib may be missing) ────────────────────────────────
def _write_overlay_png(
    b04_href: str,
    bbox: tuple[float, float, float, float],
    spots: list[tuple[float, float]],
    control: list[tuple[float, float]],
    out_path: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        log.warning("matplotlib not available — skipping overlay PNG")
        return False

    try:
        return _write_overlay_png_inner(
            b04_href, bbox, spots, control, out_path,
        )
    except Exception as exc:
        log.warning("Overlay PNG generation failed (%s) — skipping", exc)
        return False


def _write_overlay_png_inner(
    b04_href: str,
    bbox: tuple[float, float, float, float],
    spots: list[tuple[float, float]],
    control: list[tuple[float, float]],
    out_path: Path,
) -> bool:
    import matplotlib.pyplot as plt  # type: ignore
    with rasterio.open(b04_href) as src:
        # Reproject bbox EPSG:4326 → dataset CRS for the windowed read
        from rasterio.windows import from_bounds
        epsg = src.crs.to_epsg() if src.crs is not None else None
        if epsg is not None and epsg != 4326 and _HAS_PYPROJ:
            xf = _PyprojTransformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            lon_min, lat_min, lon_max, lat_max = bbox
            xs = [xf.transform(lon, lat)[0] for lon, lat in
                  [(lon_min, lat_min), (lon_min, lat_max),
                   (lon_max, lat_min), (lon_max, lat_max)]]
            ys = [xf.transform(lon, lat)[1] for lon, lat in
                  [(lon_min, lat_min), (lon_min, lat_max),
                   (lon_max, lat_min), (lon_max, lat_max)]]
            utm_bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            utm_bbox = bbox
        win = from_bounds(utm_bbox[0], utm_bbox[1], utm_bbox[2], utm_bbox[3], src.transform)
        N = 8
        r0, r1 = int(max(0, win.row_off)), int(min(src.height, win.row_off + win.height))
        c0, c1 = int(max(0, win.col_off)), int(min(src.width, win.col_off + win.width))
        if r1 <= r0 or c1 <= c0:
            log.warning("Bbox does not overlap raster — skipping overlay")
            return False
        data = src.read(
            1,
            window=((r0, r1), (c0, c1)),
            out_shape=(max(1, (r1-r0)//N), max(1, (c1-c0)//N)),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float32) / _DN_SCALE
        transform = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, c1-c0, r1-r0), src.transform,
        )

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(data, cmap="Blues_r", extent=(
        transform.c, transform.c + transform.a * (c1-c0),
        transform.f + transform.e * (r1-r0), transform.f,
    ), vmin=0, vmax=0.15)
    plt.colorbar(im, ax=ax, label="B04 BOA reflectance")
    if control:
        cs = np.array(control)
        ax.scatter(cs[:, 0], cs[:, 1], c="cyan", s=4, alpha=0.5, label=f"control n={len(control)}")
    if spots:
        ss = np.array(spots)
        ax.scatter(ss[:, 0], ss[:, 1], c="red", s=6, alpha=0.7, label=f"fishing n={len(spots)}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"B04 reflectance + fishing spots + control\nscene: {b04_href.split('/')[-3]}")
    ax.legend(loc="lower right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Wrote overlay PNG: %s", out_path)
    return True


# ── Output writers ────────────────────────────────────────────────────────────
def _write_csv(
    spots: list[dict],
    control_vals: list[float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "name", "lon", "lat", "b04_reflectance"])
        for s in spots:
            v = s.get("b04")
            w.writerow(["fishing", s.get("name", ""), s.get("lon", ""),
                        s.get("lat", ""), f"{v:.5f}" if v is not None else ""])
        for v in control_vals:
            w.writerow(["control", "", "", "", f"{v:.5f}"])
    log.info("Wrote CSV: %s", out_path)


def _write_summary_md(
    out_path: Path,
    scene_id: str,
    n_fishing: int,
    n_control: int,
    fish_arr: np.ndarray,
    ctrl_arr: np.ndarray,
    t_stat: float,
    p_val: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    delta = float(fish_arr.mean() - ctrl_arr.mean())
    pct_dark_fish  = float((fish_arr < np.percentile(np.concatenate([fish_arr, ctrl_arr]), 20)).mean() * 100)
    pct_dark_ctrl  = float((ctrl_arr < np.percentile(np.concatenate([fish_arr, ctrl_arr]), 20)).mean() * 100)
    verdict = (
        "**SUPPORTS the reef-proxy hypothesis** — fishing spots are darker than "
        "random ocean points (t < 0, p < 0.05)."
        if (t_stat < 0 and p_val < 0.05) else
        "**DOES NOT support** the reef-proxy hypothesis with this scene/sample."
    )
    with open(out_path, "w") as f:
        f.write("# Cross-reference hypothesis test\n\n")
        f.write(f"Scene: `{scene_id}`\n\n")
        f.write("| Metric | Fishing spots | Random control |\n|---|---:|---:|\n")
        f.write(f"| n (valid) | {n_fishing} | {n_control} |\n")
        f.write(f"| mean B04 | {fish_arr.mean():.5f} | {ctrl_arr.mean():.5f} |\n")
        f.write(f"| std B04  | {fish_arr.std(ddof=1):.5f} | {ctrl_arr.std(ddof=1):.5f} |\n")
        f.write(f"| median   | {float(np.median(fish_arr)):.5f} | {float(np.median(ctrl_arr)):.5f} |\n")
        f.write(f"| in darkest 20% | {pct_dark_fish:.1f}% | {pct_dark_ctrl:.1f}% |\n\n")
        f.write(f"**Mean difference (fishing − control):** `{delta:+.5f}`  (negative = fishing darker)\n\n")
        f.write(f"**Welch's t-statistic:** `{t_stat:+.3f}`\n")
        f.write(f"**two-sided p-value (normal approx):** `{p_val:.3e}`\n\n")
        f.write(f"**Verdict:** {verdict}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="crossref_fishing_spots_mini",
        description="Test whether fishing spots land on darker pixels than random ocean.",
    )
    p.add_argument("--scene-id", type=str,
                   default="S2B_29SNB_20250925_0_L2A",
                   help="STAC item id (default: 2025-09-25 — user-confirmed best scene).")
    p.add_argument("--spots", type=str,
                   default="outputs/reef_habitat_mini/fishing_spots/fishing_spots_filtered.csv",
                   help="Fishing spots CSV from extract_fishing_spots_mini.py.")
    p.add_argument("--depth", type=str,
                   default="outputs/reef_habitat/emodnet/depth_algarve.tif",
                   help="EMODnet depth raster (read-only, for control sampling).")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[-8.6, 36.85, -7.7, 37.30],
                   help="Bbox for control sampling (default: Algarve).")
    p.add_argument("--min-depth", type=float, default=3.0)
    p.add_argument("--max-depth", type=float, default=25.0)
    p.add_argument("--n-control", type=int, default=1617,
                   help="Number of random control points (default: 1617, same as fishing).")
    p.add_argument("--out-dir", type=str,
                   default="outputs/reef_habitat_mini/crossref_2025_09_25",
                   help="Output directory (isolated from outputs/reef_habitat/).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")
    sys.stdout.flush()

    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    spots_csv = Path(args.spots)
    depth_path = Path(args.depth)

    if not spots_csv.exists():
        log.error("Spots CSV not found: %s.  Run extract_fishing_spots_mini.py first.", spots_csv)
        return 1
    if not depth_path.exists():
        log.error("EMODnet depth raster not found: %s", depth_path)
        return 1

    log.info("Resolving scene: %s", args.scene_id)
    assets = _get_scene_assets(args.scene_id)
    b04_href = _resolve_b04_href(assets)
    if b04_href is None:
        log.error("Scene %s has no B04 asset", args.scene_id)
        return 1
    log.info("B04 asset: %s", b04_href)

    # Load fishing spots
    spots = []
    with open(spots_csv) as f:
        for row in csv.DictReader(f):
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (KeyError, ValueError):
                continue
            spots.append({"name": row.get("name", ""), "lon": lon, "lat": lat})
    log.info("Loaded %d fishing spots", len(spots))

    # Build random control
    control = _build_random_control(
        depth_path, tuple(args.bbox), args.n_control,
        args.min_depth, args.max_depth,
    )

    # Sample B04 at both sets
    log.info("Sampling B04 at fishing spots (n=%d)…", len(spots))
    fish_vals = _sample_b04(b04_href, [(s["lon"], s["lat"]) for s in spots])
    for s, v in zip(spots, fish_vals):
        s["b04"] = v

    log.info("Sampling B04 at control points (n=%d)…", len(control))
    ctrl_vals = _sample_b04(b04_href, control)

    # Drop None
    fish_arr = np.array([v for v in fish_vals if v is not None], dtype=np.float64)
    ctrl_arr = np.array([v for v in ctrl_vals if v is not None], dtype=np.float64)
    log.info("Valid samples: fishing=%d  control=%d", len(fish_arr), len(ctrl_arr))

    if len(fish_arr) < 10 or len(ctrl_arr) < 10:
        log.error("Too few valid samples — cannot run hypothesis test.")
        return 2

    t_stat, p_val = _welch_t(fish_arr, ctrl_arr)
    log.info("Welch t=%.3f  p=%.3e  (fish_mean=%.5f, ctrl_mean=%.5f)",
             t_stat, p_val, fish_arr.mean(), ctrl_arr.mean())

    # Outputs
    _write_csv(spots, list(ctrl_arr), out_dir / "crossref_summary.csv")
    _write_summary_md(
        out_dir / "crossref_summary.md",
        args.scene_id,
        len(fish_arr), len(ctrl_arr),
        fish_arr, ctrl_arr, t_stat, p_val,
    )
    log_path = out_dir / "crossref_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.write(f"timestamp: {ts}\n")
        f.write(f"scene_id: {args.scene_id}\n")
        f.write(f"b04_href: {b04_href}\n")
        f.write(f"spots:    {args.spots}  ({len(spots)} loaded)\n")
        f.write(f"depth:    {args.depth}\n")
        f.write(f"bbox:     {args.bbox}\n")
        f.write(f"depth_window: [{args.min_depth}, {args.max_depth}] m\n")
        f.write(f"n_control:    {args.n_control}\n")
        f.write(f"valid_samples: fishing={len(fish_arr)}  control={len(ctrl_arr)}\n")
        f.write(f"fish_mean:    {fish_arr.mean():.6f}\n")
        f.write(f"ctrl_mean:    {ctrl_arr.mean():.6f}\n")
        f.write(f"t_statistic:  {t_stat:.4f}\n")
        f.write(f"p_value:      {p_val:.4e}\n")

    # Overlay PNG (optional)
    _write_overlay_png(
        b04_href, tuple(args.bbox),
        [(s["lon"], s["lat"]) for s in spots if s.get("b04") is not None],
        control,
        out_dir / "crossref_overlay.png",
    )

    print("\n── Cross-reference hypothesis test ───────────────────────────")
    print(f"  Scene         : {args.scene_id}")
    print(f"  Fishing n     : {len(fish_arr)}")
    print(f"  Control n     : {len(ctrl_arr)}")
    print(f"  Fish  mean B04: {fish_arr.mean():.5f}")
    print(f"  Ctrl  mean B04: {ctrl_arr.mean():.5f}")
    print(f"  Δ (fish-ctrl) : {fish_arr.mean() - ctrl_arr.mean():+.5f}")
    print(f"  Welch t       : {t_stat:+.3f}")
    print(f"  p-value       : {p_val:.3e}")
    print(f"  Outputs       : {out_dir}")
    if t_stat < 0 and p_val < 0.05:
        print("  → SUPPORTS the reef-proxy hypothesis (fishing spots are darker)")
    else:
        print("  → does NOT support (no significant darkness difference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
