#!/usr/bin/env python3
"""
union_reef_candidates_mini.py — Combine model + B04 + EMODnet TRI evidence.

Inputs three independent signals of reef likelihood and outputs a single ranked
list of the most credible reef candidates:

  1. **Model probability**  (from ``score_fishing_spots_mini.py``)
  2. **B04 darkness**       (from ``crossref_fishing_spots_mini.py``)
  3. **EMODnet TRI**        (Terrain Ruggedness Index — bathymetric complexity)

A spot that scores well on multiple signals is much more credible than one that
scores on only one.  This script computes a composite rank, then emits the
top-K most credible candidates ready for human curation.

Inputs
------
- ``--model-scores``   CSV from score_fishing_spots_mini.py
- ``--b04-scores``     CSV from crossref_fishing_spots_mini.py (B04 ranked)
- ``--tri-raster``     EMODnet TRI GeoTIFF (read-only)
- ``--tri-min``        Minimum TRI (m) for a "high-complexity" cell (default 0.5)

Outputs (under ``--out-dir``, default ``outputs/reef_habitat_mini/union_reef_candidates/``)
--------------------------------------------------------------------------------------------
- ``union_reef_candidates.csv``  — all union candidates with rank columns
- ``union_reef_candidates.md``   — top 50 by composite rank
- ``union_overlay.png``          — TRI raster with the top-30 candidates overlaid

Example
-------
    python scripts/union_reef_candidates_mini.py \\
        --model-scores outputs/reef_habitat_mini/model_scored_fishing/model_scores.csv \\
        --b04-scores outputs/reef_habitat_mini/crossref_2025_09_25/named_reef_candidates.csv \\
        --tri-raster outputs/reef_habitat/emodnet/tri_algarve.tif

Notes
-----
* SAFE MODE: only creates new files in ``outputs/reef_habitat_mini/``.
  Baseline ``outputs/reef_habitat/`` and EMODnet rasters are untouched.
* The composite rank is ``model_rank + b04_rank`` (lower = better).  TRI
  is added as a *multiplier*: spots with TRI < --tri-min are penalised.
* "Top-50 model + top-50 B04" is configurable via ``--top-k``.
* The output is a *recommendation* for human curation.  No candidate is
  injected into the baseline ``REEF_SITES`` automatically.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

try:
    from pyproj import Transformer as _PyprojTransformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("union_reef_candidates_mini")


# ── Data record ───────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    name: str
    lon: float
    lat: float
    emodnet_depth_m: Optional[float] = None
    b04: Optional[float] = None
    model_prob: Optional[float] = None
    tri_value: Optional[float] = None
    model_rank: Optional[int] = None
    b04_rank: Optional[int] = None
    source: list = field(default_factory=list)   # ["model", "b04", "tri"]
    composite_rank: Optional[int] = None


# ── Loaders ───────────────────────────────────────────────────────────────────
def _load_model_scores(path: Path) -> dict[tuple[str, str], Candidate]:
    out: dict[tuple[str, str], Candidate] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                lon = float(r["lon"]); lat = float(r["lat"])
            except (KeyError, ValueError):
                continue
            key = (f"{lon:.6f}", f"{lat:.6f}")
            d = r.get("emodnet_depth_m") or ""
            try: d = float(d)
            except Exception: d = None
            mp = r.get("model_prob") or ""
            try: mp = float(mp)
            except Exception: mp = None
            b04 = r.get("b04_reflectance") or ""
            try: b04 = float(b04)
            except Exception: b04 = None
            c = Candidate(
                name=r.get("name", ""), lon=lon, lat=lat,
                emodnet_depth_m=d, b04=b04, model_prob=mp,
                source=["model"] if mp is not None else [],
            )
            out[key] = c
    log.info("Loaded %d model scores from %s", len(out), path)
    return out


def _load_b04_scores(
    path: Path,
    into: dict[tuple[str, str], Candidate],
) -> None:
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                lon = float(r["lon"]); lat = float(r["lat"])
            except (KeyError, ValueError):
                continue
            key = (f"{lon:.6f}", f"{lat:.6f}")
            d = r.get("emodnet_depth_m") or ""
            try: d = float(d)
            except Exception: d = None
            b04 = r.get("b04_reflectance") or ""
            try: b04 = float(b04)
            except Exception: b04 = None
            if key in into:
                if into[key].b04 is None and b04 is not None:
                    into[key].b04 = b04
                if into[key].emodnet_depth_m is None and d is not None:
                    into[key].emodnet_depth_m = d
                if "b04" not in into[key].source:
                    into[key].source.append("b04")
            else:
                into[key] = Candidate(
                    name=r.get("name", ""), lon=lon, lat=lat,
                    emodnet_depth_m=d, b04=b04, model_prob=None,
                    source=["b04"] if b04 is not None else [],
                )
    log.info("Merged B04 scores (total candidates now: %d)", len(into))


# ── TRI sampling ─────────────────────────────────────────────────────────────
def _sample_tri(
    tri_path: Path,
    candidates: list[Candidate],
) -> None:
    """Sample the EMODnet TRI raster at each candidate's lon/lat."""
    log.info("Sampling TRI at %d candidates", len(candidates))
    with rasterio.open(tri_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        if src.crs is not None and src.crs.to_epsg() != 4326 and _HAS_PYPROJ:
            xf = _PyprojTransformer.from_crs("EPSG:4326",
                                              f"EPSG:{src.crs.to_epsg()}",
                                              always_xy=True)
        else:
            xf = None
        n_ok = 0
        for c in candidates:
            try:
                if xf is not None:
                    cx, cy = xf.transform(c.lon, c.lat)
                else:
                    cx, cy = c.lon, c.lat
                r, col = src.index(cx, cy)
                if not (0 <= r < src.height and 0 <= col < src.width):
                    continue
                v = float(src.read(1, window=((r, r+1), (col, col+1)))[0, 0])
                if v == nodata:
                    continue
                c.tri_value = v
                n_ok += 1
            except Exception:
                continue
    log.info("TRI sampled: %d / %d valid", n_ok, len(candidates))


# ── Rank + composite ─────────────────────────────────────────────────────────
def _rank_and_combine(
    candidates: list[Candidate],
    top_k_model: int,
    top_k_b04: int,
    tri_min: float,
) -> list[Candidate]:
    """Assign ranks, take union of top-K, compute composite score."""
    # Sort for ranking
    by_model = sorted([c for c in candidates if c.model_prob is not None],
                      key=lambda c: -c.model_prob)
    by_b04   = sorted([c for c in candidates if c.b04 is not None],
                      key=lambda c: c.b04)  # lower = darker = more reef-like

    for i, c in enumerate(by_model, start=1):
        c.model_rank = i
    for i, c in enumerate(by_b04, start=1):
        c.b04_rank = i

    # Union: top-K from each
    top_m = by_model[:top_k_model]
    top_b = by_b04[:top_k_b04]
    keys_seen: set = set()
    union: list[Candidate] = []
    for c in top_m + top_b:
        k = (c.name, f"{c.lon:.6f}", f"{c.lat:.6f}")
        if k in keys_seen:
            continue
        keys_seen.add(k)
        union.append(c)

    # Composite: lower is better
    for c in union:
        c.composite_rank = (c.model_rank or top_k_model + 1) + (c.b04_rank or top_k_b04 + 1)
        # Penalise if TRI is below threshold
        if c.tri_value is None or c.tri_value < tri_min:
            c.composite_rank += 1000
        else:
            c.source.append("tri")

    union.sort(key=lambda c: c.composite_rank)
    for i, c in enumerate(union, start=1):
        c.composite_rank = i
    return union


# ── Output writers ────────────────────────────────────────────────────────────
def _write_csv(cands: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["composite_rank", "name", "lon", "lat",
                    "emodnet_depth_m", "b04", "model_prob", "tri",
                    "model_rank", "b04_rank", "sources"])
        for c in cands:
            w.writerow([
                c.composite_rank, c.name, f"{c.lon:.6f}", f"{c.lat:.6f}",
                "" if c.emodnet_depth_m is None else f"{c.emodnet_depth_m:.3f}",
                "" if c.b04 is None else f"{c.b04:.5f}",
                "" if c.model_prob is None else f"{c.model_prob:.4f}",
                "" if c.tri_value is None else f"{c.tri_value:.3f}",
                c.model_rank or "", c.b04_rank or "",
                "+".join(c.source),
            ])
    log.info("Wrote CSV: %s  (%d rows)", path, len(cands))


def _write_md(cands: list[Candidate], path: Path, n_top: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n_top, len(cands))
    with open(path, "w") as f:
        f.write("# Union reef candidates — top by composite rank\n\n")
        f.write(f"Total candidates in union: {len(cands)}  "
                f"(top-{min(50, len(cands))} shown below)\n\n")
        f.write("Composite rank = model_rank + b04_rank + 1000 (penalty) if TRI is "
                "below threshold.\n\n")
        f.write("| rank | name | lon | lat | depth_m | b04 | model_prob | tri_m | sources |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        for c in cands[:n]:
            f.write(
                f"| {c.composite_rank} | {c.name} | {c.lon:.4f} | {c.lat:.4f} | "
                f"{'' if c.emodnet_depth_m is None else f'{c.emodnet_depth_m:.1f}'} | "
                f"{'' if c.b04 is None else f'{c.b04:.5f}'} | "
                f"{'' if c.model_prob is None else f'{c.model_prob:.4f}'} | "
                f"{'' if c.tri_value is None else f'{c.tri_value:.2f}'} | "
                f"{'+'.join(c.source)} |\n"
            )
    log.info("Wrote Markdown: %s  (top %d)", path, n)


def _write_overlay_png(
    tri_path: Path,
    cands: list[Candidate],
    out_path: Path,
    bbox: tuple[float, float, float, float],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        log.warning("matplotlib not available — skipping overlay PNG")
        return

    try:
        with rasterio.open(tri_path) as src:
            from rasterio.windows import from_bounds
            epsg = src.crs.to_epsg() if src.crs is not None else None
            if epsg is not None and epsg != 4326 and _HAS_PYPROJ:
                xf = _PyprojTransformer.from_crs(
                    "EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                xs, ys = [], []
                for lon, lat in [
                    (bbox[0], bbox[1]), (bbox[0], bbox[3]),
                    (bbox[2], bbox[1]), (bbox[2], bbox[3]),
                ]:
                    x, y = xf.transform(lon, lat)
                    xs.append(x); ys.append(y)
                utm_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                utm_bbox = bbox
            win = from_bounds(*utm_bbox, src.transform)
            N = 8
            r0 = int(max(0, win.row_off)); r1 = int(min(src.height, win.row_off + win.height))
            c0 = int(max(0, win.col_off)); c1 = int(min(src.width, win.col_off + win.width))
            if r1 <= r0 or c1 <= c0:
                log.warning("Bbox does not overlap raster — skipping overlay")
                return
            data = src.read(
                1,
                window=((r0, r1), (c0, c1)),
                out_shape=(max(1, (r1-r0)//N), max(1, (c1-c0)//N)),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)
            transform = rasterio.windows.transform(
                rasterio.windows.Window(c0, r0, c1-c0, r1-r0), src.transform,
            )
        fig, ax = plt.subplots(figsize=(10, 12))
        vmax = float(np.nanpercentile(data, 95)) if data.size else 1.0
        im = ax.imshow(data, cmap="viridis", extent=(
            transform.c, transform.c + transform.a * (c1-c0),
            transform.f + transform.e * (r1-r0), transform.f,
        ), vmin=0, vmax=max(vmax, 0.1))
        plt.colorbar(im, ax=ax, label="TRI (m)")
        for i, c in enumerate(cands[:30]):
            ax.scatter(c.lon, c.lat, c="red", s=20, edgecolors="white", linewidths=0.5)
            ax.annotate(f"{i+1}", (c.lon, c.lat), color="white", fontsize=6,
                        ha="center", va="center")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title("EMODnet TRI + top-30 union candidates")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        log.info("Wrote overlay PNG: %s", out_path)
    except Exception as exc:
        log.warning("Overlay PNG failed (%s) — skipping", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="union_reef_candidates_mini",
        description="Union of top-K model + top-K B04, intersected with TRI.",
    )
    p.add_argument("--model-scores", type=str,
                   default="outputs/reef_habitat_mini/model_scored_fishing/model_scores.csv")
    p.add_argument("--b04-scores", type=str,
                   default="outputs/reef_habitat_mini/crossref_2025_09_25/named_reef_candidates.csv")
    p.add_argument("--tri-raster", type=str,
                   default="outputs/reef_habitat/emodnet/tri_algarve.tif")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-K from each method (default: 50).")
    p.add_argument("--tri-min", type=float, default=0.5,
                   help="Minimum TRI (m) for a 'high-complexity' cell (default: 0.5).")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[-8.6, 36.85, -7.7, 37.30])
    p.add_argument("--out-dir", type=str,
                   default="outputs/reef_habitat_mini/union_reef_candidates")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")
    sys.stdout.flush()

    args = _parse_args(argv)
    tri_path = Path(args.tri_raster)
    if not tri_path.exists():
        log.error("TRI raster not found: %s", tri_path)
        return 1

    cand = _load_model_scores(Path(args.model_scores))
    _load_b04_scores(Path(args.b04_scores), cand)
    _sample_tri(tri_path, list(cand.values()))

    union = _rank_and_combine(list(cand.values()), args.top_k, args.top_k, args.tri_min)
    out_dir = Path(args.out_dir)
    _write_csv(union, out_dir / "union_reef_candidates.csv")
    _write_md(union,  out_dir / "union_reef_candidates.md")
    _write_overlay_png(tri_path, union, out_dir / "union_overlay.png", tuple(args.bbox))

    log_path = out_dir / "union_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.write(f"timestamp:        {ts}\n")
        f.write(f"model_scores:     {args.model_scores}\n")
        f.write(f"b04_scores:       {args.b04_scores}\n")
        f.write(f"tri_raster:       {args.tri_raster}\n")
        f.write(f"top_k (each):     {args.top_k}\n")
        f.write(f"tri_min (m):      {args.tri_min}\n")
        f.write(f"union size:       {len(union)}\n")
        tri_ok = sum(1 for c in union if c.tri_value is not None)
        f.write(f"union with TRI:   {tri_ok}\n")
        f.write(f"union with TRI >= tri_min: "
                f"{sum(1 for c in union if c.tri_value is not None and c.tri_value >= args.tri_min)}\n")

    # Print summary
    n_tri_ok = sum(1 for c in union if c.tri_value is not None
                                    and c.tri_value >= args.tri_min)
    print("\n── Union reef candidates ─────────────────────────────────────")
    print(f"  Top-K (each)        : {args.top_k}")
    print(f"  Union size          : {len(union)}")
    print(f"  With TRI >= {args.tri_min} m    : {n_tri_ok}")
    print(f"  Outputs             : {out_dir}")
    print("\n  Top 10 by composite rank:")
    print(f"  {'rank':<5}{'name':<18}{'lon':<10}{'lat':<10}{'b04':<8}{'model':<8}{'tri':<6}  sources")
    for c in union[:10]:
        b04_s = f"{c.b04:.4f}" if c.b04 is not None else "    -"
        mp_s  = f"{c.model_prob:.3f}" if c.model_prob is not None else "    -"
        tri_s = f"{c.tri_value:.2f}" if c.tri_value is not None else "    -"
        print(f"  {c.composite_rank:<5}{c.name[:16]:<18}{c.lon:<10.4f}{c.lat:<10.4f}"
              f"{b04_s:<8}{mp_s:<8}{tri_s:<6}  {'+'.join(c.source)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
