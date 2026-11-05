#!/usr/bin/env python3
"""
eval_scene_quality_mini.py — Scene-quality ranking for Sentinel-2 L2A scenes
covering the Algarve bbox.

This script makes the choice of a "best" Sentinel-2 scene for reef-bottom
classification *auditable*.  The baseline ``scripts/build_reef_dataset.py``
hard-codes a single scene id (``S2B_29SNB_20240930_0_L2A``) with no empirical
justification.  Here we rank all clear (cloud < ``--max-cloud``) Sentinel-2
L2A scenes that intersect the Algarve bbox in a date window by a composite
optical quality score and emit a CSV + Markdown table of the top-N scenes.

Methodology
-----------
For each Sentinel-2 L2A scene returned by Earth Search STAC intersecting the
bbox in the date range with cloud cover < ``--max-cloud``:

1. Use ``pystac-client`` against
   ``https://earth-search.aws.element84.com/v1`` (same endpoint as
   ``src/s2_patch_extractor.py``).
2. Open only the B04 (red, 10 m) COG asset via ``rasterio`` and read a small
   windowed subset centred on the bbox centroid (windowed reads — no full
   file download).
3. Compute per-scene metrics:
   - ``cloud_cover_pct``     : from ``item.properties["eo:cloud_cover"]``
   - ``mean_reflectance_B04``: mean of valid pixels divided by 10000
                               (S2 L2A DN→reflectance scaling)
   - ``std_reflectance_B04`` : std of valid pixels (reflectance units)
   - ``snr_db``              : 20 * log10(mean / (std + 1e-6))  — rough SNR proxy
   - ``bvi_score``           : heuristic "bottom visibility index" proxy in [0,1]
                               defined as::

                                   bvi = clip(snr_db / 50.0, 0, 1)
                                       * clip(1 - cloud_cover_pct/100, 0, 1)

                               This is a *heuristic* — it is NOT the production
                               BVI from ``src/reef_ml_predictor_acolite.py``.
                               It is intended only for auditable ranking.
   - ``composite_rank``      : rank by ``bvi_score`` (desc),
                               tie-break by ``cloud_cover_pct`` (asc).

Outputs (under ``--out-dir``, default ``outputs/reef_habitat_mini/scene_ranking``):
- ``scene_table.csv`` — top-N rows with all metrics.
- ``scene_table.md``  — same table formatted as Markdown for the thesis.
- ``scene_quality_log.txt`` — human-readable run log.

Usage
-----
    python scripts/eval_scene_quality_mini.py \\
        --bbox -8.6 36.85 -7.7 37.30 \\
        --date-range 2023-06-01/2025-10-31 \\
        --max-cloud 10 \\
        --top 15

Notes
-----
* Network calls are wrapped in try/except — a single bad scene is logged and
  skipped, never crashes the whole run.
* Only one new file is created.  All outputs go to ``outputs/reef_habitat_mini/``;
  the existing ``outputs/reef_habitat/`` is never touched.
* The BVI heuristic here is intentionally simple — it is a *ranking* tool, not
  a measurement.  The production BVI used in the ML pipeline is much more
  elaborate (see ``src/reef_ml_predictor_acolite.py``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from pystac_client import Client
    _HAS_PYSTAC = True
except Exception:  # pragma: no cover
    _HAS_PYSTAC = False

import requests

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("eval_scene_quality_mini")

# ── Constants mirroring the baseline ───────────────────────────────────────────
# (kept here verbatim to avoid coupling the new file to import-time side
#  effects in src/s2_patch_extractor.py)
_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"

# B04 aliases in the Earth Search asset dictionary
_B04_ALIASES = ["red", "B04", "B4"]

# S2 L2A DN scaling → BOA reflectance
_DN_SCALE = 10000.0


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class SceneStats:
    """Per-scene quality metrics."""
    rank: int
    scene_id: str
    datetime: str
    platform: str
    cloud_cover_pct: float
    mean_B04: float
    std_B04: float
    snr_db: float
    bvi_score: float


# ── Validation helpers ─────────────────────────────────────────────────────────
def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    lon_min, lat_min, lon_max, lat_max = bbox
    if not (lon_min < lon_max):
        raise ValueError(
            f"Invalid bbox: lon_min ({lon_min}) must be < lon_max ({lon_max})"
        )
    if not (lat_min < lat_max):
        raise ValueError(
            f"Invalid bbox: lat_min ({lat_min}) must be < lat_max ({lat_max})"
        )
    if not (-180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0):
        raise ValueError(f"Invalid bbox longitudes (must be in [-180, 180]): {bbox}")
    if not (-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0):
        raise ValueError(f"Invalid bbox latitudes (must be in [-90, 90]): {bbox}")


_DATE_RANGE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$")


def _validate_date_range(date_range: str) -> None:
    if not _DATE_RANGE_RE.match(date_range):
        raise ValueError(
            f"Invalid --date-range '{date_range}'. "
            f"Expected format 'YYYY-MM-DD/YYYY-MM-DD'."
        )


# ── STAC query ─────────────────────────────────────────────────────────────────
def _search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: str,
    max_cloud: float,
    max_items: int,
) -> list[dict]:
    """Return a list of STAC item dicts (id, datetime, cloud_cover, platform, assets).

    Uses pystac-client when available; falls back to the raw STAC /search
    endpoint otherwise.  Pattern adapted from
    ``src.s2_patch_extractor._find_best_scene``.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    intersects = {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]],
    }

    if _HAS_PYSTAC:
        try:
            cat = Client.open(_EARTH_SEARCH)
            results = cat.search(
                collections=[_S2_COLLECTION],
                intersects=intersects,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": max_cloud}},
                sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
                max_items=max_items,
            )
            items = list(results.items())
            return [
                {
                    "id":          it.id,
                    "datetime":    it.properties.get("datetime", ""),
                    "cloud_cover": it.properties.get("eo:cloud_cover", 99.0),
                    "platform":    it.properties.get("platform", "unknown"),
                    "assets":      {k: v.href for k, v in it.assets.items()},
                }
                for it in items
            ]
        except Exception as exc:
            log.warning("pystac-client search failed (%s) — falling back to raw API", exc)

    # Raw API fallback
    payload = {
        "collections": [_S2_COLLECTION],
        "intersects":  intersects,
        "datetime":    date_range,
        "query":       {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby":      [{"field": "eo:cloud_cover", "direction": "asc"}],
        "limit":       max_items,
    }
    try:
        resp = requests.post(f"{_EARTH_SEARCH}/search", json=payload, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        return [
            {
                "id":          f["id"],
                "datetime":    f["properties"].get("datetime", ""),
                "cloud_cover": f["properties"].get("eo:cloud_cover", 99.0),
                "platform":    f["properties"].get("platform", "unknown"),
                "assets":      {k: v["href"] for k, v in f["assets"].items()},
            }
            for f in features
        ]
    except Exception as exc:
        log.error("Raw STAC API search failed: %s", exc)
        return []


# ── Band download (windowed COG read) ──────────────────────────────────────────
def _resolve_b04_href(assets: dict) -> Optional[str]:
    """Return the B04 asset href from the Earth Search asset dict (case-insensitive)."""
    aliases_lower = {a.lower() for a in _B04_ALIASES}
    for key, href in assets.items():
        if key.lower() in aliases_lower:
            return href
    return None


def _read_b04_window(
    href: str,
    centre_lon: float,
    centre_lat: float,
    half_side_deg: float,
) -> Optional[np.ndarray]:
    """Open the B04 COG at *href* and read a small window centred on (centre_lon, centre_lat).

    The window is computed in geographic coordinates (assumed EPSG:4326) and
    translated to a rasterio window via the dataset's transform.  No full file
    is downloaded — only the requested tile range is fetched.

    Returns a 1-D float32 array of valid pixels in BOA reflectance units
    (DN / 10000), or None on failure.
    """
    try:
        with rasterio.open(href) as src:
            # Translate geographic centre → raster pixel coordinates
            try:
                px, py = src.index(centre_lon, centre_lat)
            except Exception:
                # Fallback: use dataset bounds to construct a windowed read
                # around the centre directly
                px, py = src.width // 2, src.height // 2

            # Determine a window in pixel space corresponding to ~half_side_deg.
            # Use the dataset transform: 1 deg ≈ 1 / transform.a (lon) or
            # 1 / transform.e (lat) pixels at this latitude.
            try:
                deg_per_px_x = abs(src.transform.a)
                deg_per_px_y = abs(src.transform.e)
            except Exception:
                deg_per_px_x = deg_per_px_y = 1.0 / 1000.0  # very small fallback

            # Guard against zero divisors
            if deg_per_px_x <= 0:
                deg_per_px_x = 1.0 / 1000.0
            if deg_per_px_y <= 0:
                deg_per_px_y = 1.0 / 1000.0

            half_px = max(1, int(half_side_deg / max(deg_per_px_x, deg_per_px_y)))

            col_off = max(0, px - half_px)
            row_off = max(0, py - half_px)
            col_end = min(src.width,  px + half_px)
            row_end = min(src.height, py + half_px)

            if col_end <= col_off or row_end <= row_off:
                log.debug("Empty window for href=%s", href)
                return None

            window = Window.from_slices(rows=(row_off, row_end),
                                        cols=(col_off, col_end))
            data = src.read(1, window=window)

            # Mask invalid pixels: nodata, zero, negative
            nodata = src.nodata
            valid = np.ones(data.shape, dtype=bool)
            if nodata is not None:
                valid &= (data != nodata)
            valid &= (data > 0)

            if not np.any(valid):
                return None

            reflect = data[valid].astype(np.float32) / _DN_SCALE
            return reflect

    except Exception as exc:
        log.warning("B04 windowed read failed for %s: %s", href, exc)
        return None


# ── Metric computation ─────────────────────────────────────────────────────────
def _compute_bvi(snr_db: float, cloud_cover_pct: float) -> float:
    """Heuristic bottom-visibility proxy in [0, 1].

    Formula (documented in module docstring):

        bvi = clip(snr_db / 50.0, 0, 1) * clip(1 - cloud_cover_pct/100, 0, 1)

    This is a *ranking* heuristic, not a measurement.  The 50 dB reference
    corresponds roughly to a CV ≈ 0.32% — well above noise floor for BOA
    reflectance over deep clear water.
    """
    snr_term = max(0.0, min(1.0, snr_db / 50.0))
    cloud_term = max(0.0, min(1.0, 1.0 - cloud_cover_pct / 100.0))
    return snr_term * cloud_term


def _rank_scenes(
    rows: list[dict],
    top: int,
) -> list[SceneStats]:
    """Sort by bvi_score desc, tie-break by cloud_cover_pct asc; keep top-N."""
    rows_sorted = sorted(
        rows,
        key=lambda r: (-r["bvi_score"], r["cloud_cover_pct"]),
    )
    out: list[SceneStats] = []
    for i, r in enumerate(rows_sorted[:top], start=1):
        out.append(SceneStats(
            rank=i,
            scene_id=r["scene_id"],
            datetime=r["datetime"],
            platform=r["platform"],
            cloud_cover_pct=round(float(r["cloud_cover_pct"]), 3),
            mean_B04=round(float(r["mean_B04"]), 6),
            std_B04=round(float(r["std_B04"]), 6),
            snr_db=round(float(r["snr_db"]), 3),
            bvi_score=round(float(r["bvi_score"]), 6),
        ))
    return out


# ── Output writers ─────────────────────────────────────────────────────────────
def _write_csv(rows: list[SceneStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "rank", "scene_id", "datetime", "platform", "cloud_cover_pct",
        "mean_B04", "std_B04", "snr_db", "bvi_score",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    log.info("Wrote CSV: %s  (%d rows)", path, len(rows))


def _write_md(rows: list[SceneStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "scene_id", "datetime", "platform", "cloud_cover_pct",
            "mean_B04", "std_B04", "snr_db", "bvi_score"]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(asdict(r)[c]) for c in cols) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Wrote Markdown table: %s", path)


def _write_log(
    log_path: Path,
    args: argparse.Namespace,
    n_found: int,
    n_failed: int,
    rows: list[SceneStats],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log_path, "w") as f:
        f.write("Scene-quality ranking run\n")
        f.write(f"  timestamp (UTC) : {ts}\n")
        f.write(f"  bbox            : {args.bbox}\n")
        f.write(f"  date_range      : {args.date_range}\n")
        f.write(f"  max_cloud (%)   : {args.max_cloud}\n")
        f.write(f"  sample_bbox_deg : {args.sample_bbox}\n")
        f.write(f"  max_items       : {args.max_items}\n")
        f.write(f"  top             : {args.top}\n")
        f.write(f"  out_dir         : {args.out_dir}\n")
        f.write(f"\n  scenes found    : {n_found}\n")
        f.write(f"  scenes failed   : {n_failed}\n")
        f.write(f"  rows written    : {len(rows)}\n")
        f.write("\n── Top scenes ──\n")
        for r in rows:
            f.write(
                f"  #{r.rank:>2}  {r.scene_id}  cloud={r.cloud_cover_pct:.2f}%  "
                f"B04={r.mean_B04:.4f}±{r.std_B04:.4f}  SNR={r.snr_db:6.2f}dB  "
                f"BVI={r.bvi_score:.4f}\n"
            )
    log.info("Wrote log: %s", log_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_scene_quality_mini",
        description=(
            "Rank Sentinel-2 L2A scenes over the Algarve bbox by a composite "
            "optical quality score (BVI heuristic). Writes a CSV + Markdown "
            "table of the top-N scenes to outputs/reef_habitat_mini/scene_ranking/."
        ),
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[-8.6, 36.85, -7.7, 37.30],
                   help="Algarve bbox (lon_min lat_min lon_max lat_max). Default: -8.6 36.85 -7.7 37.30")
    p.add_argument("--date-range", type=str, default="2023-06-01/2025-10-31",
                   help="STAC datetime filter, 'YYYY-MM-DD/YYYY-MM-DD'. Default: 2023-06-01/2025-10-31")
    p.add_argument("--max-cloud", type=float, default=10.0,
                   help="Maximum cloud cover percent. Default: 10.0")
    p.add_argument("--top", type=int, default=15,
                   help="Number of top scenes to write to the output table. Default: 15")
    p.add_argument("--out-dir", type=str, default="outputs/reef_habitat_mini/scene_ranking",
                   help="Output directory (isolated from outputs/reef_habitat/).")
    p.add_argument("--sample-bbox", type=float, default=0.02,
                   help="Half-side of the per-scene sample window in degrees. Default: 0.02")
    p.add_argument("--max-items", type=int, default=50,
                   help="Maximum number of STAC items to consider. Default: 50")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    # Mandatory safety banner — must be the very first stdout line.
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")
    sys.stdout.flush()

    args = _parse_args(argv)
    bbox = tuple(args.bbox)  # type: tuple[float, float, float, float]
    _validate_bbox(bbox)
    _validate_date_range(args.date_range)

    if args.top <= 0:
        raise ValueError(f"--top must be > 0 (got {args.top})")
    if args.max_cloud < 0 or args.max_cloud > 100:
        raise ValueError(f"--max-cloud must be in [0, 100] (got {args.max_cloud})")
    if args.sample_bbox <= 0:
        raise ValueError(f"--sample-bbox must be > 0 (got {args.sample_bbox})")
    if args.max_items <= 0:
        raise ValueError(f"--max-items must be > 0 (got {args.max_items})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    centre_lon = 0.5 * (bbox[0] + bbox[2])
    centre_lat = 0.5 * (bbox[1] + bbox[3])

    log.info("Searching STAC: bbox=%s  range=%s  max_cloud=%s",
             bbox, args.date_range, args.max_cloud)

    scenes = _search_scenes(bbox, args.date_range, args.max_cloud, args.max_items)
    log.info("STAC returned %d candidate scenes", len(scenes))

    raw_rows: list[dict] = []
    n_failed = 0

    for sc in scenes:
        try:
            href = _resolve_b04_href(sc["assets"])
            if href is None:
                log.warning("Scene %s has no B04/red asset — skipping", sc["id"])
                n_failed += 1
                continue

            reflect = _read_b04_window(href, centre_lon, centre_lat, args.sample_bbox)
            if reflect is None or reflect.size < 10:
                log.warning("Scene %s produced <10 valid B04 pixels — skipping", sc["id"])
                n_failed += 1
                continue

            mean_v = float(np.mean(reflect))
            std_v  = float(np.std(reflect))
            snr_db = 20.0 * np.log10(mean_v / (std_v + 1e-6))
            bvi    = _compute_bvi(snr_db, float(sc["cloud_cover"]))

            raw_rows.append({
                "scene_id":       sc["id"],
                "datetime":       (sc["datetime"] or "")[:19],
                "platform":       sc["platform"],
                "cloud_cover_pct": float(sc["cloud_cover"]),
                "mean_B04":       mean_v,
                "std_B04":        std_v,
                "snr_db":         snr_db,
                "bvi_score":      bvi,
            })
        except Exception as exc:
            log.warning("Unexpected failure on scene %s: %s", sc.get("id", "?"), exc)
            n_failed += 1
            continue

    log.info("Computed metrics for %d scenes (failed=%d)", len(raw_rows), n_failed)

    top_rows = _rank_scenes(raw_rows, top=args.top)
    log.info("Top-%d scenes selected", len(top_rows))

    _write_csv(top_rows, out_dir / "scene_table.csv")
    _write_md(top_rows,  out_dir / "scene_table.md")
    _write_log(out_dir / "scene_quality_log.txt", args,
               n_found=len(scenes), n_failed=n_failed, rows=top_rows)

    # ── Final stdout summary (top 5) ──────────────────────────────────────────
    print("\n── Scene-quality ranking summary (top 5) ────────────────────────")
    print(f"  Bbox: {bbox}")
    print(f"  Date range: {args.date_range}    max-cloud: {args.max_cloud}%")
    print(f"  Scenes considered: {len(scenes)}    metrics ok: {len(raw_rows)}    failed: {n_failed}")
    print(f"  Outputs: {out_dir}")
    print()
    print(f"  {'rank':>4}  {'scene_id':<35}  {'date':<10}  {'cloud':>6}  {'B04_mean':>9}  {'SNR_dB':>7}  {'BVI':>6}")
    for r in top_rows[:5]:
        print(
            f"  {r.rank:>4}  {r.scene_id:<35}  {r.datetime[:10]:<10}  "
            f"{r.cloud_cover_pct:>5.2f}%  {r.mean_B04:>9.4f}  {r.snr_db:>7.2f}  {r.bvi_score:>6.3f}"
        )
    print("────────────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    sys.exit(main())