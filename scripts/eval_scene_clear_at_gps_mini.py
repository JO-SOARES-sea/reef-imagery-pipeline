#!/usr/bin/env python3
"""
eval_scene_clear_at_gps_mini.py — Sidecar: does the cloud miss the reef?

This script answers a different question from ``eval_scene_quality_mini_v4.py``:
a Sentinel-2 scene with high scene-level cloud cover can still be perfect for
a specific reef if the cloud is elsewhere in the tile.  Conversely, a scene
with low scene-level cloud cover can still have a cloud AT the reef GPS.

It scores each candidate scene by reading the **Scene Classification Layer
(SCL)** band of each scene directly at the reef GPS coordinates, and
classifying the pixel as ``clear`` (SCL ∈ {4, 5, 6}) or ``cloudy``
(SCL ∈ {8, 9, 10} or shadow {3}).

This is the operational metric that the HTML report was already computing
("Cloud GPS=0.0%" labels) — now extracted into a reproducible sidecar.

SCL class reference (Copernicus Sentinel-2 L2A):
  0  NO_DATA
  1  SATURATED_OR_DEFECTIVE
  2  DARK_AREA_PIXELS
  3  CLOUD_SHADOWS         → treated as "not clear"
  4  VEGETATION            → clear (land)
  5  NOT_VEGETATED         → clear (land)
  6  WATER                → clear (water — what we want at a reef)
  7  UNCLASSIFIED          → unknown, treated as "not clear"
  8  CLOUD_MEDIUM_PROBABILITY
  9  CLOUD_HIGH_PROBABILITY
  10 THIN_CIRRUS
  11 SNOW

Inputs
------
- Sentinel-2 L2A STAC query (Earth Search)
- A list of reef GPS coordinates (default: imports REEF_SITES from
  ``scripts.build_reef_dataset`` so the baseline is reused)

Outputs (under ``--out-dir``, default ``outputs/reef_habitat_mini/scene_clear_at_gps/``)
------------------------------------------------------------------------------------
- ``scene_clear_at_gps.csv`` — one row per scene: scene_id, datetime,
  platform, cloud_cover_pct, SCL_at_each_site, all_clear (bool)
- ``scene_clear_at_gps.md``  — same table as Markdown
- ``scene_clear_at_gps_log.txt`` — human-readable summary

Example
-------
    python scripts/eval_scene_clear_at_gps_mini.py \\
        --date-range 2023-08-25/2025-10-31 \\
        --max-cloud 5 \\
        --top 20

Notes
-----
* SAFE MODE: this script ONLY creates new files in ``outputs/reef_habitat_mini/``.
  The baseline ``outputs/reef_habitat/`` is never touched.
* The script imports ``REEF_SITES`` from ``scripts.build_reef_dataset`` (read-only).
  No baseline file is modified.
* A scene is ``all_clear`` iff every REEF_SITES pixel returns SCL ∈ {4, 5, 6}.
* This is a **complement** to ``eval_scene_quality_mini_v4.py`` — v4 ranks by
  measured B04 SNR/BVI over a small window; this script filters by SCL at
  the exact reef GPS.  Both should be inspected together.
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

import rasterio
from rasterio.windows import Window

try:
    from pystac_client import Client
    _HAS_PYSTAC = True
except Exception:  # pragma: no cover
    _HAS_PYSTAC = False

import requests
try:
    from pyproj import Transformer as _PyprojTransformer
    _HAS_PYPROJ = True
except Exception:  # pragma: no cover
    _HAS_PYPROJ = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("eval_scene_clear_at_gps_mini")

# ── Constants ──────────────────────────────────────────────────────────────────
_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"

# SCL alias on Earth Search
_SCL_ALIASES = ["scl", "SCL", "scene_classification"]

# SCL class membership
_CLEAR_SCL = {4, 5, 6}
_CLOUD_SCL = {3, 7, 8, 9, 10}  # 3=shadow, 7=unclassified, 8/9/10=cloud
_OTHER_SCL = {0, 1, 2, 11}


@dataclass
class SiteSCL:
    """SCL classification at a single GPS site."""
    name: str
    lon: float
    lat: float
    scl_value: int
    classification: str  # "clear" | "cloud" | "shadow" | "other" | "out_of_tile"


@dataclass
class SceneClearReport:
    """Per-scene SCL at each GPS site."""
    rank: int
    scene_id: str
    datetime: str
    platform: str
    cloud_cover_pct: float
    sites: list[SiteSCL] = field(default_factory=list)
    all_clear: bool = False
    n_clear: int = 0
    n_cloud: int = 0


# ── STAC query (mirrors v4 to keep consistent) ────────────────────────────────
def _search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: str,
    max_cloud: float,
    max_items: int,
    *,
    tile_filter: Optional[str] = "29SNB",
) -> list[dict]:
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
                max_items=max_items,
            )
            items = list(results.items())
            if tile_filter:
                items = [it for it in items if tile_filter in it.id]
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
        "limit":       max_items,
    }
    try:
        resp = requests.post(f"{_EARTH_SEARCH}/search", json=payload, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if tile_filter:
            features = [f for f in features if tile_filter in f["id"]]
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


# ── SCL pixel read at GPS ──────────────────────────────────────────────────────
def _resolve_scl_href(assets: dict) -> Optional[str]:
    aliases_lower = {a.lower() for a in _SCL_ALIASES}
    for key, href in assets.items():
        if key.lower() in aliases_lower:
            return href
    return None


def _classify_scl(scl: int) -> str:
    if scl in _CLEAR_SCL:
        return "clear"
    if scl == 3:
        return "shadow"
    if scl in {8, 9, 10}:
        return "cloud"
    if scl in _OTHER_SCL:
        return "other"
    return "unknown"


def _read_scl_at_lonlat(href: str, lon: float, lat: float) -> Optional[int]:
    """Open the SCL COG and read a 1-pixel value at (lon, lat)."""
    try:
        with rasterio.open(href) as src:
            dst_crs = src.crs
            epsg = dst_crs.to_epsg() if dst_crs is not None else None
            if epsg is not None and epsg != 4326 and _HAS_PYPROJ:
                xformer = _PyprojTransformer.from_crs(
                    "EPSG:4326", f"EPSG:{epsg}", always_xy=True,
                )
                cx, cy = xformer.transform(lon, lat)
            else:
                cx, cy = lon, lat
            # SCL is 20 m, but we want 1 pixel.  Use src.index + window of 1.
            try:
                px, py = src.index(cx, cy)
            except Exception:
                return None
            if not (0 <= px < src.width and 0 <= py < src.height):
                return None
            data = src.read(
                1,
                window=Window(col_off=px, row_off=py, width=1, height=1),
            )
            if data.size == 0:
                return None
            return int(data[0, 0])
    except Exception as exc:
        log.debug("SCL read failed for %s at (%.4f, %.4f): %s", href, lon, lat, exc)
        return None


# ── Reef sites loader (import, do not copy) ────────────────────────────────────
def _load_reef_sites() -> list[dict]:
    """Import REEF_SITES from scripts.build_reef_dataset (read-only)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.build_reef_dataset import REEF_SITES  # type: ignore
        return list(REEF_SITES)
    except Exception as exc:
        log.warning("Could not import REEF_SITES from scripts.build_reef_dataset: %s", exc)
        log.warning("Using fallback (Pedra Sta Eulália only).")
        return [{"name": "Pedra_Sta_Eulalia_fallback", "lon": -8.1695, "lat": 37.0643}]


# ── Per-scene evaluation ──────────────────────────────────────────────────────
def _evaluate_scene(scene: dict, sites: list[dict]) -> SceneClearReport:
    rep = SceneClearReport(
        rank=0,
        scene_id=scene["id"],
        datetime=scene["datetime"],
        platform=scene["platform"],
        cloud_cover_pct=round(float(scene["cloud_cover"]), 3),
    )
    assets = scene["assets"]
    scl_href = _resolve_scl_href(assets)
    if scl_href is None:
        log.warning("Scene %s has no SCL asset — marking all sites as out_of_tile", scene["id"])
        for s in sites:
            rep.sites.append(SiteSCL(
                name=s["name"], lon=s["lon"], lat=s["lat"],
                scl_value=-1, classification="out_of_tile",
            ))
        return rep

    for s in sites:
        val = _read_scl_at_lonlat(scl_href, s["lon"], s["lat"])
        if val is None:
            rep.sites.append(SiteSCL(
                name=s["name"], lon=s["lon"], lat=s["lat"],
                scl_value=-1, classification="out_of_tile",
            ))
        else:
            cls = _classify_scl(val)
            rep.sites.append(SiteSCL(
                name=s["name"], lon=s["lon"], lat=s["lat"],
                scl_value=val, classification=cls,
            ))
            if cls == "clear":
                rep.n_clear += 1
            else:
                rep.n_cloud += 1
    rep.all_clear = (rep.n_clear == len(sites))
    return rep


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_scene_clear_at_gps_mini",
        description="Score Sentinel-2 scenes by SCL clear/cloud at reef GPS points.",
    )
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[-8.6, 36.85, -7.7, 37.30],
                   help="Algarve bbox (default: -8.6 36.85 -7.7 37.30)")
    p.add_argument("--date-range", type=str, default="2023-08-25/2025-10-31",
                   help="STAC datetime filter (default: 2023-08-25/2025-10-31)")
    p.add_argument("--max-cloud", type=float, default=5.0,
                   help="Max scene-level cloud %% (default: 5.0)")
    p.add_argument("--top", type=int, default=20,
                   help="Number of scenes in output table (default: 20)")
    p.add_argument("--max-items", type=int, default=50,
                   help="Max STAC items per query page (default: 50)")
    p.add_argument("--tile-filter", type=str, default="29SNB",
                   help="MGRS tile filter (default: 29SNB). Pass '' to disable.")
    p.add_argument("--out-dir", type=str,
                   default="outputs/reef_habitat_mini/scene_clear_at_gps",
                   help="Output directory (isolated from outputs/reef_habitat/).")
    return p.parse_args(argv)


def _rank(reports: list[SceneClearReport], top: int) -> list[SceneClearReport]:
    """Sort: all_clear first, then by n_clear desc, then cloud_cover_pct asc."""
    out = sorted(
        reports,
        key=lambda r: (not r.all_clear, -r.n_clear, r.cloud_cover_pct),
    )
    for i, r in enumerate(out[:top], start=1):
        r.rank = i
    return out[:top]


# ── Output writers ────────────────────────────────────────────────────────────
def _write_csv(reports: list[SceneClearReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "scene_id", "datetime", "platform", "cloud_cover_pct",
        "n_clear", "n_cloud", "all_clear",
    ]
    if reports and reports[0].sites:
        fieldnames += [f"scl_{s.name}" for s in reports[0].sites]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in reports:
            row = {
                "rank":             r.rank,
                "scene_id":         r.scene_id,
                "datetime":         r.datetime,
                "platform":         r.platform,
                "cloud_cover_pct":  r.cloud_cover_pct,
                "n_clear":          r.n_clear,
                "n_cloud":          r.n_cloud,
                "all_clear":        r.all_clear,
            }
            for s in r.sites:
                row[f"scl_{s.name}"] = f"{s.scl_value}({s.classification})"
            w.writerow(row)
    log.info("Wrote CSV: %s  (%d rows)", path, len(reports))


def _write_md(reports: list[SceneClearReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "scene_id", "datetime", "platform", "cloud_cover_pct",
            "n_clear", "n_cloud", "all_clear"]
    if reports and reports[0].sites:
        cols += [f"scl_{s.name}" for s in reports[0].sites]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in reports:
        cells = [
            str(r.rank), r.scene_id, r.datetime[:10], r.platform,
            f"{r.cloud_cover_pct:.2f}", str(r.n_clear), str(r.n_cloud),
            "✓" if r.all_clear else "✗",
        ]
        for s in r.sites:
            cells.append(f"{s.scl_value}({s.classification})")
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Wrote Markdown: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")
    sys.stdout.flush()

    args = _parse_args(argv)
    bbox = tuple(args.bbox)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"Invalid bbox: {bbox}")

    sites = _load_reef_sites()
    log.info("Loaded %d reef sites: %s",
             len(sites), ", ".join(s["name"] for s in sites))

    log.info("Searching STAC: bbox=%s  range=%s  max_cloud=%s",
             bbox, args.date_range, args.max_cloud)
    items = _search_scenes(
        bbox, args.date_range, args.max_cloud, args.max_items,
        tile_filter=args.tile_filter or None,
    )
    log.info("STAC returned %d candidate scenes", len(items))

    reports: list[SceneClearReport] = []
    failed = 0
    for it in items:
        try:
            rep = _evaluate_scene(it, sites)
            reports.append(rep)
        except Exception as exc:
            log.warning("Scene %s failed: %s", it["id"], exc)
            failed += 1

    top = _rank(reports, args.top)
    out_dir = Path(args.out_dir)
    _write_csv(top, out_dir / "scene_clear_at_gps.csv")
    _write_md(top,  out_dir / "scene_clear_at_gps.md")

    log_path = out_dir / "scene_clear_at_gps_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log_path, "w") as f:
        f.write("Scene clear-at-GPS evaluation\n")
        f.write(f"  timestamp (UTC) : {ts}\n")
        f.write(f"  bbox            : {bbox}\n")
        f.write(f"  date_range      : {args.date_range}\n")
        f.write(f"  max_cloud (%)   : {args.max_cloud}\n")
        f.write(f"  tile_filter     : {args.tile_filter!r}\n")
        f.write(f"  scenes total    : {len(items)}\n")
        f.write(f"  scenes scored   : {len(reports)}\n")
        f.write(f"  scenes failed   : {failed}\n")
        f.write(f"  reef sites      : {len(sites)}\n")
        f.write(f"  all_clear       : {sum(1 for r in reports if r.all_clear)}\n")
    log.info("Wrote log: %s", log_path)

    # Print summary
    n_all_clear = sum(1 for r in reports if r.all_clear)
    print("\n── Scene clear-at-GPS summary (top 10) ──────────────────────")
    print(f"  Bbox: {bbox}    max-cloud: {args.max_cloud}%    sites: {len(sites)}")
    print(f"  Scenes considered: {len(items)}    scored: {len(reports)}    "
          f"all_clear: {n_all_clear}    failed: {failed}")
    print(f"  Outputs: {out_dir}\n")
    hdr = ["rank", "scene_id", "date", "cloud%", "n_clr", "all?"]
    print("  " + "  ".join(f"{h:<22}" for h in hdr[:5]) + "  " + "  ".join(f"{h:<6}" for h in hdr[5:]))
    for r in top[:10]:
        print(f"  {r.rank:<4}  {r.scene_id:<32}  {r.datetime[:10]:<10}  "
              f"{r.cloud_cover_pct:5.2f}  {r.n_clear:>3}    "
              f"{'YES' if r.all_clear else 'no':<6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
