#!/usr/bin/env python3
"""
extract_fishing_spots_mini.py — Mine reef-adjacent fishing spots from a GPX file.

Local fishermen's GPX exports are a rich source of *implicit* reef labels: fish
aggregate around structural complexity, so a fishing spot in the optical depth
window is a strong prior for "rocky reef or seagrass bed at this location".

This script:
  1. Parses a GPX file (default: ``/Volumes/BACKUP/00000-GPS/CarlosBarrinha.gpx``).
  2. Filters waypoints to a bounding box (default: Algarve).
  3. Samples EMODnet depth at each waypoint via
     ``src.emodnet_bathymetry.sample_depth_at_point`` (import, do not copy).
  4. Keeps only waypoints whose EMODnet depth is in the optical window
     ``[--min-depth, --max-depth]`` (default: 3–25 m).
  5. Writes a CSV + GeoJSON of the surviving spots.

The CSV is an *additive* resource.  The user manually curates which spots
to add to ``REEF_SITES`` in the baseline — this script never edits the
baseline.

Inputs
------
- GPX file (any path; the default is the user's local Carlos Barrinha export)
- EMODnet depth GeoTIFF (default: ``outputs/reef_habitat/emodnet/depth_algarve.tif``)

Outputs (under ``--out-dir``, default ``outputs/reef_habitat_mini/fishing_spots/``)
--------------------------------------------------------------------------------
- ``fishing_spots_filtered.csv``  — waypoints inside bbox AND in depth window
- ``fishing_spots_all.csv``       — all waypoints inside bbox (with EMODnet depth)
- ``fishing_spots_filtered.geojson`` — same as the CSV, for QGIS / Leaflet
- ``fishing_spots_log.txt``       — counts and bbox used

Example
-------
    python scripts/extract_fishing_spots_mini.py \\
        --gpx /Volumes/BACKUP/00000-GPS/CarlosBarrinha.gpx \\
        --depth outputs/reef_habitat/emodnet/depth_algarve.tif \\
        --min-depth 3 --max-depth 25

Notes
-----
* SAFE MODE: this script ONLY creates new files in
  ``outputs/reef_habitat_mini/``.  The baseline ``outputs/reef_habitat/`` is
  never touched.  The GPX file is read-only.
* Depth is taken from EMODnet, NOT from the GPX ``wptx1:Depth`` extension
  (which is always 0.00 in the default file and is unreliable).
* ``sample_depth_at_point`` is imported from ``src.emodnet_bathymetry``
  (single source of truth for depth lookup).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("extract_fishing_spots_mini")

# GPX 1.1 namespaces
_GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1",
           "wptx1": "http://www.garmin.com/xmlschemas/WaypointExtension/v1"}


# ── GPX parsing ───────────────────────────────────────────────────────────────
@dataclass
class Waypoint:
    name: str
    lon: float
    lat: float
    time: Optional[str] = None
    gpx_depth: Optional[float] = None
    emodnet_depth_m: Optional[float] = None
    in_optical_window: bool = False
    in_bbox: bool = False


def _parse_gpx(gpx_path: Path) -> list[Waypoint]:
    """Parse a GPX file into a list of Waypoint records (read-only)."""
    log.info("Parsing GPX: %s", gpx_path)
    tree = ET.parse(str(gpx_path))
    root = tree.getroot()
    wpts = []
    for w in root.findall("gpx:wpt", _GPX_NS):
        try:
            lat = float(w.get("lat"))
            lon = float(w.get("lon"))
        except (TypeError, ValueError):
            continue
        name_el = w.find("gpx:name", _GPX_NS)
        name = name_el.text.strip() if name_el is not None and name_el.text else ""
        time_el = w.find("gpx:time", _GPX_NS)
        time = time_el.text if time_el is not None and time_el.text else None
        # Depth in extensions (often 0.00 in the default file — unreliable)
        d_el = w.find(
            "gpx:extensions/wptx1:WaypointExtension/wptx1:Depth", _GPX_NS,
        )
        d_val: Optional[float] = None
        if d_el is not None and d_el.text:
            try:
                d_val = float(d_el.text)
            except ValueError:
                d_val = None
        wpts.append(Waypoint(name=name, lon=lon, lat=lat, time=time, gpx_depth=d_val))
    log.info("Parsed %d waypoints", len(wpts))
    return wpts


# ── Depth sampling (import, do not copy) ──────────────────────────────────────
def _sample_emodnet_depths(
    wpts: list[Waypoint],
    depth_path: Path,
) -> None:
    """Sample EMODnet depth at each waypoint.  Mutates the wpts in place."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.emodnet_bathymetry import sample_depth_at_point  # type: ignore
    except Exception as exc:
        log.error("Could not import sample_depth_at_point from src.emodnet_bathymetry: %s", exc)
        log.error("Cannot continue without depth sampling.  Aborting.")
        raise

    log.info("Sampling EMODnet depth at %d waypoints (raster=%s)", len(wpts), depth_path)
    n_ok = 0
    n_no_data = 0
    for w in wpts:
        d = sample_depth_at_point(depth_path, w.lon, w.lat)
        # sample_depth_at_point returns None for nodata or out-of-raster.
        # EMODnet depth is NEGATIVE below sea level (e.g., -15 for 15 m depth).
        if d is None:
            w.emodnet_depth_m = None
            n_no_data += 1
        else:
            w.emodnet_depth_m = float(d)
            n_ok += 1
    log.info("Depth sampling done: %d valid, %d nodata/out-of-raster", n_ok, n_no_data)


# ── Bbox + depth-window filter ────────────────────────────────────────────────
def _filter(
    wpts: list[Waypoint],
    bbox: tuple[float, float, float, float],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[list[Waypoint], list[Waypoint]]:
    """Returns (all_in_bbox, in_bbox_and_in_depth_window)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    in_bbox: list[Waypoint] = []
    survivors: list[Waypoint] = []
    for w in wpts:
        if not (lon_min <= w.lon <= lon_max and lat_min <= w.lat <= lat_max):
            continue
        w.in_bbox = True
        in_bbox.append(w)
        if w.emodnet_depth_m is None:
            continue
        # EMODnet depth is negative; window is [min_depth, max_depth] in absolute
        if -max_depth_m <= w.emodnet_depth_m <= -min_depth_m:
            w.in_optical_window = True
            survivors.append(w)
    return in_bbox, survivors


# ── Output writers ────────────────────────────────────────────────────────────
def _write_csv(wpts: list[Waypoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "lon", "lat", "time", "emodnet_depth_m",
                  "in_optical_window", "gpx_depth"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for wp in wpts:
            w.writerow({
                "name":              wp.name,
                "lon":               wp.lon,
                "lat":               wp.lat,
                "time":              wp.time or "",
                "emodnet_depth_m":   "" if wp.emodnet_depth_m is None
                                     else f"{wp.emodnet_depth_m:.3f}",
                "in_optical_window": wp.in_optical_window,
                "gpx_depth":         "" if wp.gpx_depth is None
                                     else f"{wp.gpx_depth:.2f}",
            })
    log.info("Wrote CSV: %s  (%d rows)", path, len(wpts))


def _write_geojson(wpts: list[Waypoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for wp in wpts:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [wp.lon, wp.lat]},
            "properties": {
                "name":              wp.name,
                "time":              wp.time,
                "emodnet_depth_m":   wp.emodnet_depth_m,
                "in_optical_window": wp.in_optical_window,
            },
        })
    geo = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(geo, f, indent=2)
    log.info("Wrote GeoJSON: %s  (%d features)", path, len(features))


def _write_log(
    log_path: Path,
    args: argparse.Namespace,
    n_total: int,
    n_in_bbox: int,
    n_no_depth: int,
    n_survivors: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log_path, "w") as f:
        f.write("Fishing-spot extraction run\n")
        f.write(f"  timestamp (UTC)        : {ts}\n")
        f.write(f"  gpx_path               : {args.gpx}\n")
        f.write(f"  emodnet_depth_raster   : {args.depth}\n")
        f.write(f"  bbox (lon_min lat_min lon_max lat_max) : {args.bbox}\n")
        f.write(f"  depth_window (m)       : [{args.min_depth}, {args.max_depth}]\n")
        f.write(f"  waypoints total        : {n_total}\n")
        f.write(f"  waypoints in bbox      : {n_in_bbox}\n")
        f.write(f"  waypoints with no depth: {n_no_depth}\n")
        f.write(f"  survivors (in window)  : {n_survivors}\n")
    log.info("Wrote log: %s", log_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="extract_fishing_spots_mini",
        description="Filter a fisherman's GPX to spots in S2's optical window.",
    )
    p.add_argument("--gpx", type=str,
                   default="/Volumes/BACKUP/00000-GPS/CarlosBarrinha.gpx",
                   help="Path to GPX file (default: CarlosBarrinha.gpx).")
    p.add_argument("--depth", type=str,
                   default="outputs/reef_habitat/emodnet/depth_algarve.tif",
                   help="EMODnet depth GeoTIFF (read-only, default: depth_algarve.tif).")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[-8.6, 36.85, -7.7, 37.30],
                   help="Bbox filter (default: Algarve).")
    p.add_argument("--min-depth", type=float, default=3.0,
                   help="Min depth in metres (default: 3.0).")
    p.add_argument("--max-depth", type=float, default=25.0,
                   help="Max depth in metres (default: 25.0 = S2 optical limit).")
    p.add_argument("--out-dir", type=str,
                   default="outputs/reef_habitat_mini/fishing_spots",
                   help="Output directory (isolated from outputs/reef_habitat/).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")
    sys.stdout.flush()

    args = _parse_args(argv)
    gpx_path = Path(args.gpx)
    if not gpx_path.exists():
        log.error("GPX file not found: %s", gpx_path)
        return 1
    depth_path = Path(args.depth)
    if not depth_path.exists():
        log.error("EMODnet depth raster not found: %s", depth_path)
        return 1
    if args.bbox[0] >= args.bbox[2] or args.bbox[1] >= args.bbox[3]:
        raise ValueError(f"Invalid bbox: {args.bbox}")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        raise ValueError(f"Invalid depth window: [{args.min_depth}, {args.max_depth}]")

    wpts = _parse_gpx(gpx_path)
    _sample_emodnet_depths(wpts, depth_path)
    in_bbox, survivors = _filter(
        wpts, tuple(args.bbox), args.min_depth, args.max_depth,
    )
    n_no_depth = sum(1 for w in in_bbox if w.emodnet_depth_m is None)
    out_dir = Path(args.out_dir)
    _write_csv(in_bbox,    out_dir / "fishing_spots_all.csv")
    _write_csv(survivors,  out_dir / "fishing_spots_filtered.csv")
    _write_geojson(survivors, out_dir / "fishing_spots_filtered.geojson")
    _write_log(out_dir / "fishing_spots_log.txt",
               args, len(wpts), len(in_bbox), n_no_depth, len(survivors))

    print("\n── Fishing-spot extraction summary ───────────────────────────")
    print(f"  GPX waypoints total       : {len(wpts)}")
    print(f"  Waypoints in bbox         : {len(in_bbox)}")
    print(f"  Waypoints with no depth   : {n_no_depth}  (land / out of raster)")
    print(f"  Survivors (in depth window: {args.min_depth}–{args.max_depth} m) : {len(survivors)}")
    print(f"  Outputs: {out_dir}")
    print("    fishing_spots_all.csv         (all in-bbox waypoints)")
    print("    fishing_spots_filtered.csv    (depth-window survivors — review these)")
    print("    fishing_spots_filtered.geojson (Leaflet / QGIS overlay)")
    print("──────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    sys.exit(main())
