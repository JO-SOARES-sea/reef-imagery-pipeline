#!/usr/bin/env python3
"""Example STAC ingestion CLI for Sentinel-2 scenes in the Algarve.

This script demonstrates how to use the repo STAC helper to discover a Sentinel-2
scene, choose the least cloudy candidate, and print signed band URLs for B02/B03/B04.
"""

from __future__ import annotations

import argparse
import json
import os

from src.stac_ingest import (
    EARTH_SEARCH_STAC_URL,
    PC_STAC_URL,
    DEFAULT_S2_COLLECTION,
    get_asset_hrefs,
    choose_least_cloudy,
    search_sentinel2_scenes,
    scene_summary,
)

DEFAULT_LAT = 37.068978
DEFAULT_LON = -8.210328
DEFAULT_START = "2025-09-01"
DEFAULT_END = "2025-09-30"
DEFAULT_MAX_CLOUD = 20.0
DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentinel-2 STAC ingestion example for Algarve")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Site latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Site longitude")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD, help="Maximum STAC cloud cover")
    parser.add_argument(
        "--catalog",
        choices=["auto", "pc", "earthsearch"],
        default="auto",
        help="STAC endpoint to query; auto prefers Earth Search and falls back to Planetary Computer"
    )
    parser.add_argument("--output", default="sentinel2_stac_example.json", help="Optional metadata output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.catalog == "auto":
        catalog_url = None
    elif args.catalog == "pc":
        catalog_url = PC_STAC_URL
    else:
        catalog_url = EARTH_SEARCH_STAC_URL

    date_range = (args.start, args.end)
    scenes = search_sentinel2_scenes(
        args.lat,
        args.lon,
        date_range=date_range,
        max_cloud_cover=args.max_cloud,
        catalog_url=catalog_url,
        collection=DEFAULT_S2_COLLECTION,
        limit=10,
    )

    if not scenes:
        raise SystemExit("No Sentinel-2 scenes found for the requested window.")

    chosen = choose_least_cloudy(scenes)
    if chosen is None:
        raise SystemExit("No suitable Sentinel-2 scene could be selected.")

    summary = scene_summary(chosen)
    summary["assets"] = get_asset_hrefs(chosen, DEFAULT_BANDS)
    summary["catalog_url"] = catalog_url
    summary["search_bounds"] = {
        "lat": args.lat,
        "lon": args.lon,
        "start": args.start,
        "end": args.end,
    }

    print("Selected Sentinel-2 scene:")
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nWrote metadata to {args.output}")


if __name__ == "__main__":
    main()
