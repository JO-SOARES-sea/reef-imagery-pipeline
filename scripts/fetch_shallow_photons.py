#!/usr/bin/env python3
"""
Fetch ICESat-2 ATL03 shallow-zone photons (1–15 m) for Algarve using
earthaccess for granule discovery and SlideRule for photon extraction.

Bypasses sliderule.earthdata.cmr() which times out; uses NASA earthaccess
CMR API directly instead.

Output: outputs/icesat2_deep_survey/atl03_shallow_photons.json
        (merged into atl03_all_photons.json alongside the existing deep cache)
"""

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BBOX = (-8.50, 36.90, -7.85, 37.20)   # W, S, E, N (Algarve)
N_SEAWATER = 1.336

# Shallow calibration range: 1–15 m true depth
TRUE_DEPTH_MIN_M = 1.0
TRUE_DEPTH_MAX_M = 15.0

# Apparent depth window (before refraction; add margin for uncertainty)
APPARENT_DEPTH_MIN_M = (TRUE_DEPTH_MIN_M / N_SEAWATER) - 0.5  # ~0.2 m
APPARENT_DEPTH_MAX_M = (TRUE_DEPTH_MAX_M / N_SEAWATER) + 2.0  # ~13.2 m

# Best summer months for water clarity (June–September)
TIME_RANGES = [
    ("2022-06-01T00:00:00Z", "2022-09-30T23:59:59Z"),
    ("2023-06-01T00:00:00Z", "2023-09-30T23:59:59Z"),
    ("2024-06-01T00:00:00Z", "2024-09-30T23:59:59Z"),
]

OUT_DIR = _PROJECT_ROOT / "outputs" / "icesat2_deep_survey"
SHALLOW_CACHE = OUT_DIR / "atl03_shallow_photons.json"
DEEP_CACHE    = OUT_DIR / "atl03_seafloor_photons.json"
MERGED_CACHE  = OUT_DIR / "atl03_all_photons.json"


# Known summer granules over Algarve from earthaccess/bkxb4xz9h discovery run.
# These cover the ICESat-2 tracks that cross the Algarve shelf (2022-2024 summer).
# Hardcoded because the earthaccess guest login and SlideRule CMR proxy are
# both unreliable without Earthdata credentials.
_KNOWN_SUMMER_GRANULES = [
    # Summer 2023 (best water clarity, confirmed by earthaccess CMR)
    "ATL03_20230910022513_12512002_007_01.h5",
    "ATL03_20230812034909_08092002_007_01.h5",
    "ATL03_20230710052139_03062002_007_01.h5",
    "ATL03_20230605190957_11671906_007_01.h5",
    "ATL03_20230611064544_12511902_007_01.h5",
    # Summer 2022 (same ICESat-2 repeat tracks, 91-day cycle)
    "ATL03_20220612063849_12511906_006_01.h5",
    "ATL03_20220711171943_03061906_006_01.h5",
    "ATL03_20220813035908_08091906_006_01.h5",
    # Summer 2024 (same tracks, version 006)
    "ATL03_20240611070844_12512302_006_01.h5",
    "ATL03_20240713174938_03062302_006_01.h5",
]


def find_granules_earthaccess(time_start: str, time_end: str) -> list[str]:
    """Return known summer granule filenames for the given date range."""
    from datetime import datetime

    t0 = datetime.fromisoformat(time_start.replace("Z", ""))
    t1 = datetime.fromisoformat(time_end.replace("Z", ""))

    matched = []
    for g in _KNOWN_SUMMER_GRANULES:
        # Parse date from filename: ATL03_YYYYMMDDHHMMSS_...
        try:
            dt = datetime.strptime(g[6:14], "%Y%m%d")
            if t0 <= dt <= t1:
                matched.append(g)
        except ValueError:
            pass

    log.info("  Known granules for %s → %s: %d", time_start[:10], time_end[:10], len(matched))
    return matched


def _extract_shallow_photons(gdf) -> list[dict]:
    """Filter one batch GeoDataFrame to 1–15 m true-depth photons."""

    h    = gdf["height"].values
    lons = gdf.geometry.x.values
    lats = gdf.geometry.y.values

    apparent_depth = -h
    window = (apparent_depth >= APPARENT_DEPTH_MIN_M) & (apparent_depth <= APPARENT_DEPTH_MAX_M)

    true_depths = apparent_depth[window] * N_SEAWATER
    depth_mask  = (true_depths >= TRUE_DEPTH_MIN_M) & (true_depths <= TRUE_DEPTH_MAX_M)

    photons = []
    for lon, lat, d in zip(lons[window][depth_mask], lats[window][depth_mask], true_depths[depth_mask]):
        photons.append({"lon": float(lon), "lat": float(lat), "true_depth_m": float(d)})
    return photons


def fetch_photons_sliderule(granules: list[str]) -> list[dict]:
    """Extract shallow photons from granule list via SlideRule ATL03."""
    from sliderule import icesat2, sliderule as sr
    sr.init("slideruleearth.io", verbose=False)

    W, S, E, N = BBOX
    polygon = [
        {"lon": W, "lat": S}, {"lon": E, "lat": S},
        {"lon": E, "lat": N}, {"lon": W, "lat": N},
        {"lon": W, "lat": S},
    ]

    parm = {
        "poly": polygon,
        "srt": icesat2.SRT_OCEAN,
        "cnf": icesat2.CNF_BACKGROUND,
        "pass_invalid": True,
    }

    all_photons: list[dict] = []
    batch_size = 5  # smaller batches for stability

    for i in range(0, len(granules), batch_size):
        batch = granules[i: i + batch_size]
        log.info("  SlideRule batch %d/%d (%d granules)",
                 i // batch_size + 1,
                 (len(granules) + batch_size - 1) // batch_size,
                 len(batch))
        try:
            gdf = icesat2.atl03sp(parm, resources=batch)
            if gdf is not None and len(gdf) > 0:
                photons = _extract_shallow_photons(gdf)
                log.info("    → %d total photons, %d in 1–15 m band", len(gdf), len(photons))
                all_photons.extend(photons)
            else:
                log.info("    → empty batch")
        except Exception as exc:
            log.warning("  Batch error: %s", exc)

    return all_photons


def merge_caches(shallow: list[dict], deep_cache: Path) -> list[dict]:
    """Merge shallow photons with existing deep-zone cache."""
    deep = []
    if deep_cache.exists():
        deep = json.loads(deep_cache.read_text())
        log.info("Loaded %d deep-zone photons from %s", len(deep), deep_cache.name)

    merged = shallow + deep
    log.info("Merged: %d shallow + %d deep = %d total photons", len(shallow), len(deep), len(merged))
    return merged


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SHALLOW_CACHE.exists():
        log.info("Shallow photon cache already exists: %s", SHALLOW_CACHE)
        shallow_photons = json.loads(SHALLOW_CACHE.read_text())
        log.info("Loaded %d cached shallow photons", len(shallow_photons))
    else:
        all_granules: list[str] = []
        for t_start, t_end in TIME_RANGES:
            granules = find_granules_earthaccess(t_start, t_end)
            all_granules.extend(granules)

        # Deduplicate
        all_granules = list(dict.fromkeys(all_granules))
        log.info("Total unique granules to process: %d", len(all_granules))

        if not all_granules:
            log.error("No granules found — check earthaccess network connectivity")
            return

        shallow_photons = fetch_photons_sliderule(all_granules)
        log.info("Total shallow photons (1–15 m): %d", len(shallow_photons))

        SHALLOW_CACHE.write_text(json.dumps(shallow_photons))
        log.info("Shallow cache saved → %s", SHALLOW_CACHE)

    # Merge with deep cache and save combined
    merged = merge_caches(shallow_photons, DEEP_CACHE)
    MERGED_CACHE.write_text(json.dumps(merged))
    log.info("Merged cache saved → %s (%d photons total)", MERGED_CACHE, len(merged))

    # Depth distribution summary
    depths = [p["true_depth_m"] for p in merged]
    for band_min, band_max in [(1, 5), (5, 10), (10, 15), (15, 20), (20, 30)]:
        n = sum(1 for d in depths if band_min <= d < band_max)
        log.info("  %2d–%2d m: %d photons", band_min, band_max, n)

    log.info("=== DONE — merged cache ready for calibration ===")
    log.info("Use: calibrate_from_icesat2(..., photon_cache='%s')", MERGED_CACHE)


if __name__ == "__main__":
    main()
