#!/usr/bin/env python3
"""
build_reef_dataset.py — End-to-end pipeline: EMODnet → labels → S2 patches.

Steps
-----
1. Fetch EMODnet bathymetry for the Algarve study area.
2. Compute TRI (Terrain Ruggedness Index) from the depth raster.
3. Generate reef/sand candidate labels:
     - Reef: GPS dive-confirmed sites (Pedra Sta. Eulália, Pedra do Alto).
     - Sand: EMODnet cells with TRI < threshold AND depth in [3, 25 m].
4. Sample sand candidates at random to match reef count (up to sand_multiplier × n_reef).
5. Extract 128×128 Sentinel-2 patches (B02/B03/B04/B08) for each label.
6. Write outputs/reef_habitat/labels/label_manifest.json and patch .npy files.

Usage
-----
    python scripts/build_reef_dataset.py [--scene-id S2B_29SNB_20240930_0_L2A]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import xy as rio_xy

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.emodnet_bathymetry import fetch_depth, compute_tri, build_habitat_mask
from src.s2_patch_extractor import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Study area and ground truth ────────────────────────────────────────────────

ALGARVE_BBOX = (-8.6, 36.85, -7.7, 37.30)   # (lon_min, lat_min, lon_max, lat_max)

# Reef GPS sources:
#  1. Dive-confirmed GPS (human audit 2026-06-22): Pedra Sta Eulália + Pedra do Alto
#  2. Fishing GPS (CarlosBarrinha.gpx) — western Algarve spots with ≥4 repeated
#     visits at optical depth (5–22 m). Repeated visits to the same coordinates
#     on a fishfinder strongly indicate rocky substrate / reef bottom.
REEF_SITES = [
    # ── Dive-confirmed ─────────────────────────────────────────────────────────
    {"name": "Pedra_Sta_Eulalia",    "lon": -8.2102,  "lat": 37.0691 },
    {"name": "Pedra_do_Alto",        "lon": -8.20982, "lat": 37.05815},
    # ── Fishing GPS (CarlosBarrinha.gpx, ≥4 visits, 5–22 m depth) ─────────────
    {"name": "fishing_N_eulalia_8m", "lon": -8.20294, "lat": 37.07504},  # 6 visits, 8 m — shallow top of reef N of Pedra Sta Eulália
    {"name": "fishing_gale_16m",     "lon": -8.22965, "lat": 37.05822},  # 6 visits, 16 m — offshore Galé
    {"name": "fishing_gale_15m",     "lon": -8.23384, "lat": 37.06220},  # 5 visits, 15 m — coastal Galé
    {"name": "fishing_eulalia_11m",  "lon": -8.21477, "lat": 37.06629},  # 4 visits, 11 m — between confirmed sites
    {"name": "fishing_midshelf_19m", "lon": -8.17101, "lat": 37.04593},  # 4 visits, 19 m — mid shelf
]

# Hard-coded inshore sand areas confirmed by bathymetric chart —
# these are flat, sandy shelf at 5–15 m depth, 2+ km from reef sites.
KNOWN_SAND_LOCATIONS = [
    {"name": "sand_praia_faro",    "lon": -7.9800, "lat": 36.9700},
    {"name": "sand_armacao_pera",  "lon": -8.3600, "lat": 37.0600},
    {"name": "sand_sagres_shelf",  "lon": -8.9500, "lat": 37.0000},
]

# Paths
OUTPUT_BASE   = Path("outputs/reef_habitat")
EMODNET_DIR   = OUTPUT_BASE / "emodnet"
LABELS_DIR    = OUTPUT_BASE / "labels"
PATCHES_DIR   = OUTPUT_BASE / "patches"
DEPTH_PATH    = EMODNET_DIR / "depth_algarve.tif"
TRI_PATH      = EMODNET_DIR / "tri_algarve.tif"
LABEL_PATH    = LABELS_DIR  / "label_manifest.json"

# Classification parameters (matching emodnet_bathymetry.py defaults)
MAX_DEPTH_M         = 25.0
TRI_REEF_THRESHOLD  = 0.30
SAND_MULTIPLIER     = 5     # max sand candidates = SAND_MULTIPLIER × n_reef


def step1_fetch_emodnet() -> None:
    if DEPTH_PATH.exists():
        log.info("Depth raster already exists: %s (skipping download)", DEPTH_PATH)
        return
    log.info("Step 1: Fetching EMODnet bathymetry for bbox=%s", ALGARVE_BBOX)
    fetch_depth(ALGARVE_BBOX, DEPTH_PATH)


def step2_compute_tri() -> None:
    if TRI_PATH.exists():
        log.info("TRI raster already exists: %s (skipping)", TRI_PATH)
        return
    log.info("Step 2: Computing TRI")
    compute_tri(DEPTH_PATH, TRI_PATH)


def step3_generate_labels(seed: int = 42) -> None:
    if LABEL_PATH.exists():
        log.info("Label manifest already exists: %s (skipping)", LABEL_PATH)
        return

    log.info("Step 3: Generating labels")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reef labels: GPS-confirmed dive sites ──────────────────────────────
    reef_labels = []
    for site in REEF_SITES:
        reef_labels.append({
            "lon":    site["lon"],
            "lat":    site["lat"],
            "class":  "reef",
            "source": "gps_dive_confirmed",
            "name":   site["name"],
        })
    log.info("Reef labels: %d GPS-confirmed sites", len(reef_labels))

    # ── Sand labels: known sand locations + EMODnet low-TRI candidates ────
    sand_labels = []

    # Hard-coded known sand locations
    for loc in KNOWN_SAND_LOCATIONS:
        sand_labels.append({
            "lon":    loc["lon"],
            "lat":    loc["lat"],
            "class":  "sand",
            "source": "bathymetric_chart_confirmed",
            "name":   loc["name"],
        })

    # EMODnet-derived sand candidates: low TRI + in depth window
    habitat = build_habitat_mask(
        DEPTH_PATH, TRI_PATH,
        max_depth_m=MAX_DEPTH_M,
        tri_reef_threshold=TRI_REEF_THRESHOLD,
    )
    mask = habitat["mask"]
    transform = habitat["transform"]

    # Extract pixel coordinates of sand candidates (class==1)
    sand_rows, sand_cols = np.where(mask == 1)
    log.info("EMODnet sand candidate pixels: %d", len(sand_rows))

    # Sample at most SAND_MULTIPLIER × n_reef candidates to keep dataset manageable
    n_reef = len(reef_labels)
    target_sand = SAND_MULTIPLIER * n_reef
    rng = random.Random(seed)

    if len(sand_rows) > target_sand:
        idx = rng.sample(range(len(sand_rows)), target_sand)
    else:
        idx = list(range(len(sand_rows)))

    for i in idx:
        lon, lat = rio_xy(transform, sand_rows[i] + 0.5, sand_cols[i] + 0.5)
        sand_labels.append({
            "lon":    float(lon),
            "lat":    float(lat),
            "class":  "sand",
            "source": "emodnet_low_tri",
            "name":   f"emodnet_sand_{i:05d}",
        })

    log.info(
        "Total sand labels: %d (%d from chart + %d from EMODnet)",
        len(sand_labels),
        len(KNOWN_SAND_LOCATIONS),
        len(sand_labels) - len(KNOWN_SAND_LOCATIONS),
    )

    all_labels = reef_labels + sand_labels
    manifest = {
        "labels":             all_labels,
        "n_reef":             len(reef_labels),
        "n_sand":             len(sand_labels),
        "bbox":               list(ALGARVE_BBOX),
        "depth_path":         str(DEPTH_PATH),
        "tri_path":           str(TRI_PATH),
        "tri_reef_threshold": TRI_REEF_THRESHOLD,
        "max_depth_m":        MAX_DEPTH_M,
    }
    with open(LABEL_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Label manifest written: %s  (total=%d)", LABEL_PATH, len(all_labels))


def step4_extract_patches(scene_id: str | None, date_range: str) -> dict:
    log.info("Step 4: Extracting S2 patches (scene=%s)", scene_id or "auto")
    return build_dataset(
        labels_path=LABEL_PATH,
        output_dir=PATCHES_DIR,
        scene_id=scene_id,
        date_range=date_range,
        patch_px=128,
        val_fraction=0.2,
        max_cloud=5.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reef classification dataset")
    parser.add_argument(
        "--scene-id",
        default="S2B_29SNB_20240930_0_L2A",
        help="Sentinel-2 scene ID (default: best clear scene 2024-09-30)",
    )
    parser.add_argument(
        "--date-range",
        default="2024-06-01/2024-12-31",
        help="STAC date range for auto scene selection",
    )
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if outputs exist")
    args = parser.parse_args()

    if args.force:
        for p in [DEPTH_PATH, TRI_PATH, LABEL_PATH]:
            if p.exists():
                p.unlink()
                log.info("Removed existing file: %s", p)

    step1_fetch_emodnet()
    step2_compute_tri()
    step3_generate_labels()
    summary = step4_extract_patches(args.scene_id, args.date_range)

    print("\n── Dataset build complete ──────────────────────────────────────")
    print(f"  Reef  : train={summary.get('n_reef_train',0)}  val={summary.get('n_reef_val',0)}")
    print(f"  Sand  : train={summary.get('n_sand_train',0)}  val={summary.get('n_sand_val',0)}")
    print(f"  Failed: {summary.get('n_failed', 0)}")
    print(f"  Scene : {summary.get('scene_id','?')}  ({summary.get('scene_date','?')})")
    print(f"  Output: {PATCHES_DIR}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
