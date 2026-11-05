#!/usr/bin/env python3
"""
algarve_reef_survey.py
======================
Orchestrate a complete reef survey of the central Algarve coast
from Tavira to Armação de Pêra using Sentinel-2 multispectral imagery.

Uses Planetary Computer STAC to find low-cloud scenes, downloads all 11
bands via download_multiband_s2, and runs multiband reef analysis on each
site.

Outputs:
  - algarve_survey_results.csv   (one row per site)
  - algarve_survey_map.png       (overview map)
  - algarve_survey_summary.json  (full structured results)

CLI:
  python algarve_reef_survey.py [--output-dir outputs/algarve_survey] \\
      [--date-range 2024-01-01/2024-12-31] [--parallel 3]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


sys.path.insert(0, os.path.dirname(__file__))
from download_multiband_s2 import download_multiband
from multiband_reef_analysis import run_analysis

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from pystac_client import Client
    import planetary_computer as pc
    HAS_STAC = True
except ImportError:
    HAS_STAC = False

log = logging.getLogger("algarve_reef_survey")

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

SURVEY_SITES = [
    ("tavira_west",          37.1050, -7.6800, 12),
    ("ilha_de_tavira",       37.0950, -7.7200, 10),
    ("fuseta",               37.0600, -7.7600, 14),
    ("olhao_offshore",       37.0350, -7.8200, 12),
    ("faro_east",            37.0100, -7.8800, 15),
    ("praia_de_faro",        36.9800, -7.9400, 12),
    ("ancão_peninsula",      36.9650, -8.0000, 10),
    ("quarteira",            37.0650, -8.1000, 12),
    ("vilamoura",            37.0750, -8.1300, 14),
    ("olhos_de_agua",        37.0900, -8.1600, 12),
    ("pedra_sta_eulalia",    37.069081, -8.210242, 12),
    ("albufeira_reef",       37.0690, -8.2105, 16),
    ("galé",                 37.0560, -8.2296, 14),
    ("salgados",             37.0950, -8.3000, 12),
    ("armacao_de_pera",      37.0700, -8.3600, 10),
]


def find_best_scene(lat, lon, date_start, date_end):
    """Search Planetary Computer for the lowest-cloud S2 L2A scene."""
    if not HAS_STAC:
        log.error("pystac_client / planetary_computer not installed")
        return None, None

    catalog = Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        limit=5,
    )
    items = list(search.items())
    if not items:
        return None, None

    best = items[0]
    scene_id = best.id
    cloud = best.properties.get("eo:cloud_cover", -1)
    date_str = best.properties.get("datetime", "")[:10]
    return scene_id, {"date": date_str, "cloud_cover": cloud}


def process_site(name, lat, lon, expected_depth, date_start, date_end,
                 output_root):
    """Download + analyse a single survey site. Returns a result dict."""
    site_dir = os.path.join(output_root, name)
    dl_dir = os.path.join(site_dir, "bands")
    an_dir = os.path.join(site_dir, "analysis")
    t0 = time.time()

    result = {
        "site": name,
        "lat": lat,
        "lon": lon,
        "expected_depth_m": expected_depth,
        "status": "pending",
    }

    try:
        log.info("[%s] Finding best scene ...", name)
        scene_id, scene_info = find_best_scene(lat, lon, date_start, date_end)
        if scene_id is None:
            result["status"] = "no_scene"
            result["error"] = "No S2 L2A scene found"
            log.warning("[%s] No scene found", name)
            return result

        result["scene_id"] = scene_id
        result["scene_date"] = scene_info["date"]
        result["cloud_cover"] = scene_info["cloud_cover"]
        log.info("[%s] Best scene: %s (cloud=%.1f%%, date=%s)",
                 name, scene_id, scene_info["cloud_cover"], scene_info["date"])

        log.info("[%s] Downloading bands ...", name)
        dl_result = download_multiband(
            lat=lat, lon=lon, date=scene_info["date"],
            buffer_m=1000, output_dir=dl_dir,
        )
        if not dl_result:
            result["status"] = "download_failed"
            result["error"] = "download_multiband returned empty"
            log.error("[%s] Download failed", name)
            return result

        result["bands_downloaded"] = len(dl_result.get("bands", {}))

        log.info("[%s] Running reef analysis ...", name)
        summary = run_analysis(
            input_dir=dl_dir, output_dir=an_dir,
            lat=lat, lon=lon, depth_min=-20.0, depth_max=-4.0,
        )

        result["best_method"] = summary.get("best_method", "unknown")
        result["reef_candidate_pixels"] = summary.get("reef_candidate_pixels", 0)

        reef_geojson = summary.get("reef_candidates_geojson")
        n_polys = 0
        total_area = 0.0
        if reef_geojson and os.path.exists(reef_geojson):
            with open(reef_geojson) as f:
                fc = json.load(f)
            features = fc.get("features", [])
            n_polys = len(features)
            for feat in features:
                props = feat.get("properties", {})
                total_area += props.get("area_m2", 0.0)

        result["reef_polygons"] = n_polys
        result["reef_area_m2"] = round(total_area, 1)

        method_key_map = {
            "stumpf": "stumpf_2band",
            "lyzenga": "lyzenga_6band",
            "pca": "pca_based",
        }
        best_name = summary.get("best_method", "")
        best_stats = summary.get("methods", {}).get(
            method_key_map.get(best_name, ""), {})
        if not best_stats or best_stats.get("mean") is None:
            for key in ("stumpf_2band", "lyzenga_6band", "pca_based"):
                s = summary.get("methods", {}).get(key)
                if s and s.get("mean") is not None:
                    best_stats = s
                    break
        result["mean_depth"] = best_stats.get("mean")
        result["status"] = "success"

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        log.error("[%s] Failed: %s", name, exc, exc_info=True)

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def classify_reef(result):
    """Classify a site as green/yellow/red based on reef detection."""
    if result.get("status") != "success":
        return "red"
    polys = result.get("reef_polygons", 0)
    area = result.get("reef_area_m2", 0.0)
    if polys >= 3 and area >= 500:
        return "green"
    if polys >= 1 and area >= 100:
        return "yellow"
    return "red"


def generate_map(results, output_path):
    """Generate overview map of all survey sites and reef detections."""
    if not HAS_MPL:
        log.warning("matplotlib not available — skipping map")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    coast_lons = [-7.70, -7.80, -7.90, -8.00, -8.10, -8.20, -8.30, -8.40]
    coast_lats = [37.00, 36.95, 36.92, 36.93, 36.95, 37.00, 37.02, 37.03]
    ax.fill_between(coast_lons, [36.85] * len(coast_lons), coast_lats,
                    color="#e8dcc8", alpha=0.5, zorder=0)
    ax.plot(coast_lons, coast_lats, color="#8b7355", linewidth=1.5, zorder=1)

    color_map = {"green": "#2ecc71", "yellow": "#f1c40f", "red": "#e74c3c"}
    labels_map = {"green": "Reef found", "yellow": "Marginal", "red": "No reef"}

    for r in results:
        cat = classify_reef(r)
        area = r.get("reef_area_m2", 0.0)
        size = max(30, min(300, area / 20.0))
        ax.scatter(r["lon"], r["lat"], s=size, c=color_map[cat],
                   edgecolors="black", linewidths=0.5, zorder=3, alpha=0.85)
        ax.annotate(r["site"], (r["lon"], r["lat"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=6.5, color="#333", zorder=4)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(color_map.values(), labels_map.values())]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              framealpha=0.9, title="Classification")

    ax.set_xlim(-8.45, -7.60)
    ax.set_ylim(36.88, 37.18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Algarve Reef Survey — Central Coast (Tavira → Armação de Pêra)",
                 fontsize=13, fontweight="bold")
    ax.set_aspect(1.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Survey map → %s", output_path)
    return output_path


def write_csv(results, output_path):
    """Write results to CSV."""
    if not results:
        return
    fieldnames = [
        "site", "lat", "lon", "expected_depth_m", "status",
        "scene_id", "scene_date", "cloud_cover",
        "bands_downloaded", "best_method",
        "reef_candidate_pixels", "reef_polygons", "reef_area_m2",
        "mean_depth", "elapsed_s", "error",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    log.info("Results CSV → %s", output_path)


def write_json(results, output_path):
    """Write full structured JSON summary."""
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "n_sites": len(results),
        "n_success": sum(1 for r in results if r.get("status") == "success"),
        "n_failed": sum(1 for r in results if r.get("status") != "success"),
        "total_reef_area_m2": sum(r.get("reef_area_m2", 0) for r in results),
        "sites": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Summary JSON → %s", output_path)


def run_survey(output_dir, date_start, date_end, max_workers):
    """Run the full Algarve reef survey."""
    os.makedirs(output_dir, exist_ok=True)
    log.info("=" * 70)
    log.info("ALGARVE REEF SURVEY — %d sites", len(SURVEY_SITES))
    log.info("Date range: %s to %s", date_start, date_end)
    log.info("Output: %s", output_dir)
    log.info("Parallel workers: %d", max_workers)
    log.info("=" * 70)

    results = [None] * len(SURVEY_SITES)

    def _task(idx):
        name, lat, lon, depth = SURVEY_SITES[idx]
        return idx, process_site(name, lat, lon, depth,
                                 date_start, date_end, output_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, i): i for i in range(len(SURVEY_SITES))}
        completed = 0
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1

            if completed % 5 == 0 or completed == len(SURVEY_SITES):
                n_ok = sum(1 for r in results if r and r.get("status") == "success")
                n_fail = sum(1 for r in results if r and r.get("status") != "success")
                log.info("─── Progress: %d/%d complete (%d ok, %d failed) ───",
                         completed, len(SURVEY_SITES), n_ok, n_fail)

            partial_csv = os.path.join(output_dir, "algarve_survey_results_partial.csv")
            write_csv([r for r in results if r is not None], partial_csv)

    results = [r for r in results if r is not None]

    write_csv(results, os.path.join(output_dir, "algarve_survey_results.csv"))
    write_json(results, os.path.join(output_dir, "algarve_survey_summary.json"))
    generate_map(results, os.path.join(output_dir, "algarve_survey_map.png"))

    n_ok = sum(1 for r in results if r.get("status") == "success")
    total_area = sum(r.get("reef_area_m2", 0) for r in results)
    log.info("=" * 70)
    log.info("SURVEY COMPLETE: %d/%d sites processed, %.0f m² total reef area",
             n_ok, len(results), total_area)
    log.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Algarve reef survey — central coast Tavira to Armação de Pêra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="outputs/algarve_survey",
                        help="Root output directory")
    parser.add_argument("--date-range", default="2024-01-01/2024-12-31",
                        help="Date range as YYYY-MM-DD/YYYY-MM-DD")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Number of parallel workers")
    args = parser.parse_args()

    date_start, date_end = args.date_range.split("/")

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "algarve_survey.log")
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    run_survey(args.output_dir, date_start, date_end, args.parallel)


if __name__ == "__main__":
    main()
