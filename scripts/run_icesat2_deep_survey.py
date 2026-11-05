#!/usr/bin/env python3
"""
ICESat-2 ATL03 deep-zone reef survey — Algarve coast 15–30 m.

Uses SlideRule to extract sub-surface photon returns from all ATL03
overpasses (2022-2024) over the Algarve coast, applies a refraction
correction (n_water ≈ 1.336), filters to the 15–30 m depth band,
clusters into candidate reef patches, and writes validated GeoJSON
for the dashboard /api/candidates?layer=icesat2_deep route.

Usage:
    python scripts/run_icesat2_deep_survey.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Full Algarve extent: Cabo de Santa Maria → Cabo do Carvoeiro
BBOX = (-8.50, 36.90, -7.85, 37.20)   # W, S, E, N

# ICESat-2 green laser refractive index in seawater (532 nm)
N_SEAWATER = 1.336

TRUE_DEPTH_MIN_M = 15.0
TRUE_DEPTH_MAX_M = 30.0

# Apparent depth search window (add margin for refraction uncertainty)
APPARENT_DEPTH_MIN_M = (TRUE_DEPTH_MIN_M / N_SEAWATER) - 2.0   # ~9.2 m
APPARENT_DEPTH_MAX_M = (TRUE_DEPTH_MAX_M / N_SEAWATER) + 3.0   # ~25.4 m

OUT_DIR = _PROJECT_ROOT / "outputs" / "icesat2_deep_survey"

TIME_START = "2022-01-01T00:00:00Z"
TIME_END   = "2024-12-31T00:00:00Z"


def build_polygon():
    W, S, E, N = BBOX
    return [
        {"lon": W, "lat": S},
        {"lon": E, "lat": S},
        {"lon": E, "lat": N},
        {"lon": W, "lat": N},
        {"lon": W, "lat": S},
    ]


def _extract_seafloor_photons(gdf) -> list[dict]:
    """
    From one batch GeoDataFrame: apply refraction correction and return
    only photons in the 15–30 m true-depth band as plain dicts.
    """

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


def fetch_and_filter_photons(granules: list[str]) -> list[dict]:
    """
    Query SlideRule ATL03 batch-by-batch and immediately filter to
    seafloor photons in the 15–30 m true-depth band.  Never accumulates
    the full multi-million-photon DataFrame in memory.
    """
    from sliderule import icesat2, sliderule as sr
    sr.init("slideruleearth.io", verbose=False)

    parm = {
        "poly": build_polygon(),
        "srt": icesat2.SRT_OCEAN,
        "cnf": icesat2.CNF_BACKGROUND,
        "pass_invalid": True,
    }

    log.info("Querying %d ATL03 granules via SlideRule (batch)...", len(granules))
    batch_size = 10

    all_photons: list[dict] = []
    for i in range(0, len(granules), batch_size):
        batch = granules[i: i + batch_size]
        log.info("  Batch %d/%d (%d granules)...",
                 i // batch_size + 1,
                 (len(granules) + batch_size - 1) // batch_size,
                 len(batch))
        try:
            gdf = icesat2.atl03sp(parm, resources=batch)
            if gdf is not None and len(gdf) > 0:
                batch_photons = _extract_seafloor_photons(gdf)
                log.info("    → %d total photons, %d in depth band",
                         len(gdf), len(batch_photons))
                all_photons.extend(batch_photons)
        except Exception as exc:
            log.warning("  Batch error: %s", exc)

    log.info("Total seafloor photons (15–30 m): %d", len(all_photons))
    return all_photons


def cluster_to_candidates(photons: list[dict], cluster_deg: float = 0.015) -> list[dict]:
    """
    Grid-cell clustering at 0.015° (~1.4 × 1.1 km at 37°N).
    Cells with ≥3 photons become candidates.
    """
    from collections import defaultdict
    cells: dict[tuple, list] = defaultdict(list)

    for p in photons:
        key = (
            round(p["lon"] / cluster_deg) * cluster_deg,
            round(p["lat"] / cluster_deg) * cluster_deg,
        )
        cells[key].append(p["true_depth_m"])

    candidates = []
    for (clon, clat), depths in cells.items():
        if len(depths) < 3:
            continue
        candidates.append({
            "lon": clon,
            "lat": clat,
            "photon_count": len(depths),
            "depth_mean_m": float(np.mean(depths)),
            "depth_std_m":  float(np.std(depths)),
            "depth_min_m":  float(np.min(depths)),
            "depth_max_m":  float(np.max(depths)),
        })

    log.info("Clustered %d photons → %d candidates (≥3/cell)", len(photons), len(candidates))
    return candidates


def score_candidates(candidates: list[dict]) -> list[dict]:
    """Confidence score based on photon density and depth consistency."""
    for c in candidates:
        score = 40
        notes = []

        n = c["photon_count"]
        if n >= 15:   score += 30; notes.append(f"Dense returns ({n} photons)")
        elif n >= 7:  score += 20; notes.append(f"Good returns ({n} photons)")
        elif n >= 3:  score += 10; notes.append(f"Sparse returns ({n} photons)")

        std = c["depth_std_m"]
        if std < 2.0:  score += 20; notes.append(f"Consistent depth (±{std:.1f}m)")
        elif std < 4:  score += 10; notes.append(f"Moderate variation (±{std:.1f}m)")
        else:          score -= 10; notes.append(f"High variation (±{std:.1f}m)")

        dm = c["depth_mean_m"]
        notes.append(f"Mean depth {dm:.1f}m (refraction-corrected)")

        score = max(0, min(100, score))
        if score >= 70:   cls = "HIGH CONFIDENCE — Likely real reef"
        elif score >= 50: cls = "MODERATE — Needs field verification"
        elif score >= 30: cls = "LOW — Possible noise/artifact"
        else:             cls = "REJECT — Likely false positive"

        c["confidence_score"] = score
        c["validation_class"] = cls
        c["validation_notes"] = " | ".join(notes)
        c["source"]           = "ICESat-2 ATL03 (SlideRule)"
        c["depth_zone"]       = "15–30 m"

    return candidates


def to_geojson(candidates: list[dict], out_path: Path) -> None:
    features = []
    for i, c in enumerate(candidates):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
            "properties": {
                "id":               i,
                "confidence_score": c["confidence_score"],
                "validation_class": c["validation_class"],
                "validation_notes": c["validation_notes"],
                "photon_count":     c["photon_count"],
                "depth_mean_m":     round(c["depth_mean_m"], 1),
                "depth_std_m":      round(c["depth_std_m"], 1),
                "source":           c["source"],
                "depth_zone":       c["depth_zone"],
            },
        })

    fc = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(fc, indent=2))
    log.info("Saved %d features → %s", len(features), out_path)


def main():
    import sliderule.earthdata as ed
    from sliderule import sliderule as sr
    sr.init("slideruleearth.io", verbose=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "atl03_seafloor_photons.json"

    if cache_path.exists():
        log.info("Loading cached photons from %s", cache_path)
        photons = json.loads(cache_path.read_text())
    else:
        log.info("=== Discovering ATL03 granules (bbox: %.2f,%.2f → %.2f,%.2f) ===", *BBOX)
        granules = ed.cmr(
            short_name="ATL03",
            polygon=build_polygon(),
            time_start=TIME_START,
            time_end=TIME_END,
        )
        log.info("Found %d ATL03 granules", len(granules))

        if not granules:
            log.error("No ATL03 granules found — check network connectivity")
            to_geojson([], OUT_DIR / "candidates_icesat2_deep.geojson")
            return

        photons = fetch_and_filter_photons(granules)
        cache_path.write_text(json.dumps(photons))
        log.info("Cached %d refraction-corrected photons → %s", len(photons), cache_path)

    if not photons:
        log.warning("No photons in 15–30 m depth band after refraction correction")
        to_geojson([], OUT_DIR / "candidates_icesat2_deep.geojson")
        return

    candidates = cluster_to_candidates(photons)
    if not candidates:
        log.warning("No clusters with ≥3 photons")
        to_geojson([], OUT_DIR / "candidates_icesat2_deep.geojson")
        return

    candidates = score_candidates(candidates)

    high     = sum(1 for c in candidates if c["confidence_score"] >= 70)
    moderate = sum(1 for c in candidates if 50 <= c["confidence_score"] < 70)
    low      = sum(1 for c in candidates if c["confidence_score"] < 50)

    log.info("=== RESULTS ===")
    log.info("Total candidates : %d", len(candidates))
    log.info("HIGH  (≥70)      : %d", high)
    log.info("MODERATE (50–69) : %d", moderate)
    log.info("LOW/REJECT (<50) : %d", low)

    out_path = OUT_DIR / "candidates_icesat2_deep.geojson"
    to_geojson(candidates, out_path)
    log.info("=== DONE — %s ===", out_path)


if __name__ == "__main__":
    main()
