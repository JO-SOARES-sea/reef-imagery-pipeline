#!/usr/bin/env python3
"""
Explore detailed bathymetric context around Pedra Sta Eulália reef.
Uses cached IH isobath data + BathyFeatureEngine + live DGRM API query.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
REEF_LON = -8.210328
REEF_LAT = 37.068978
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "ih_isobaths_-8.295_36.990_-8.022_37.108_10,20,30,50,100.json"
TARGET_ISOBATHS = [10, 20, 30, 50, 100]

# ── Helpers ───────────────────────────────────────────────────────────────────
M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    p1, l1 = math.radians(lat1), math.radians(lon1)
    p2, l2 = math.radians(lat2), math.radians(lon2)
    dp, dl = p2 - p1, l2 - l1
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def min_dist_to_line(lon, lat, coords):
    """Min haversine distance from point to any vertex in coords list."""
    best = float("inf")
    for c in coords:
        d = haversine_m(lon, lat, c[0], c[1])
        if d < best:
            best = d
    return best


# ── 1. Load cached isobath data ──────────────────────────────────────────────
print("=" * 72)
print("BATHYMETRIC CONTEXT: Pedra Sta Eulália  (37.068978, -8.210328)")
print("=" * 72)

if not CACHE_FILE.exists():
    print(f"ERROR: Cache file not found: {CACHE_FILE}")
    sys.exit(1)

with open(CACHE_FILE) as f:
    raw = json.load(f)

features = raw["features"]
# Convert GeoJSON Feature → internal format
iso_features = []
for feat in features:
    props = feat["properties"]
    coords = feat["geometry"]["coordinates"]
    iso_features.append({
        "depth": float(props["depth"]),
        "coords": coords,
        "length_m": props.get("length_m", 0.0),
    })

depths_found = sorted(set(int(f["depth"]) for f in iso_features))
print(f"\nLoaded {len(iso_features)} isobath polylines from cache")
print(f"Depths present: {depths_found}")

# ── 2. Distance from reef to each isobath ────────────────────────────────────
print("\n" + "-" * 72)
print("DISTANCE TO EACH ISOBATH (nearest vertex, haversine)")
print("-" * 72)
print(f"{'Isobath':>10}  {'Dist (m)':>12}  {'Dist (km)':>10}  {'#Polylines':>10}")
print("-" * 50)

distances = {}
for target in TARGET_ISOBATHS:
    matching = [f for f in iso_features if int(f["depth"]) == target]
    if not matching:
        distances[target] = float("inf")
        print(f"{target:>7}m   {'N/A':>12}  {'N/A':>10}  {0:>10}")
        continue
    best = float("inf")
    for feat in matching:
        d = min_dist_to_line(REEF_LON, REEF_LAT, feat["coords"])
        if d < best:
            best = d
    distances[target] = best
    print(f"{target:>7}m   {best:>12.1f}  {best/1000:>10.3f}  {len(matching):>10}")

# ── 3. Depth gradient analysis ───────────────────────────────────────────────
print("\n" + "-" * 72)
print("DEPTH GRADIENT ANALYSIS")
print("-" * 72)

available = sorted([(d, distances[d]) for d in TARGET_ISOBATHS if distances[d] < float("inf")])
if len(available) >= 2:
    for i in range(len(available) - 1):
        d1, dist1 = available[i]
        d2, dist2 = available[i + 1]
        dist_change = abs(dist2 - dist1)
        depth_change = abs(d2 - d1)
        if dist_change > 0:
            slope = depth_change / dist_change  # m depth per m horizontal
            slope_deg = math.degrees(math.atan(slope))
            print(f"  {d1}m → {d2}m:  Δdepth={depth_change}m over Δhoriz={dist_change:.0f}m"
                  f"  → slope={slope:.4f} ({slope_deg:.2f}°)")
        else:
            print(f"  {d1}m → {d2}m:  overlapping isobaths (steep!)")

    # Overall gradient: shallowest to deepest available
    d_sh, dist_sh = available[0]
    d_dp, dist_dp = available[-1]
    total_horiz = abs(dist_dp - dist_sh)
    total_depth = abs(d_dp - d_sh)
    if total_horiz > 0:
        overall_slope = total_depth / total_horiz
        print(f"\n  Overall ({d_sh}m→{d_dp}m): {total_depth}m depth over {total_horiz:.0f}m horiz"
              f"  → avg slope={overall_slope:.4f} ({math.degrees(math.atan(overall_slope)):.2f}°)")
else:
    print("  Not enough isobaths to compute gradient.")

# Nearest isobath → depth estimate
nearest_depth_val = available[0][0] if available else None
nearest_dist_val = available[0][1] if available else float("inf")
print(f"\n  Nearest isobath: {nearest_depth_val}m at {nearest_dist_val:.1f}m distance")
if nearest_dist_val < 200:
    print(f"  → Reef is very close to the {nearest_depth_val}m contour (likely ~{nearest_depth_val}m depth)")

# ── 4. BathyFeatureEngine: all 11 IH features ───────────────────────────────
print("\n" + "=" * 72)
print("IH BATHYMETRY FEATURES (BathyFeatureEngine)")
print("=" * 72)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ih_bathy_features import BathyFeatureEngine

engine = BathyFeatureEngine(cache_dir=str(CACHE_DIR))
feats = engine.compute_features_for_point(REEF_LON, REEF_LAT, buffer_m=5_000.0)

print(f"\n{'Feature':<35}  {'Value':>15}")
print("-" * 52)
for k, v in feats.items():
    if isinstance(v, float):
        if v == float("inf") or v == np.inf:
            display = "∞ (not found)"
        else:
            display = f"{v:.4f}"
    elif v is None:
        display = "N/A"
    else:
        display = str(v)
    print(f"  {k:<33}  {display:>15}")

# ── 5. Location classification ───────────────────────────────────────────────
print("\n" + "-" * 72)
print("LOCATION CLASSIFICATION")
print("-" * 72)

zone = feats.get("bathymetry_zone_class", "unknown")
slope_proxy = feats.get("bathymetry_slope_proxy", 0.0)
density = feats.get("contour_density_proxy", 0.0)
n_iso = feats.get("n_isobaths_in_aoi", 0)

zone_descriptions = {
    "very_shallow": "Shelf / Very shallow (< 200m from 10m isobath)",
    "shallow_reef": "Shallow reef platform (< 500m from 20m isobath)",
    "nearshore_mid": "Nearshore mid-depth zone",
    "mid_depth": "Mid-depth / upper slope",
    "offshore": "Offshore / outer shelf",
    "unknown": "Unknown (insufficient data)",
}
print(f"  Zone class:       {zone}")
print(f"  Description:      {zone_descriptions.get(zone, 'N/A')}")
print(f"  Slope proxy:      {slope_proxy:.4f} (std of nearby isobath depths)")
print(f"  Contour density:  {density:.2f} m/m²")
print(f"  Isobaths in AOI:  {n_iso}")

# Terrain interpretation
print("\n  Terrain interpretation:")
if slope_proxy > 15:
    print("    → Steep slope / escarpment (high depth variance)")
elif slope_proxy > 8:
    print("    → Moderate slope (transition zone)")
elif slope_proxy > 3:
    print("    → Gentle slope / gradual descent")
else:
    print("    → Flat shelf / plateau (low depth variance)")

if density > 0.01:
    print("    → High contour density (complex bathymetry)")
elif density > 0.005:
    print("    → Moderate contour density")
else:
    print("    → Low contour density (simple/flat bottom)")

# ── 6. Live DGRM API query (tight bbox around reef) ─────────────────────────
print("\n" + "=" * 72)
print("LIVE DGRM API QUERY (±0.01° bbox around reef)")
print("=" * 72)

import requests

_TIGHT_BBOX = (
    REEF_LON - 0.01, REEF_LAT - 0.01,
    REEF_LON + 0.01, REEF_LAT + 0.01,
)
IH_QUERY = "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/Dados_entidades_externas/Batimetrica_IH/MapServer/0/query"

params = {
    "where": "1=1",
    "geometry": f"{_TIGHT_BBOX[0]},{_TIGHT_BBOX[1]},{_TIGHT_BBOX[2]},{_TIGHT_BBOX[3]}",
    "geometryType": "esriGeometryEnvelope",
    "spatialRel": "esriSpatialRelIntersects",
    "outFields": "OBJECTID,Depth,Shape_Leng",
    "returnGeometry": "true",
    "f": "json",
}

print(f"\n  Bbox: [{_TIGHT_BBOX[0]:.4f}, {_TIGHT_BBOX[1]:.4f}, {_TIGHT_BBOX[2]:.4f}, {_TIGHT_BBOX[3]:.4f}]")
print(f"  URL: {IH_QUERY}")

try:
    resp = requests.get(IH_QUERY, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        print(f"  API error: {data['error']}")
    else:
        api_features = data.get("features", [])
        print(f"  Features returned: {len(api_features)}")

        if api_features:
            api_depths = set()
            for feat in api_features:
                d = feat.get("attributes", {}).get("Depth")
                if d is not None:
                    api_depths.add(float(d))

            print(f"  Raw isobath depths found: {sorted(api_depths)}")

            # Per-depth detail
            print(f"\n  {'Depth':>8}  {'#Segments':>10}  {'Min dist (m)':>14}")
            print("  " + "-" * 38)
            for d in sorted(api_depths):
                segments = [f for f in api_features if f["attributes"]["Depth"] == d]
                # Compute min distance
                best = float("inf")
                for seg in segments:
                    paths = seg.get("geometry", {}).get("paths", [])
                    for path in paths:
                        for node in path:
                            dist = haversine_m(REEF_LON, REEF_LAT, node[0], node[1])
                            if dist < best:
                                best = dist
                dist_str = f"{best:.1f}" if best < float("inf") else "N/A"
                print(f"  {d:>7.0f}m  {len(segments):>10}  {dist_str:>14}")

            # Check for depths not in our target list
            extra = sorted(api_depths - set(float(t) for t in TARGET_ISOBATHS))
            if extra:
                print(f"\n  Additional isobaths in area (not in target set): {extra}")
        else:
            print("  No features found in this tight bbox.")
except Exception as e:
    print(f"  Request failed: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  Reef location:        ({REEF_LAT}, {REEF_LON})")
print(f"  Nearest isobath:      {nearest_depth_val}m at {nearest_dist_val:.1f}m")
print(f"  Bathymetric zone:     {zone}")
print(f"  Terrain:              {'Shelf' if slope_proxy < 5 else 'Slope' if slope_proxy < 15 else 'Escarpment'}")
if slope_proxy < 5 and (nearest_depth_val is None or nearest_depth_val <= 20):
    print("  Classification:       Likely shelf/plateau reef with gentle surrounding bathymetry")
elif slope_proxy >= 5:
    print("  Classification:       Reef on or near a bathymetric slope/transition")
else:
    print("  Classification:       Insufficient data for definitive classification")
print("=" * 72)
