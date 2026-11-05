#!/usr/bin/env python3
"""
Full Algarve reef candidate survey — Cabo de Santa Maria → Cabo do Carvoeiro.

Pipeline:
  1. Download EMODnet bathymetry for the full extent (≤15 m depth focus)
  2. Compute TRI + BPI morphological indices
  3. Detect reef candidate polygons (depth 1–15 m, high TRI + positive BPI)
  4. Validate and score each candidate
  5. Save validated GeoJSON + HTML viewer

Extent:
  W = -8.50  (Cabo do Carvoeiro / Lagoa)
  E = -7.85  (Cabo de Santa Maria / Faro)
  S =  36.90 (offshore buffer)
  N =  37.20 (onshore buffer)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from reef_bathy_module import (
    compute_bathy_indices,
    detect_reef_candidates,
    _safe_write_tif,
    _pix_dims,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Full Algarve extent
BBOX = (-8.50, 36.90, -7.85, 37.20)   # W, S, E, N
DEPTH_MIN = -15.0
DEPTH_MAX = -1.0
EMODNET_WCS = "https://ows.emodnet-bathymetry.eu/wcs"
DATE_STR = datetime.now(timezone.utc).strftime("%Y%m%d")


def download_emodnet_bbox(bbox, output_dir: Path, resolution_m: float = 115.0) -> Path | None:
    """Download EMODnet bathymetry directly for a bounding box."""
    import requests

    w, s, e, n = bbox
    px_w, px_h = _pix_dims(bbox, target_m=resolution_m, max_px=1024)
    out_path = output_dir / f"bathy_emodnet_algarve_{DATE_STR}.tif"

    if out_path.exists():
        log.info("EMODnet bathy already cached: %s", out_path)
        return out_path

    params = {
        "SERVICE":  "WCS",
        "VERSION":  "1.0.0",
        "REQUEST":  "GetCoverage",
        "COVERAGE": "emodnet:mean",
        "CRS":      "EPSG:4326",
        "BBOX":     f"{w},{s},{e},{n}",
        "WIDTH":    str(px_w),
        "HEIGHT":   str(px_h),
        "FORMAT":   "image/tiff",
    }

    log.info("Downloading EMODnet bathy: bbox=(%.2f,%.2f)→(%.2f,%.2f)  %dx%d px",
             w, s, e, n, px_w, px_h)
    try:
        resp = requests.get(EMODNET_WCS, params=params, timeout=60)
        resp.raise_for_status()
        if b"ServiceException" in resp.content[:500]:
            log.error("EMODnet WCS error: %s", resp.content[:300])
            return None

        # Write raw bytes as GeoTIFF
        tmp = output_dir / "_emodnet_raw.tif"
        tmp.write_bytes(resp.content)

        # Re-open and write with correct georeferencing
        with rasterio.open(tmp) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()

        transform = from_bounds(w, s, e, n, px_w, px_h)
        profile.update(
            driver="GTiff", dtype="float32", crs="EPSG:4326",
            transform=transform, width=px_w, height=px_h,
            count=1, compress="lzw",
        )
        _safe_write_tif(str(out_path), profile, data[np.newaxis])
        tmp.unlink(missing_ok=True)
        log.info("Saved: %s  (%dx%d px, res≈%.0fm)", out_path.name, px_w, px_h, resolution_m)
        return out_path

    except Exception as exc:
        log.error("EMODnet download failed: %s", exc)
        return None


def validate_candidates(candidates_geojson: Path, bathy_tif: Path,
                        tri_tif: Path, bpi_tif: Path) -> Path:
    """Score each candidate and return path to validated GeoJSON."""
    import geopandas as gpd

    gdf = gpd.read_file(candidates_geojson)
    log.info("Validating %d candidates …", len(gdf))

    with rasterio.open(bathy_tif) as src:
        bathy_arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        if nodata is not None:
            bathy_arr = np.where(bathy_arr == nodata, np.nan, bathy_arr)

    with rasterio.open(tri_tif) as src:
        tri_arr = src.read(1).astype(np.float32)

    with rasterio.open(bpi_tif) as src:
        bpi_arr = src.read(1).astype(np.float32)

    scores, classes, notes_list = [], [], []

    for _, row in gdf.iterrows():
        geom = row.geometry
        area_m2 = float(row.get("area_m2", 0)) or (geom.area * (111_320 ** 2))

        mask = rasterize([(geom, 1)], out_shape=bathy_arr.shape,
                         transform=transform, fill=0, dtype=np.uint8)
        if mask.sum() == 0:
            scores.append(0); classes.append("ERROR"); notes_list.append("No raster overlap")
            continue

        b_vals = bathy_arr[mask == 1]; b_vals = b_vals[np.isfinite(b_vals)]
        t_vals = tri_arr[mask == 1];   t_vals = t_vals[np.isfinite(t_vals)]
        p_vals = bpi_arr[mask == 1];   p_vals = p_vals[np.isfinite(p_vals)]

        if len(b_vals) == 0:
            scores.append(0); classes.append("ERROR"); notes_list.append("No valid bathy")
            continue

        depth_std  = float(np.std(b_vals))
        tri_mean   = float(np.mean(t_vals)) if len(t_vals) else 0.0
        bpi_mean   = float(np.mean(p_vals)) if len(p_vals) else 0.0
        pixels     = int(mask.sum())

        score = 50
        notes = []

        if area_m2 < 10_000:
            score -= 20; notes.append("Small area (<1 ha) — may be noise")
        elif area_m2 > 50_000:
            score += 10; notes.append("Large area — consistent with reef structure")

        if depth_std > 5:
            score -= 15; notes.append(f"High depth variation ({depth_std:.1f}m)")
        else:
            score += 10; notes.append("Consistent depth — good")

        if tri_mean > 2.0:
            score += 15; notes.append(f"High rugosity (TRI={tri_mean:.1f})")
        elif tri_mean > 1.0:
            score += 5;  notes.append(f"Moderate rugosity (TRI={tri_mean:.1f})")
        else:
            score -= 10; notes.append(f"Low rugosity (TRI={tri_mean:.1f}) — may be sediment")

        if bpi_mean > 2.0:
            score += 15; notes.append(f"Positive BPI ({bpi_mean:.1f}) — elevated structure")
        elif bpi_mean > 0.5:
            score += 5;  notes.append(f"Weakly positive BPI ({bpi_mean:.1f})")
        else:
            score -= 5;  notes.append(f"Near-zero BPI ({bpi_mean:.1f})")

        expected_px = area_m2 / (115 * 115)
        if pixels < expected_px * 0.5:
            score -= 10; notes.append("Fragmented shape — possible artifact")

        score = max(0, min(100, score))
        if score >= 70:
            cls = "HIGH CONFIDENCE — Likely real reef"
        elif score >= 50:
            cls = "MODERATE — Needs verification"
        elif score >= 30:
            cls = "LOW — Probably noise/artifact"
        else:
            cls = "REJECT — Likely false positive"

        scores.append(score)
        classes.append(cls)
        notes_list.append(" | ".join(notes))

    gdf["confidence_score"]  = scores
    gdf["validation_class"]  = classes
    gdf["validation_notes"]  = notes_list

    out_path = candidates_geojson.parent / "candidates_validated.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    log.info("Validated GeoJSON: %s", out_path)
    return out_path


def build_html_viewer(validated_geojson: Path, bathy_tif: Path, output_dir: Path) -> Path:
    """Build a Leaflet HTML viewer for the validated candidates."""
    import geopandas as gpd

    gdf = gpd.read_file(validated_geojson)
    gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf

    high      = gdf[gdf["confidence_score"] >= 70]
    moderate  = gdf[(gdf["confidence_score"] >= 50) & (gdf["confidence_score"] < 70)]
    low       = gdf[gdf["confidence_score"] < 50]

    geojson_str = gdf_wgs.to_json()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Algarve Reef Survey — Cabo de Santa Maria → Cabo do Carvoeiro</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, sans-serif; background: #0a0f18; }}
  #map {{ width: 100vw; height: 100vh; }}
  .info-panel {{
    position: absolute; top: 16px; left: 16px; z-index: 1000;
    background: rgba(15,23,42,0.92); color: #e2e8f0;
    padding: 16px 20px; border-radius: 10px; min-width: 240px;
    border: 1px solid #334155; font-size: 13px; line-height: 1.8;
  }}
  .info-panel h2 {{ font-size: 15px; color: #fff; margin-bottom: 8px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  .dot-green  {{ background:#22c55e; }}
  .dot-yellow {{ background:#f59e0b; }}
  .dot-red    {{ background:#ef4444; }}
  .leaflet-popup-content-wrapper {{ border-radius: 10px; }}
</style>
</head>
<body>
<div class="info-panel">
  <h2>Algarve Reef Survey 2025-09-25</h2>
  <div><span class="dot dot-green"></span>High confidence: <b>{len(high)}</b></div>
  <div><span class="dot dot-yellow"></span>Moderate: <b>{len(moderate)}</b></div>
  <div><span class="dot dot-red"></span>Low / Reject: <b>{len(low)}</b></div>
  <hr style="margin:10px 0;border-color:#334155">
  <div>Depth range: 1–15 m</div>
  <div>Source: EMODnet + TRI/BPI</div>
  <div>Extent: Santa Maria → Carvoeiro</div>
</div>
<div id="map"></div>
<script>
const map = L.map('map').setView([37.05, -8.17], 11);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  subdomains: 'abcd', maxZoom: 19
}}).addTo(map);

L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Esri World Imagery', opacity: 0.5
}}).addTo(map);

const candidates = {geojson_str};

function colorFor(score) {{
  if (score >= 70) return '#22c55e';
  if (score >= 50) return '#f59e0b';
  return '#ef4444';
}}

L.geoJSON(candidates, {{
  style: f => ({{
    color: colorFor(f.properties.confidence_score || 0),
    weight: 2, opacity: 0.9,
    fillOpacity: 0.25,
    fillColor: colorFor(f.properties.confidence_score || 0),
  }}),
  onEachFeature: (f, layer) => {{
    const p = f.properties;
    const score = p.confidence_score || 0;
    const area  = p.area_m2 ? (p.area_m2 / 10000).toFixed(1) + ' ha' : 'n/a';
    layer.bindPopup(`
      <b style="color:${{colorFor(score)}}">${{p.validation_class || 'Unknown'}}</b><br>
      <b>Score:</b> ${{score}}/100<br>
      <b>Area:</b> ${{area}}<br>
      <small style="color:#6b7280">${{p.validation_notes || ''}}</small>
    `);
  }}
}}).addTo(map);
</script>
</body>
</html>"""

    out = output_dir / "algarve_reef_survey.html"
    out.write_text(html)
    log.info("HTML viewer: %s", out)
    return out


def main():
    out_dir = _PROJECT_ROOT / "outputs" / "algarve_reef_survey"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download bathymetry
    log.info("=== STEP 1: EMODnet Bathymetry ===")
    bathy_path = download_emodnet_bbox(BBOX, out_dir)
    if bathy_path is None:
        log.error("Bathymetry download failed — aborting")
        sys.exit(1)

    # 2. Compute TRI + BPI indices
    log.info("=== STEP 2: TRI + BPI Indices ===")
    indices = compute_bathy_indices(str(bathy_path), str(out_dir))
    if not indices.get("tri") or not indices.get("bpi_broad"):
        log.error("Index computation failed — aborting")
        sys.exit(1)
    log.info("TRI: %s", indices["tri"])
    log.info("BPI broad: %s", indices["bpi_broad"])

    # 3. Detect reef candidates (1–15 m, high TRI + positive BPI)
    log.info("=== STEP 3: Reef Candidate Detection ===")
    candidates_path = detect_reef_candidates(
        bathy_tif=str(bathy_path),
        indices_dir=str(out_dir),
        output_dir=str(out_dir),
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        tri_pct=65.0,
        bpi_pct=55.0,
        min_area_px=3,
        date_str=DATE_STR,
    )
    if candidates_path is None:
        log.error("No candidates detected — check depth range or bathymetry coverage")
        sys.exit(1)
    log.info("Candidates GeoJSON: %s", candidates_path)

    # 4. Validate candidates
    log.info("=== STEP 4: Validation & Scoring ===")
    validated_path = validate_candidates(
        candidates_geojson=Path(candidates_path),
        bathy_tif=bathy_path,
        tri_tif=Path(indices["tri"]),
        bpi_tif=Path(indices["bpi_broad"]),
    )

    # 5. Summary
    try:
        import geopandas as gpd
        gdf = gpd.read_file(validated_path)
        high     = (gdf["confidence_score"] >= 70).sum()
        moderate = ((gdf["confidence_score"] >= 50) & (gdf["confidence_score"] < 70)).sum()
        low      = (gdf["confidence_score"] < 50).sum()
        log.info("=== RESULTS ===")
        log.info("Total candidates : %d", len(gdf))
        log.info("HIGH (≥70)       : %d", high)
        log.info("MODERATE (50-69) : %d", moderate)
        log.info("LOW/REJECT (<50) : %d", low)

        # 6. HTML viewer
        log.info("=== STEP 5: HTML Viewer ===")
        viewer = build_html_viewer(validated_path, bathy_path, out_dir)
        import subprocess
        subprocess.Popen(["open", "-a", "Safari", str(viewer)])
        log.info("Opened in Safari: %s", viewer)
    except Exception as exc:
        log.warning("Could not build viewer: %s", exc)

    log.info("=== DONE — outputs in %s ===", out_dir)


if __name__ == "__main__":
    main()
