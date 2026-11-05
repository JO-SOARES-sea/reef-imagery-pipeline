#!/usr/bin/env python3
"""
download_multiband_pleiades.py — Pléiades Neo Multi-Band Downloader
══════════════════════════════════════════════ Pléiades Neo Multi-Band Downloader
══════════════════════════════════════════════════════════════════════════════════
Downloads Pléiades Neo imagery via multiple sources:
  1. Airbus OneAtlas STAC API (commercial, requires API key)
  2. ESA Third Party Missions (TPM) — free for research
  3. OrtoSat2023 WMS fallback (Portugal DGT, 30cm Pléiades-derived)

For reef identification, only B02 (Blue, 490nm) at 1.2m resolution matters.
This module downloads B02 + supporting bands for atmospheric correction.

Usage:
  python scripts/download_multiband_pleiades.py --lat 37.068978 --lon -8.210328
  python scripts/download_multiband_pleiades.py --source wms --date 2023-06-01

Sources:
  - airbus_stac: Airbus OneAtlas STAC (needs AIRBUS_API_KEY env var)
  - esa_tpm: ESA Third Party Missions portal
  - wms: OrtoSat2023 WMS from DGT Portugal (always available, 3-band RGB+NIR)
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
import argparse
import logging
import numpy as np
from pathlib import Path

import requests
import rasterio
from pyproj import Transformer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


log = logging.getLogger("pleiades_download")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SITE_LAT = 37.068978
SITE_LON = -8.210328
BUFFER_M = 500
DEFAULT_OUTPUT_DIR = "outputs/pleiades_neo"

# OrtoSat2023 WMS (Portugal DGT — always available)
ORTOSAT_WMS = "https://ortos.dgterritorio.gov.pt/wms/ortosat2023"
ORTOSAT_LAYERS = {
    "true_color": "ortoSat2023-CorVerdadeira",
    "false_color": "ortoSat2023-FalsaCor",
    "nir": "ortoSat2023-FalsaCor",  # NIR in false color composite
}

# Airbus OneAtlas STAC
AIRBUS_STAC_URL = "https://api.airbus.com/stac/v1"
AIRBUS_COLLECTION = "pleiades-neo"


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: OrtoSat2023 WMS (always available)
# ═══════════════════════════════════════════════════════════════════════════════

def download_ortosat_wms(lat, lon, buffer_m=500, output_dir=None, layers=None):
    """
    Download OrtoSat2023 imagery via WMS (Portugal DGT).

    This is Pléiades-derived imagery at 30cm resolution.
    Available layers: RGB true color, false color (NIR-G-B).

    Returns: dict with paths to downloaded GeoTIFFs.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = ["true_color", "false_color"]

    # Compute bbox in UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
    x, y = transformer.transform(lon, lat)
    bbox_utm = (x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m)

    # Convert bbox to WGS84 for WMS
    transformer_inv = Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer_inv.transform(bbox_utm[0], bbox_utm[1])
    lon_max, lat_max = transformer_inv.transform(bbox_utm[2], bbox_utm[3])

    # WMS request
    width = int(buffer_m * 2 / 0.3)  # 30cm resolution
    height = width

    results = {}
    for layer_name in layers:
        wms_layer = ORTOSAT_LAYERS.get(layer_name)
        if not wms_layer:
            continue

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": wms_layer,
            "CRS": "EPSG:4326",
            "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "FORMAT": "image/png",
            "TRANSPARENT": "false",
        }

        log.info(f"  Downloading OrtoSat2023 {layer_name} ({wms_layer})...")
        log.info(f"    BBox: {lat_min:.6f},{lon_min:.6f} → {lat_max:.6f},{lon_max:.6f}")
        log.info(f"    Size: {width}x{height} pixels (30cm)")

        try:
            resp = requests.get(ORTOSAT_WMS, params=params, timeout=120)
            resp.raise_for_status()

            out_file = out_path / f"ortosat2023_{layer_name}.tif"
            with open(out_file, "wb") as f:
                f.write(resp.content)

            log.info(f"    Saved: {out_file} ({len(resp.content) / 1024:.0f} KB)")
            results[layer_name] = str(out_file)

        except Exception as e:
            log.warning(f"    Failed to download {layer_name}: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: Airbus OneAtlas STAC (commercial, needs API key)
# ═══════════════════════════════════════════════════════════════════════════════

def search_airbus_stac(lat, lon, date_range=None, max_cloud=20):
    """
    Search Airbus OneAtlas STAC for Pléiades Neo scenes.

    Requires AIRBUS_API_KEY environment variable.

    Returns: list of STAC items (or empty list if unavailable).
    """
    api_key = os.environ.get("AIRBUS_API_KEY", "")
    if not api_key:
        log.info("  AIRBUS_API_KEY not set — skipping Airbus STAC search")
        log.info("  Get API key: https://www.airbus.com/en/space/intelligence")
        return []

    headers = {"Authorization": f"Bearer {api_key}"}

    # Search params
    search_body = {
        "collections": [AIRBUS_COLLECTION],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "limit": 20,
    }
    if date_range:
        search_body["datetime"] = date_range
    if max_cloud is not None:
        search_body["query"] = {"eo:cloud_cover": {"lt": max_cloud}}

    try:
        resp = requests.post(
            f"{AIRBUS_STAC_URL}/search",
            json=search_body,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("features", [])
        log.info(f"  Found {len(items)} Pléiades Neo scenes via Airbus STAC")
        return items
    except Exception as e:
        log.warning(f"  Airbus STAC search failed: {e}")
        return []


def download_airbus_pleiades(lat, lon, date=None, buffer_m=500, output_dir=None):
    """
    Download Pléiades Neo bands via Airbus OneAtlas.

    Returns: dict with paths to downloaded bands.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Search for scenes
    date_range = f"{date}/{date}" if date else None
    items = search_airbus_stac(lat, lon, date_range)
    if not items:
        return {}

    # Pick best scene (lowest cloud cover)
    best = min(items, key=lambda i: i["properties"].get("eo:cloud_cover", 100))
    log.info(f"  Best scene: {best['id']} (cloud: {best['properties'].get('eo:cloud_cover', '?')}%)")

    # Download bands
    results = {}
    api_key = os.environ.get("AIRBUS_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"}

    for band_key in ["B02", "B03", "B04", "B08", "PAN"]:
        if band_key not in best.get("assets", {}):
            continue

        asset = best["assets"][band_key]
        href = asset.get("href", "")
        if not href:
            continue

        log.info(f"  Downloading {band_key}...")
        try:
            resp = requests.get(href, headers=headers, timeout=120)
            resp.raise_for_status()

            out_file = out_path / f"pleiades_neo_{band_key}.tif"
            with open(out_file, "wb") as f:
                f.write(resp.content)

            log.info(f"    Saved: {out_file}")
            results[band_key] = str(out_file)

        except Exception as e:
            log.warning(f"    Failed to download {band_key}: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: ESA Third Party Missions (free for research)
# ═══════════════════════════════════════════════════════════════════════════════

def search_esa_tpm(lat, lon, date_range=None, max_cloud=20):
    """
    Search ESA Third Party Missions for Pléiades Neo.

    Uses the Copernicus Data Space Ecosystem (CDSE) which hosts some
    Pléiades Neo data under the ESA TPM programme.

    Returns: list of available scenes.
    """
    # CDSE STAC endpoint
    cdse_stac = "https://catalogue.dataspace.copernicus.eu/stac"

    search_body = {
        "collections": ["PLEIADES_NEO"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "limit": 20,
    }
    if date_range:
        search_body["datetime"] = date_range

    try:
        resp = requests.post(
            f"{cdse_stac}/search",
            json=search_body,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("features", [])
            log.info(f"  Found {len(items)} Pléiades Neo scenes via ESA TPM/CDSE")
            return items
        else:
            log.info(f"  ESA TPM search returned {resp.status_code} — may not be available")
            return []
    except Exception as e:
        log.info(f"  ESA TPM search not available: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def download_pleiades(lat, lon, date=None, buffer_m=500, output_dir=None,
                      source="auto"):
    """
    Download Pléiades Neo imagery from the best available source.

    Source priority:
      1. Airbus STAC (if AIRBUS_API_KEY available)
      2. ESA TPM/CDSE (free for research)
      3. OrtoSat2023 WMS (always available, 30cm, RGB+NIR)

    Args:
        lat, lon: Center coordinates (WGS84)
        date: Optional date string (YYYY-MM-DD)
        buffer_m: Buffer in meters around center
        output_dir: Output directory
        source: "auto", "airbus_stac", "esa_tpm", or "wms"

    Returns: dict with source info and paths to downloaded bands.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    log.info("=" * 70)
    log.info("  Pléiades Neo Download — Multi-Source")
    log.info(f"  GPS: {lat:.6f}°N, {abs(lon):.6f}°W | Buffer: ±{buffer_m}m")
    log.info(f"  Source: {source}")
    log.info("=" * 70)

    result = {
        "sensor": "pleiades-neo",
        "lat": lat, "lon": lon,
        "buffer_m": buffer_m,
        "bands": {},
        "source": None,
    }

    # Try source 1: Airbus STAC
    if source in ("auto", "airbus_stac"):
        log.info("\n[1] Trying Airbus OneAtlas STAC...")
        airbus = download_airbus_pleiades(lat, lon, date, buffer_m, output_dir)
        if airbus:
            result["bands"] = airbus
            result["source"] = "airbus_stac"
            return result

    # Try source 2: ESA TPM
    if source in ("auto", "esa_tpm"):
        log.info("\n[2] Trying ESA Third Party Missions...")
        tpm = search_esa_tpm(lat, lon)
        if tpm:
            log.info(f"  Found {len(tpm)} scenes — use CDSE to download")
            result["source"] = "esa_tpm"
            result["scenes"] = tpm
            # Full download would need CDSE token auth

    # Try source 3: OrtoSat2023 WMS (always available)
    if source in ("auto", "wms"):
        log.info("\n[3] Downloading OrtoSat2023 WMS (Pléiades-derived, 30cm)...")
        wms = download_ortosat_wms(lat, lon, buffer_m, output_dir)
        if wms:
            result["bands"] = wms
            result["source"] = "wms_ortosat2023"
            result["resolution_m"] = 0.30
            result["note"] = "RGB+NIR only (no raw multispectral). Derived from Pléiades Neo."
            return result

    log.warning("  No Pléiades Neo data available from any source")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# B02 EXTRACTION FROM WMS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_b02_from_wms(tif_path, output_path=None):
    """
    Extract B02-equivalent (Blue band) from OrtoSat2023 false color composite.

    OrtoSat2023_FalsaCor = NIR(842) | Green(560) | Blue(490)
    So band 3 (index 2) is the Blue (B02-equivalent) band.

    For reef identification: this is the only band that matters.
    """
    if os.path.getsize(tif_path) < 10000:
        log.warning(f"  File too small ({os.path.getsize(tif_path)} bytes) — likely error response")
        return None

    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            # False color: band 3 = Blue
            b02 = src.read(3).astype(np.float32)
        elif src.count == 1:
            b02 = src.read(1).astype(np.float32)
        else:
            log.warning(f"  Unexpected band count: {src.count}")
            return None

        # Normalize to reflectance (0-1)
        if b02.max() > 1.5:
            b02 = b02 / 255.0  # uint8 → float

        if output_path:
            profile = src.profile.copy()
            profile.update(count=1, dtype="float32", driver="GTiff")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(b02, 1)
            log.info(f"  Extracted B02: {output_path}")

        return b02


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pléiades Neo downloader")
    parser.add_argument("--lat", type=float, default=SITE_LAT)
    parser.add_argument("--lon", type=float, default=SITE_LON)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--buffer", type=int, default=BUFFER_M)
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "airbus_stac", "esa_tpm", "wms"])
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = download_pleiades(
        lat=args.lat, lon=args.lon, date=args.date,
        buffer_m=args.buffer, output_dir=args.output, source=args.source,
    )

    if result.get("bands"):
        print(f"\n  Downloaded {len(result['bands'])} band(s) from {result['source']}")
        for name, path in result["bands"].items():
            print(f"    {name}: {path}")

        # Extract B02 if from WMS
        if result["source"] == "wms_ortosat2023" and "false_color" in result["bands"]:
            b02_path = Path(args.output) / "pleiades_neo_B02.tif"
            extract_b02_from_wms(result["bands"]["false_color"], str(b02_path))
    else:
        print("\n  No data downloaded")
