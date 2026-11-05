#!/usr/bin/env python3
"""
fetch_planet_highres.py
=======================
Search Planet high-resolution imagery (SkySat 0.5m, PlanetScope 3-5m)
for a given location and date range.

Usage:
    python fetch_planet_highres.py --lat 37.069081 --lon -8.210242 \
 --date 2025-09-20/2025-09-30 --output-dir outputs/planet_highres

Requires PLANET_API_KEY environment variable (free trial at planet.com/explorer)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer


def search_planet_scenes(lat, lon, date_start, date_end, api_key):
    """Search Planet Data API v1 for matching scenes."""
    url = "https://api.planet.com/data/v1/quick-search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"api-key {api_key}",
    }
    payload = {
        "item_types": ["PSScene"],
        "filter": {
            "type": "AndFilter",
            "config": [
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {"gte": f"{date_start}T00:00:00Z", "lte": f"{date_end}T23:59:59Z"},
                },
                {
                    "type": "GeometryFilter",
                    "field_name": "geometry",
                    "config": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                },
            ],
        },
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code == 401:
        print(" ❌ Planet API key inválida ou expirou.")
        print("  → Obtém uma chave em https://planet.com/explorer")
        return []
    if resp.status_code != 200:
        print(f" ❌ HTTP {resp.status_code}: {resp.text[:200]}")
        return []

    data = resp.json()
    features = data.get("features", [])
    return features


def download_and_prepare_band(asset_href, lat, lon, buffer_m, api_key, band_name):
    """Download a local window of a Planet band via HTTP."""
    headers = {"Authorization": f"api-key {api_key}"}
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

    with env:
        with rasterio.open(asset_href, headers=headers) as src:
            tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = tf.transform(lon, lat)
            window = from_bounds(
                x - buffer_m, y - buffer_m,
                x + buffer_m, y + buffer_m,
                src.transform,
            )
            arr = src.read(1, window=window).astype(np.float32)

    return arr


def make_pure_band_image(arr, band_type):
    """Create a pure single-band PNG (R=0, G=0, B=band for blue, etc.)."""
    norm = arr / 10000.0
    norm = np.clip(norm, 0, 1.0)

    h, w = norm.shape
    if band_type == "blue":
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 2] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    elif band_type == "green":
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 1] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    elif band_type == "red":
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    else:
        rgb = np.stack([norm, norm, norm], axis=2) * 255

    import cv2
    success, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        return None
    return buf.tobytes()


def process_and_download(scene, lat, lon, buffer_m, api_key, output_dir):
    """Download all bands for a Planet scene and save as pure-band PNGs."""
    scene_id = scene["id"]
    date_str = scene["properties"].get("acquired", "")[:10]
    instrument = scene["properties"].get("instrument", scene["properties"].get("instruments", "?"))
    n_bands = len(scene.get("assets", {}))

    print(f"\n  📡 {scene_id}")
    print(f"     Date: {date_str} | Instrument: {instrument} | Bands: {n_bands}")

    site_dir = Path(output_dir) / f"{scene_id}_{date_str}"
    site_dir.mkdir(parents=True, exist_ok=True)

    # Save scene metadata
    with open(site_dir / "scene_meta.json", "w") as f:
        json.dump({"id": scene_id, "date": date_str, "properties": scene["properties"]}, f, indent=2, default=str)

    # Determine which bands are available
    assets = scene.get("assets", {})
    band_map = {}
    for key in ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]:
        if key in assets:
            band_map[key] = assets[key]["href"]

    if not band_map:
        print("     ⚠️  Sem bandas disponíveis")
        return None

    results = {}
    for band_key, href in band_map.items():
        try:
            arr = download_and_prepare_band(href, lat, lon, buffer_m, api_key, band_key)
            if arr is None or arr.max() == 0:
                continue

            # Band type mapping for PlanetScope (analytic bands)
            band_type_map = {
                "b1": "blue",   # Coastal Blue / Pan
                "b2": "blue",   # Blue
                "b3": "green",  # Green
                "b4": "red",    # Red
                "b5": "nir",    # NIR
                "b6": "nir2",   # Red Edge
                "b7": "nir",    # NIR
                "b8": "pan",    # Pan
            }
            btype = band_type_map.get(band_key, "gray")

            png_bytes = make_pure_band_image(arr, btype)
            if png_bytes:
                out_path = site_dir / f"planet_{band_key}_{date_str}.png"
                with open(out_path, "wb") as f:
                    f.write(png_bytes)
                results[band_key] = str(out_path)
                print(f"     ✅ {band_key} → {out_path.name}")

        except Exception as e:
            print(f"     ❌ {band_key}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Planet high-res imagery for reef sites")
    parser.add_argument("--lat", type=float, default=37.069081)
    parser.add_argument("--lon", type=float, default=-8.210242)
    parser.add_argument("--date", default="2025-09-20/2025-09-30")
    parser.add_argument("--buffer", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/planet_highres")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("PLANET_API_KEY", "")
    if not api_key:
        print("❌ PLANET_API_KEY não definida.")
        print("   export PLANET_API_KEY='your_key_here'")
        print("   Obtém uma chave gratuita em: https://planet.com/explorer")
        sys.exit(1)

    lat, lon = args.lat, args.lon
    date_start, date_end = args.date.split("/")
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PLANET HIGH-RES SEARCH")
    print(f"  {lat:.6f}N, {abs(lon):.6f}W | {date_start} → {date_end}")
    print(f"  Buffer: ±{args.buffer}m | Top {args.top_n} scenes")
    print("=" * 60)

    print("\n🔍 A procurar cenas Planet...")
    scenes = search_planet_scenes(lat, lon, date_start, date_end, api_key)
    print(f"   Encontradas {len(scenes)} cenas")

    if not scenes:
        print("\n⚠️  Nenhuma cena encontrada.")
        print("   Alternativas:")
        print("1. Usa https://planet.com/explorer — browsable sem API key")
        print("   2. Verifica se a API key é válida em planet.com/account")
        sys.exit(1)

    # Sort by cloud cover
    scenes.sort(key=lambda s: s["properties"].get("cloud_cover", 99))
    scenes = scenes[: args.top_n]

    print(f"\n📥 A descarregar top {len(scenes)} cenas...")
    for scene in scenes:
        process_and_download(scene, lat, lon, args.buffer, api_key, output_dir)

    print(f"\n✅ Done! Guardado em: {output_dir}/")


if __name__ == "__main__":
    main()
