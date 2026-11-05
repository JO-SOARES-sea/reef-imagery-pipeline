#!/usr/bin/env python3
"""
Download and process the 2025-09-25 Sentinel-2 scene for 5 shallow Algarve sites.
Clips B02/B03 COG bands around each site and runs run_predictor() for BVI + SDB.
"""

import logging
import sys
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
import rasterio.crs

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.stac_ingest import search_sentinel2_scenes, get_asset_hrefs
from src.reef_ml_predictor_acolite import run_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCENE_DATE = ("2025-09-25", "2025-09-26")
BUFFER_DEG = 0.027   # ~3 km half-box at Algarve latitude
DEPTH_TARGET = 15.0

SITES = [
    ("quarteira",       37.065, -8.100),
    ("vilamoura",       37.075, -8.130),
    ("olhos_de_agua",   37.090, -8.160),
    ("gale",            37.056, -8.230),
    ("armacao_de_pera", 37.070, -8.360),
]


def clip_cog(href: str, lat: float, lon: float, out_path: Path) -> None:
    """Read a COG window around (lat, lon) and write a local GeoTIFF."""
    bbox_wgs84 = (
        lon - BUFFER_DEG, lat - BUFFER_DEG,
        lon + BUFFER_DEG, lat + BUFFER_DEG,
    )
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="TRUE", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(href) as src:
            west, south, east, north = rasterio.warp.transform_bounds(
                "EPSG:4326", src.crs,
                *bbox_wgs84,
            )
            window = from_bounds(west, south, east, north, src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                width=data.shape[1],
                height=data.shape[0],
                transform=transform,
                compress="lzw",
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    log.info("  Wrote %s  (%d×%d px)", out_path.name, data.shape[1], data.shape[0])


def main():
    log.info("Searching for 2025-09-25 scene …")
    scenes = search_sentinel2_scenes(
        lat=SITES[0][1], lon=SITES[0][2],
        date_range=SCENE_DATE,
        max_cloud_cover=10,
    )
    if not scenes:
        log.error("No scene found — aborting")
        sys.exit(1)

    scene = scenes[0]
    log.info("Scene: %s  cloud=%.1f%%", scene.id, scene.properties.get("eo:cloud_cover", 0))

    hrefs = get_asset_hrefs(scene, ["B02", "B03"])
    b02_href = hrefs["B02"]
    b03_href = hrefs["B03"]

    results = []

    for site_name, lat, lon in SITES:
        log.info("=== %s (%.4f, %.4f) ===", site_name, lat, lon)
        out_dir = PROJECT_DIR / "outputs" / "algarve_20250925" / site_name
        b02_path = out_dir / "S2_B02_20250925.tif"
        b03_path = out_dir / "S2_B03_20250925.tif"

        log.info("  Downloading B02 …")
        clip_cog(b02_href, lat, lon, b02_path)
        log.info("  Downloading B03 …")
        clip_cog(b03_href, lat, lon, b03_path)

        log.info("  Running predictor …")
        try:
            metadata = {"date": "2025-09-25", "cloud_cover": 1.2, "scene_id": scene.id}
            result = run_predictor(
                boa_b02_path=str(b02_path),
                b03_path=str(b03_path),
                output_dir=str(out_dir),
                metadata=metadata,
                lat=lat,
                lon=lon,
                depth_target=DEPTH_TARGET,
            )
            bvi   = result.get("bvi_score", result.get("score", "n/a"))
            snr   = result.get("snr", "n/a")
            sdb   = result.get("sdb_mean_depth", result.get("depth_mean", "n/a"))
            kd    = result.get("kd490", "n/a")
            log.info("  RESULT  BVI=%.4f  SNR=%.1f  SDB=%.1fm  Kd=%.4f",
                     bvi if isinstance(bvi, float) else 0,
                     snr if isinstance(snr, float) else 0,
                     sdb if isinstance(sdb, float) else 0,
                     kd  if isinstance(kd,  float) else 0)
            results.append({"site": site_name, "lat": lat, "lon": lon,
                            "bvi": bvi, "snr": snr, "sdb_mean_m": sdb, "kd490": kd})
        except Exception as exc:
            log.error("  predictor failed: %s", exc)
            results.append({"site": site_name, "lat": lat, "lon": lon, "error": str(exc)})

    # Summary table
    print("\n" + "="*70)
    print(f"{'Site':<22} {'BVI':>7} {'SNR':>8} {'SDB mean':>10} {'Kd490':>8}")
    print("-"*70)
    for r in results:
        if "error" in r:
            print(f"{r['site']:<22}  ERROR: {r['error'][:40]}")
        else:
            bvi = r['bvi']
            snr = r['snr']
            sdb = r['sdb_mean_m']
            kd  = r['kd490']
            print(f"{r['site']:<22} {bvi:>7.4f} {snr:>8.1f} {sdb:>9.1f}m {kd:>8.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
