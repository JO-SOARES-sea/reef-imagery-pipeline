"""
download_multiband_s2.py
Download ALL Sentinel-2 L2A bands useful for reef/bathymetry detection
============================================================
Uses Microsoft Planetary Computer STAC (zero-auth) to:
  - Search for S2 L2A scene at a given point/date
  - Download 11 bands (B01-B12, B8A) clipped to AOI bbox
  - Read the 10 m bands on their NATIVE grid (integer window, exact affine transform)
  - Resample 20m/60m bands onto that exact 10 m reference grid (bilinear for reflectance DN,
    nearest for categorical layers such as SCL; nodata excluded from interpolation)
  - Save each band as GeoTIFF (stored L2A DN — NO reflectance conversion) + metadata JSON
    recording product provenance and the per-band radiometry decision
    (see src/s2_radiometry.py; Planetary Computer assets carry no scale/offset, so they are
    radiometry_unknown unless the product's MTD_MSIL2A.xml BOA_ADD_OFFSET can be read)
  - Refuse to overwrite existing files (use --overwrite explicitly)

Usage:
  python download_multiband_s2.py --lat 37.068978 --lon -8.210328 --date 2024-10-15
  python download_multiband_s2.py --output-dir my_output --buffer-m 2000

Importable:
  from download_multiband_s2 import download_multiband
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import transform as window_transform
from pystac_client import Client
import planetary_computer as pc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.s2_grid import bbox_window, unsigned_href  # noqa: E402
from src.s2_radiometry import (  # noqa: E402
    radiometry_from_product_metadata,
    radiometry_from_stac_asset,
)

# ESA physical band index used in MTD_MSIL2A.xml BOA_ADD_OFFSET band_id
ESA_BAND_ID = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
               "B08": 7, "B8A": 8, "B09": 9, "B10": 10, "B11": 11, "B12": 12}
# Layers that must never be interpolated as continuous variables
CATEGORICAL_LAYERS = {"SCL"}

log = logging.getLogger("download_multiband_s2")

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

REEF_BANDS = {
    "B01": {"wl": "443nm", "desc": "Coastal Aerosol",    "res_m": 60},
    "B02": {"wl": "490nm", "desc": "Blue",               "res_m": 10},
    "B03": {"wl": "560nm", "desc": "Green",              "res_m": 10},
    "B04": {"wl": "665nm", "desc": "Red",                "res_m": 10},
    "B05": {"wl": "705nm", "desc": "Red Edge 1",         "res_m": 20},
    "B06": {"wl": "740nm", "desc": "Red Edge 2",         "res_m": 20},
    "B07": {"wl": "783nm", "desc": "Red Edge 3",         "res_m": 20},
    "B08": {"wl": "842nm", "desc": "NIR",                "res_m": 10},
    "B8A": {"wl": "865nm", "desc": "Narrow NIR",         "res_m": 20},
    "B11": {"wl": "1610nm", "desc": "SWIR1",             "res_m": 20},
    "B12": {"wl": "2190nm", "desc": "SWIR2",             "res_m": 20},
}

DEFAULT_LAT = 37.068978
DEFAULT_LON = -8.210328
DEFAULT_DATE = "2024-10-15"
DEFAULT_BUFFER_M = 1000
DEFAULT_OUTPUT_DIR = "reef_multiband_output"


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "download_multiband.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log.info("download_multiband_s2 started — log: %s", log_path)


def deg_offset(buffer_m: float) -> float:
    return buffer_m / 111_320.0


def read_native_window(href: str, bbox_wgs84: tuple) -> tuple:
    """Read *href* over *bbox_wgs84* on its native grid. Returns (data, transform, crs, nodata)."""
    with rasterio.open(href) as src:
        window = bbox_window(src, bbox_wgs84)
        data = src.read(1, window=window)
        return data, window_transform(window, src.transform), src.crs, src.nodata


def resampling_for(band_name: str):
    """Nearest for categorical layers; bilinear for continuous reflectance DN."""
    return Resampling.nearest if band_name in CATEGORICAL_LAYERS else Resampling.bilinear


def resample_to_grid(data: np.ndarray, src_transform, src_crs, src_nodata,
                     dst_transform, dst_crs, dst_shape: tuple, resampling) -> np.ndarray:
    """
    Reproject *data* onto an EXACT reference grid. Source nodata is excluded from the
    interpolation kernel; destination cells with no valid source stay nodata.
    """
    nodata = 0 if src_nodata is None else src_nodata
    dst = np.full(dst_shape, nodata, dtype=np.float32)
    reproject(
        source=data.astype(np.float32), destination=dst,
        src_transform=src_transform, src_crs=src_crs, src_nodata=nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=nodata,
        resampling=resampling,
    )
    return dst


def product_radiometry(item, band_name: str, asset) -> dict:
    """Per-band radiometry decision from asset metadata, else from MTD_MSIL2A.xml if readable."""
    rad = radiometry_from_stac_asset(asset.to_dict(), item.properties)
    if rad.is_known:
        return rad.as_dict()
    mtd = item.assets.get("product-metadata")
    if mtd is not None and band_name in ESA_BAND_ID:
        try:
            import requests
            r = requests.get(mtd.href, timeout=60)
            r.raise_for_status()
            return radiometry_from_product_metadata(r.text, ESA_BAND_ID[band_name]).as_dict()
        except Exception as exc:  # network/parse failure → stays unknown
            d = rad.as_dict()
            d["evidence"].append(f"MTD_MSIL2A.xml unreadable: {exc}")
            return d
    return rad.as_dict()


def download_multiband(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON,
                       date: str = DEFAULT_DATE, buffer_m: float = DEFAULT_BUFFER_M,
                       output_dir: str = DEFAULT_OUTPUT_DIR, overwrite: bool = False) -> dict:
    """Download all reef-relevant S2 L2A bands to output_dir.

    Every band is written on ONE reference grid: the native 10 m grid of B02 over the bbox.
    Values are stored L2A DN (nodata=0); no reflectance conversion is applied here.

    Returns dict with keys: scene_id, bands (dict of band->filepath), meta_path, radiometry.
    """
    os.makedirs(output_dir, exist_ok=True)

    d = deg_offset(buffer_m)
    bbox_wgs84 = (lon - d, lat - d, lon + d, lat + d)

    log.info("Searching Planetary Computer STAC for sentinel-2-l2a ...")
    log.info("  Point: (%.5f, %.5f), Date: %s, Buffer: %d m", lat, lon, date, buffer_m)

    catalog = Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
    time_range = f"{date}T00:00:00Z/{date}T23:59:59Z"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=time_range,
    )
    items = list(search.items())
    if not items:
        log.error("No S2 L2A scene found for %s at (%.5f, %.5f)", date, lat, lon)
        return {}
    if len(items) > 1:
        log.warning("  %d items match; using %s — record kept in metadata", len(items), items[0].id)

    item = items[0]
    scene_id = item.id
    log.info("Found scene: %s", scene_id)

    date_str = date.replace("-", "")
    band_files = {}
    band_info = {}
    radiometry = {}

    # Reference grid = native 10 m grid of B02 over the bbox
    ref_asset = item.assets.get("B02")
    if ref_asset is None:
        log.error("Reference band B02 missing — cannot define the 10 m grid; aborting")
        return {}
    _ref, ref_transform, ref_crs, _ = read_native_window(ref_asset.href, bbox_wgs84)
    ref_shape = _ref.shape

    for band_name, band_meta in REEF_BANDS.items():
        asset = item.assets.get(band_name)
        if not asset:
            log.warning("  Asset %s not found in scene — not downloaded, not substituted", band_name)
            band_info[band_name] = {"status": "missing_asset"}
            continue

        out_path = os.path.join(output_dir, f"S2_{band_name}_{date_str}.tif")
        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"{out_path} exists — refusing to overwrite (pass overwrite=True)")
        native_res = band_meta["res_m"]
        radiometry[band_name] = product_radiometry(item, band_name, asset)

        log.info("  Downloading %s (%s, %s, %d m) ...",
                 band_name, band_meta["desc"], band_meta["wl"], native_res)

        try:
            raw, raw_transform, raw_crs, raw_nodata = read_native_window(asset.href, bbox_wgs84)
            same_grid = (raw_crs == ref_crs and raw_transform.almost_equals(ref_transform)
                         and raw.shape == ref_shape)
            if same_grid:
                data, resampled, method = raw, False, None
            else:
                method = resampling_for(band_name)
                data = resample_to_grid(raw, raw_transform, raw_crs, raw_nodata,
                                        ref_transform, ref_crs, ref_shape, method)
                resampled = True

            profile = {
                "driver": "GTiff", "compress": "lzw", "count": 1,
                "height": ref_shape[0], "width": ref_shape[1],
                "crs": ref_crs, "transform": ref_transform,
                "dtype": str(data.dtype), "nodata": 0 if raw_nodata is None else raw_nodata,
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
                dst.update_tags(1, values="L2A stored DN (no reflectance conversion applied)",
                                radiometry_status=radiometry[band_name]["status"],
                                source_href=unsigned_href(asset.href))

            file_size = os.path.getsize(out_path)
            band_files[band_name] = out_path
            band_info[band_name] = {
                "status": "ok",
                "description": band_meta["desc"],
                "wavelength": band_meta["wl"],
                "native_resolution_m": native_res,
                "resampled_to_10m": resampled,
                "resampling": None if method is None else method.name,
                "values": "stored L2A DN, nodata=0, NOT reflectance",
                "radiometry": radiometry[band_name],
                "source_href": unsigned_href(asset.href),
                "file": out_path,
                "file_size_bytes": file_size,
                "shape": list(data.shape),
                "valid_pixels": int((data != profile["nodata"]).sum()),
            }
            log.info("    ✓ %s saved → %s (%.1f KB, %dx%d) radiometry=%s",
                     band_name, out_path, file_size / 1024, data.shape[1], data.shape[0],
                     radiometry[band_name]["status"])

        except FileExistsError:
            raise
        except Exception as exc:
            log.error("    ✗ %s failed: %s", band_name, exc)
            band_info[band_name] = {"status": "download_failed", "error": str(exc)}

    meta = {
        "scene_id": scene_id,
        "date": date,
        "source": "planetary_computer_stac",
        "collection": "sentinel-2-l2a",
        "n_items_matched": len(items),
        "point": {"lat": lat, "lon": lon},
        "buffer_m": buffer_m,
        "aoi_bounds": {
            "west": bbox_wgs84[0],
            "south": bbox_wgs84[1],
            "east": bbox_wgs84[2],
            "north": bbox_wgs84[3],
        },
        "target_resolution_m": 10,
        "grid": {"crs": str(ref_crs), "transform": list(ref_transform)[:6], "shape": list(ref_shape),
                 "reference_band": "B02 native 10 m"},
        "bands": band_info,
        "radiometry_note": ("Files hold stored DN. Convert ONCE with the per-band 'radiometry' record; "
                            "status radiometry_unknown => blocked from scientific use."),
        "properties": {
            "acquired": item.properties.get("datetime"),
            "platform": item.properties.get("platform"),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "s2_datatake_id": item.properties.get("s2:datatake_id"),
            "s2_mgrs_tile": item.properties.get("s2:mgrs_tile"),
            "s2_product_uri": item.properties.get("s2:product_uri"),
            "s2_processing_baseline": item.properties.get("s2:processing_baseline"),
            "s2_granule_id": item.properties.get("s2:granule_id"),
            "s2_generation_time": item.properties.get("s2:generation_time"),
            "s2_product_type": item.properties.get("s2:product_type"),
        },
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = os.path.join(output_dir, f"S2_multiband_meta_{date_str}.json")
    if os.path.exists(meta_path) and not overwrite:
        raise FileExistsError(f"{meta_path} exists — refusing to overwrite (pass overwrite=True)")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata → %s", meta_path)

    ok_bands = {k: v for k, v in band_info.items() if v.get("status") == "ok"}
    log.info("=" * 60)
    log.info("Download summary:")
    log.info("  Scene: %s", scene_id)
    log.info("  Bands downloaded: %d / %d", len(band_files), len(REEF_BANDS))
    total_size = 0
    for bn, bi in ok_bands.items():
        sz_kb = bi["file_size_bytes"] / 1024
        total_size += bi["file_size_bytes"]
        resampled_tag = f" [resampled→10m, {bi['resampling']}]" if bi["resampled_to_10m"] else ""
        log.info("    %s (%s) — %.1f KB — %dx%d%s — radiometry=%s",
                 bn, bi["description"], sz_kb,
                 bi["shape"][1], bi["shape"][0], resampled_tag, bi["radiometry"]["status"])
    log.info("  Total size: %.1f KB (%.2f MB)", total_size / 1024, total_size / 1e6)
    log.info("  Output dir: %s", output_dir)

    return {
        "scene_id": scene_id,
        "bands": band_files,
        "meta_path": meta_path,
        "total_size_bytes": total_size,
        "radiometry": radiometry,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download all Sentinel-2 L2A reef/bathymetry bands via Planetary Computer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT,
                        help="Latitude of center point")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON,
                        help="Longitude of center point")
    parser.add_argument("--date", default=DEFAULT_DATE,
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M,
                        help="Clip radius in meters")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for GeoTIFFs")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing files in output-dir (off by default)")

    args = parser.parse_args()

    setup_logging(args.output_dir)

    result = download_multiband(
        lat=args.lat,
        lon=args.lon,
        date=args.date,
        buffer_m=args.buffer_m,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    if not result:
        log.error("Download failed — no results.")
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
