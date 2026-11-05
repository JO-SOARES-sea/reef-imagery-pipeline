"""
reef_bathy_module.py
====================
Bathymetric Reef Discovery Module — Albufeira Reef (Algarve, Portugal)
Coordinates: lat=37.069071, lon=-8.210492, buffer ~500 m, depth 0–50 m.

Data sources (all free / open):
  1. EMODnet Bathymetry  — WCS 1.0.0, ~115 m, no auth
  2. GEBCO 2024          — sub-grid ZIP download, 15 arc-sec (~460 m), no auth
  3. GEOMAR / IHM        — Portuguese Hydrographic Institute WCS, ~100 m, no auth
  4. NOAA ETOPO          — THREDDS WCS, 60 arc-sec (~1.8 km), no auth
  5. Sentinel-2 SDB      — Stumpf log-ratio shallow depth inversion from B02/B03

Processing:
  compute_bathy_indices  — slope, TRI/roughness, BPI (fine+broad), curvature
  detect_reef_candidates — depth-mask + rugosity + positive BPI → polygons

Export:
  export_candidates_geojson — GeoJSON output
  add_bathy_to_qgis         — inject layers into existing .qgs XML project

CLI:
  run_bathy_step(args)  — called by reef_imagery_pipeline_v3 for --step bathy
  standalone:  python reef_bathy_module.py --step bathy [options]

Usage:
  python reef_bathy_module.py --step bathy \\
      --bathy-source all \\
      --depth-min -50 --depth-max -1 \\
      --output-dir reef_Output_Master/reef_output_v3
"""

import argparse
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import from_bounds
from scipy.ndimage import (
    uniform_filter,
    label as ndimage_label,
)

try:
    from shapely.geometry import shape, mapping, MultiPolygon, Polygon
    from shapely.ops import unary_union
    import geopandas as gpd
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    from pyproj import Transformer as ProjTransformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_write_tif(out_path: str, profile: dict, arr) -> None:
    """Write a numpy array to a GeoTIFF safely.

    On virtiofs / FUSE mounts GDAL cannot unlink existing files, so we write
    to a temp file in the system temp dir then copy the bytes over the existing
    path using shutil.copy2 (which only needs write permission, not unlink).
    """
    if os.path.exists(out_path):
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".tif", dir=tempfile.gettempdir()
        )
        try:
            os.close(tmp_fd)  # close early so rasterio can open it
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(arr)
            shutil.copy2(tmp_path, out_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMODNET_WCS_URL  = "https://ows.emodnet-bathymetry.eu/wcs"
GEBCO_DOWNLOAD   = "https://download.gebco.net/"
IHM_WCS_URL      = "https://geomar.hidrografico.pt/geoserver/geomar/wcs"
ETOPO_WCS_URL    = (
    "https://www.ncei.noaa.gov/thredds/wcs/etopo/etopo_60s_v2022.nc"
)

# Degrees per metre (approximate at mid-latitudes)
DEG_PER_M = 1.0 / 111_320.0

log = logging.getLogger("reef_bathy")


# ===========================================================================
# Helpers
# ===========================================================================

def _setup_log(output_dir: str) -> None:
    if not log.handlers:
        fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        log_path = os.path.join(output_dir, "bathy.log")
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )


def _aoi_bbox(lat: float, lon: float, buffer_m: float, scale: float = 1.0):
    """Return (W, S, E, N) WGS-84 bbox enlarged by *scale* × buffer_m."""
    d = buffer_m * scale * DEG_PER_M
    return lon - d, lat - d, lon + d, lat + d


def _date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _validate_tiff(path: str) -> bool:
    try:
        with rasterio.open(path) as ds:
            return ds.count > 0 and ds.width > 0
    except Exception:
        return False


def _meters_per_pixel(lat: float, res_deg: float) -> float:
    """Approximate pixel width in metres at given latitude."""
    lat_m = 111_320.0 * res_deg
    lon_m = 111_320.0 * res_deg * abs(np.cos(np.radians(lat)))
    return (lat_m + lon_m) / 2.0


def _pix_dims(bbox, target_m: float = 100.0, max_px: int = 512):
    """Return (width, height) in pixels for a bbox at target_m resolution."""
    w, s, e, n = bbox
    lat_c = (s + n) / 2.0
    px_w = max(1, min(max_px, int(abs(e - w) * 111_320.0 / target_m)))
    px_h = max(1, min(max_px, int(abs(n - s) * 111_320.0 / target_m)))
    return px_w, px_h


# ===========================================================================
# 1.  EMODnet Bathymetry — WCS 1.0.0
# ===========================================================================

def download_emodnet_bathy(
    lat: float, lon: float, buffer_m: float, output_dir: str,
    date_str: str = "",
) -> str | None:
    """
    Download EMODnet bathymetry via WCS 1.0.0.

    Product : EMODnet composite mean bathymetry (DTM)
    Coverage: emodnet:mean  (~115 m / 1/128°)
    Auth    : None
    Returns : Path to GeoTIFF, or None on failure.
    """
    if not date_str:
        date_str = _date_str()
    out_path = os.path.join(output_dir, f"bathy_emodnet_{date_str}.tif")

    # Use 10× buffer for context (BPI needs surrounding terrain)
    bbox = _aoi_bbox(lat, lon, buffer_m, scale=12.0)
    w, s, e, n = bbox
    px_w, px_h = _pix_dims(bbox, target_m=115, max_px=512)

    params = {
        "SERVICE":  "WCS",
        "VERSION":  "1.0.0",
        "REQUEST":  "GetCoverage",
        "COVERAGE": "emodnet:mean",
        "CRS":      "EPSG:4326",
        "BBOX":     f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}",
        "WIDTH":    str(px_w),
        "HEIGHT":   str(px_h),
        "FORMAT":   "image/tiff",
    }
    log.info("→ EMODnet WCS request (bbox=%.4f,%.4f,%.4f,%.4f, %dx%d px)",
             w, s, e, n, px_w, px_h)
    try:
        r = requests.get(EMODNET_WCS_URL, params=params, timeout=60)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "xml" in ctype.lower() or r.content[:6] in (b"<Servi", b"<?xml "):
            log.warning("   EMODnet returned XML error: %s", r.content[:300])
            return None
        with open(out_path, "wb") as f:
            f.write(r.content)
        if _validate_tiff(out_path):
            log.info("   ✓ EMODnet GeoTIFF → %s (%.1f kB)",
                     out_path, os.path.getsize(out_path) / 1024)
            return out_path
        log.warning("   EMODnet: saved file failed TIFF validation")
        return None
    except Exception as exc:
        log.warning("   EMODnet download failed: %s", exc)
        return None


# ===========================================================================
# 2.  GEBCO 2024 — sub-grid ZIP download
# ===========================================================================

def download_gebco(
    lat: float, lon: float, buffer_m: float, output_dir: str,
    date_str: str = "",
) -> str | None:
    """
    Download GEBCO 2024 sub-grid via BODC/GEBCO download service.

    Returns a ZIP containing a NetCDF4 file; converts to GeoTIFF with rasterio.
    Coverage: ~460 m (15 arc-second)
    Auth    : None
    Returns : Path to GeoTIFF, or None on failure.
    """
    if not date_str:
        date_str = _date_str()
    out_path = os.path.join(output_dir, f"bathy_gebco_{date_str}.tif")

    bbox = _aoi_bbox(lat, lon, buffer_m, scale=12.0)
    w, s, e, n = bbox

    # GEBCO sub-grid download (returns ZIP with NetCDF4)
    params = {
        "type":   "sub_ice_topo",
        "format": "zip",
        "west":   f"{w:.4f}",
        "east":   f"{e:.4f}",
        "south":  f"{s:.4f}",
        "north":  f"{n:.4f}",
    }
    log.info("→ GEBCO 2024 sub-grid download (bbox=%.4f,%.4f,%.4f,%.4f)", w, s, e, n)
    try:
        r = requests.get(GEBCO_DOWNLOAD, params=params, timeout=120,
                         allow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        # Expect ZIP
        if "zip" in ctype or r.content[:2] == b"PK":
            nc_path = _extract_gebco_zip(r.content, output_dir, date_str)
            if nc_path:
                ok = _nc_to_geotiff(nc_path, out_path, varname="elevation")
                if ok and _validate_tiff(out_path):
                    log.info("   ✓ GEBCO GeoTIFF → %s (%.1f kB)",
                             out_path, os.path.getsize(out_path) / 1024)
                    return out_path
        # Some versions return GeoTIFF directly
        elif "tiff" in ctype or r.content[:4] == b"II*\x00":
            with open(out_path, "wb") as f:
                f.write(r.content)
            if _validate_tiff(out_path):
                log.info("   ✓ GEBCO GeoTIFF (direct) → %s", out_path)
                return out_path
        log.warning("   GEBCO: unexpected content-type '%s'", ctype)
        return None
    except Exception as exc:
        log.warning("   GEBCO download failed: %s", exc)
        return None


def _extract_gebco_zip(data: bytes, output_dir: str, date_str: str) -> str | None:
    """Extract the NetCDF4 file from a GEBCO zip payload."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                log.warning("   GEBCO ZIP: no .nc file found (names: %s)",
                            zf.namelist()[:5])
                return None
            nc_name = nc_names[0]
            nc_path = os.path.join(output_dir, f"gebco_raw_{date_str}.nc")
            with zf.open(nc_name) as src, open(nc_path, "wb") as dst:
                dst.write(src.read())
            log.info("   GEBCO: extracted %s → %s", nc_name, nc_path)
            return nc_path
    except Exception as exc:
        log.warning("   GEBCO ZIP extraction failed: %s", exc)
        return None


def _nc_to_geotiff(nc_path: str, out_path: str, varname: str = "elevation") -> bool:
    """
    Convert a NetCDF file to a single-band GeoTIFF using rasterio/GDAL.
    GDAL's NetCDF driver can open NetCDF4 files as NETCDF:file.nc:varname.
    """
    gdal_path = f"NETCDF:{nc_path}:{varname}"
    try:
        with rasterio.open(gdal_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            profile = {
                "driver":    "GTiff",
                "dtype":     rasterio.float32,
                "width":     src.width,
                "height":    src.height,
                "count":     1,
                "crs":       src.crs if src.crs else CRS.from_epsg(4326),
                "transform": src.transform,
                "compress":  "lzw",
                "nodata":    np.nan,
            }
            _safe_write_tif(out_path, profile, data[np.newaxis, ...] if data.ndim == 2 else data)
        return True
    except Exception as exc:
        log.warning("   NC→GeoTIFF conversion failed: %s", exc)
        return False


# ===========================================================================
# 3.  Portugal IHM / GEOMAR — WCS
# ===========================================================================

def download_geomar(
    lat: float, lon: float, buffer_m: float, output_dir: str,
    date_str: str = "",
) -> str | None:
    """
    Attempt to download IHM/GEOMAR bathymetry via GeoServer WCS.

    The Instituto Hidrográfico de Marinha (IHM) hosts GEOMAR / SEAMAP 2030 data
    at geomar.hidrografico.pt. This function tries to fetch a coverage from that
    endpoint; if the service is unavailable or the coverage name has changed,
    it logs a warning and returns None.

    Coverage: GEOMAR_SEAMAP (variable name / ~25–100 m)
    Auth    : None (public WCS)
    Returns : Path to GeoTIFF, or None on failure.
    """
    if not date_str:
        date_str = _date_str()
    out_path = os.path.join(output_dir, f"bathy_geomar_{date_str}.tif")

    bbox = _aoi_bbox(lat, lon, buffer_m, scale=12.0)
    w, s, e, n = bbox
    px_w, px_h = _pix_dims(bbox, target_m=100, max_px=512)

    # Known IHM GeoServer WCS coverages for Algarve bathymetry.
    # Try in order; fall through on failure.
    coverage_candidates = [
        "geomar:algarve_bathy",
        "geomar:portugal_bathy",
        "geomar:bathymetry",
    ]

    for coverage in coverage_candidates:
        params = {
            "SERVICE":    "WCS",
            "VERSION":    "1.0.0",
            "REQUEST":    "GetCoverage",
            "COVERAGE":   coverage,
            "CRS":        "EPSG:4326",
            "BBOX":       f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}",
            "WIDTH":      str(px_w),
            "HEIGHT":     str(px_h),
            "FORMAT":     "GeoTIFF",
        }
        log.info("→ IHM GEOMAR WCS (coverage=%s) …", coverage)
        try:
            r = requests.get(IHM_WCS_URL, params=params, timeout=45)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "xml" in ctype.lower() or r.content[:5] in (b"<?xml", b"<Serv"):
                log.debug("   IHM: coverage '%s' returned XML (skip)", coverage)
                continue
            with open(out_path, "wb") as f:
                f.write(r.content)
            if _validate_tiff(out_path):
                log.info("   ✓ IHM GEOMAR GeoTIFF → %s (%.1f kB)",
                         out_path, os.path.getsize(out_path) / 1024)
                return out_path
        except requests.exceptions.HTTPError as exc:
            log.debug("   IHM HTTP error for '%s': %s", coverage, exc)
        except Exception as exc:
            log.warning("   IHM GEOMAR request failed: %s", exc)
            break  # network error — don't retry

    log.warning(
        "   IHM GEOMAR: no coverage available for Algarve AOI "
        "(service may require authentication or coverage names may differ)"
    )
    return None


# ===========================================================================
# 4.  NOAA ETOPO — THREDDS WCS
# ===========================================================================

def download_etopo(
    lat: float, lon: float, buffer_m: float, output_dir: str,
    date_str: str = "",
) -> str | None:
    """
    Download NOAA ETOPO 2022 (60 arc-second, ~1.8 km) via THREDDS WCS.

    Auth    : None
    Returns : Path to GeoTIFF, or None on failure.
    """
    if not date_str:
        date_str = _date_str()
    out_path = os.path.join(output_dir, f"bathy_etopo_{date_str}.tif")

    bbox = _aoi_bbox(lat, lon, buffer_m, scale=15.0)
    w, s, e, n = bbox

    params = {
        "service":  "WCS",
        "version":  "1.0.0",
        "request":  "GetCoverage",
        "coverage": "Band1",
        "bbox":     f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}",
        "crs":      "EPSG:4326",
        "format":   "GeoTIFF",
        "resx":     "0.01667",
        "resy":     "0.01667",
    }
    log.info("→ NOAA ETOPO WCS (bbox=%.4f,%.4f,%.4f,%.4f)", w, s, e, n)
    try:
        r = requests.get(ETOPO_WCS_URL, params=params, timeout=60)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "xml" in ctype.lower() or r.content[:5] in (b"<?xml", b"<Serv"):
            log.warning("   ETOPO returned XML error: %s", r.content[:300])
            return None
        with open(out_path, "wb") as f:
            f.write(r.content)
        if _validate_tiff(out_path):
            log.info("   ✓ ETOPO GeoTIFF → %s (%.1f kB)",
                     out_path, os.path.getsize(out_path) / 1024)
            return out_path
        log.warning("   ETOPO: file failed TIFF validation")
        return None
    except Exception as exc:
        log.warning("   ETOPO download failed: %s", exc)
        return None


# ===========================================================================
# 4b. DGT LiDAR Bathymetry (STAC/S3)
# ===========================================================================

def download_dgt_lidar(
    lat: float, lon: float, buffer_m: float, output_dir: str,
    date_str: str = "",
) -> str | None:
    """
    Download high-resolution 50cm MDT from DGT STAC Catalog via S3.
    Auth: AWS env vars from .env for bucket access.
    """
    if not date_str:
        date_str = _date_str()
    out_path = os.path.join(output_dir, f"bathy_dgt_lidar_{date_str}.tif")

    bbox = _aoi_bbox(lat, lon, buffer_m, scale=1.5)
    w, s, e, n = bbox

    stac_url = "https://dgt-be.a.incd.pt:8081/search"
    payload = {
        "bbox": [w, s, e, n],
        "collections": ["MDT-50cm"],
        "limit": 10
    }
    log.info("→ DGT LiDAR STAC request (bbox=%.4f,%.4f,%.4f,%.4f)", w, s, e, n)
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.post(stac_url, json=payload, timeout=60, verify=False)
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            log.warning("   DGT LiDAR: No MDT-50cm data found for this bounding box.")
            return None
            
        import boto3
        from rasterio.session import AWSSession
        from rasterio.merge import merge
        
        endpoint_url = os.environ.get("AWS_S3_ENDPOINT", "https://stor-001.d.acnca.pt")
        boto3_session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        
        src_files = []
        for feat in features:
            assets = feat.get("assets", {})
            for asset_key, asset_val in assets.items():
                href = asset_val.get("href")
                if href and href.endswith(".tif"):
                    # Convert to s3:// URI if it's HTTP
                    if href.startswith("http"):
                        parts = href.replace("https://", "").replace("http://", "").split("/")
                        if len(parts) >= 3:
                            href = f"s3://{parts[1]}/{'/'.join(parts[2:])}"
                    src_files.append(href)
                    break
                    
        if not src_files:
            log.warning("   DGT LiDAR: Could not find TIFF assets in STAC features.")
            return None
            
        log.info("   Found %d DGT LiDAR sources. Cropping & Merging...", len(src_files))
        
        env = rasterio.Env(
            AWSSession(boto3_session, endpoint_url=endpoint_url),
            AWS_NO_SIGN_REQUEST="NO" if "AWS_ACCESS_KEY_ID" in os.environ else "YES",
            AWS_VIRTUAL_HOSTING="FALSE",
        )
        
        datasets = []
        with env:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for s3_url in src_files:
                    try:
                        datasets.append(rasterio.open(s3_url))
                    except Exception as e:
                        log.warning("   Could not open %s: %s", s3_url, e)
                        
                if not datasets:
                    return None
                    
                mosaic, out_trans = merge(datasets, bounds=(w, s, e, n))
                out_meta = datasets[0].meta.copy()
                
            for ds in datasets:
                ds.close()

        if mosaic.size == 0 or np.all(mosaic == out_meta.get("nodata")):
            log.warning("   DGT LiDAR: Merged array is empty or nodata.")
            return None

        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        _safe_write_tif(out_path, out_meta, mosaic)
        
        if _validate_tiff(out_path):
            log.info("   ✓ DGT LiDAR GeoTIFF → %s", out_path)
            return out_path
            
        return None
    except Exception as exc:
        log.warning("   DGT LiDAR fetch failed: %s", exc)
        return None

# ===========================================================================
# 5.  Sentinel-2 Shallow Depth Inversion (Stumpf 2003)
# ===========================================================================

def compute_s2_depth_inversion(
    b02_path: str, b03_path: str, output_dir: str,
    b08_path: str = "",
    m0: float = -28.0, m1: float = 32.0, n_scale: float = 1500.0,
    date_str: str = "",
) -> str | None:
    """
    Compute optically-shallow bathymetry from Sentinel-2 B02/B03.

    Uses the Stumpf et al. (2003) log-ratio transform:
        depth = m1 * ln(n * Rw_blue) / ln(n * Rw_green) + m0

    Default m0/m1 are calibrated for the Algarve oligotrophic coastal water.
    If EMODnet bathymetry is available in the output_dir, it will attempt
    robust dynamic calibration.

    Returns : Path to depth GeoTIFF (negative = below sea level), or None.
    """
    if not date_str:
        date_str = _date_str()
    out_path = os.path.join(output_dir, f"bathy_s2_stumpf_{date_str}.tif")

    if not (os.path.exists(b02_path) and os.path.exists(b03_path)):
        log.warning("   S2 SDB: B02/B03 files not found (%s, %s)", b02_path, b03_path)
        return None

    log.info("→ Sentinel-2 Stumpf depth inversion (B02=%s, B03=%s)", b02_path, b03_path)
    
    # Try dynamic calibration against EMODnet if present
    emodnet_path = os.path.join(output_dir, f"bathy_emodnet_{date_str}.tif")
    if os.path.exists(emodnet_path):
        log.info("   Found EMODnet bathy! Attempting dynamic calibration...")
        emodnet_10m = None
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.stumpf_emodnet_calibration import (
                reproject_emodnet_to_s2,
                calibrate_stumpf_vs_emodnet,
                validate_reprojected_emodnet,
            )

            # Temporary reprojected file
            emodnet_10m = os.path.join(output_dir, f"bathy_emodnet_10m_{date_str}.tif")
            reproject_emodnet_to_s2(emodnet_path, b02_path, emodnet_10m)
            validate_reprojected_emodnet(emodnet_10m, min_valid_pixels=100)

            res = calibrate_stumpf_vs_emodnet(b02_path, b03_path, emodnet_10m, out_path)
            log.info("   ✓ S2 dynamic calibration success! m0=%.3f, m1=%.3f, RMSE=%.3fm", 
                     res['m0'], res['m1'], res['rmse'])
            return out_path
        except Exception as e:
            log.warning("   Dynamic calibration failed (%s). Falling back to default Stumpf...", e)
        finally:
            if emodnet_10m and os.path.exists(emodnet_10m):
                try:
                    os.remove(emodnet_10m)
                except OSError:
                    pass
    
    # Fallback to default Stumpf inversion
    try:
        with rasterio.open(b02_path) as src_b02:
            b02 = src_b02.read(1).astype(np.float32)
            profile = src_b02.profile.copy()
            transform = src_b02.transform
            crs = src_b02.crs

        with rasterio.open(b03_path) as src_b03:
            b03 = src_b03.read(1).astype(np.float32)

        # Apply empirical sunglint correction if NIR (B08) is provided
        if b08_path and os.path.exists(b08_path):
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from src.atmospheric_corrector import AtmosphericCorrector
                
                with rasterio.open(b08_path) as src_b08:
                    b08 = src_b08.read(1).astype(np.float32)
                
                ac = AtmosphericCorrector("sentinel-2")
                b02 = ac.empirical_sunglint(b02, b08)
                
                # Approximate B03 correction (using B02 factor for simplicity)
                b03 = ac.empirical_sunglint(b03, b08)
                log.info("   ✓ Empirical sunglint correction applied using B08")
            except Exception as e:
                log.warning("   Sunglint correction failed, continuing with raw bands: %s", e)

        # Mask invalid / zero reflectance
        valid = (b02 > 0) & (b03 > 0)
        b02_s = np.where(valid, b02, np.nan)
        b03_s = np.where(valid, b03, np.nan)

        # Stumpf log-ratio  (depth increases with higher blue/green ratio)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_b = np.log(n_scale * b02_s)
            log_g = np.log(n_scale * b03_s)
            # Avoid division by tiny/negative log
            safe = (log_g > 0) & np.isfinite(log_b) & np.isfinite(log_g)
            ratio = np.where(safe, log_b / log_g, np.nan)
            depth = np.where(safe, m1 * ratio + m0, np.nan)

        # Depth is positive-up from Stumpf; negate to give negative-below-sea-level
        depth = -np.abs(depth)

        # Mask obviously-land and impossibly-deep values
        depth = np.where((depth > 0) | (depth < -60), np.nan, depth)

        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.tide_calibrator import get_tide_offset
        
        # Calculate center coordinates for tide API
        lat_c = (src_b02.bounds.top + src_b02.bounds.bottom) / 2.0
        lon_c = (src_b02.bounds.left + src_b02.bounds.right) / 2.0
        
        tide_offset = get_tide_offset(date_str, lat_c, lon_c)
        depth_corrected = depth - tide_offset

        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
        
        # Write raw
        raw_path = out_path.replace(".tif", "_raw.tif")
        _safe_write_tif(raw_path, profile, depth.astype(np.float32)[np.newaxis, ...])
        
        # Write corrected
        _safe_write_tif(out_path, profile, depth_corrected.astype(np.float32)[np.newaxis, ...])

        valid_pct = 100.0 * np.sum(np.isfinite(depth_corrected)) / depth_corrected.size
        d_med = float(np.nanmedian(depth_corrected))
        log.info("   ✓ S2 depth (tide offset: -%.2fm) → %s  (valid=%.0f%%, median=%.1f m)",
                 tide_offset, out_path, valid_pct, d_med)
        return out_path
    except Exception as exc:
        log.error("   S2 depth inversion failed: %s", exc)
        return None


# ===========================================================================
# 6.  ASCII XYZ → GeoTIFF converter
# ===========================================================================

def ascii_to_geotiff(
    ascii_path: str, output_dir: str, epsg: int = 4326, date_str: str = "",
) -> str | None:
    """
    Convert an XYZ ASCII file (lon lat depth, space/tab/comma delimited)
    to a regular-grid GeoTIFF.

    Handles:
    - Columns in any order detected by header (if present) or positional.
    - Irregular spacing: re-grids to regular grid by nearest-neighbour.
    """
    if not os.path.exists(ascii_path):
        log.warning("   ASCII→GeoTIFF: file not found: %s", ascii_path)
        return None
    if not date_str:
        date_str = _date_str()
    base = os.path.splitext(os.path.basename(ascii_path))[0]
    out_path = os.path.join(output_dir, f"{base}_{date_str}.tif")

    log.info("→ ASCII→GeoTIFF: %s", ascii_path)
    try:
        from scipy.interpolate import griddata  # type: ignore

        pts, vals = [], []
        with open(ascii_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").replace("\t", " ").split()
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    pts.append((x, y))
                    vals.append(z)
                except ValueError:
                    continue  # header line

        if not pts:
            log.warning("   ASCII→GeoTIFF: no valid XYZ points found")
            return None

        pts_arr = np.array(pts)
        vals_arr = np.array(vals, dtype=np.float32)
        xmin, xmax = pts_arr[:, 0].min(), pts_arr[:, 0].max()
        ymin, ymax = pts_arr[:, 1].min(), pts_arr[:, 1].max()

        # Infer grid resolution from median spacing
        n = min(len(pts_arr), 500)
        sample_x = np.sort(np.unique(np.round(pts_arr[:n, 0], 4)))
        res = float(np.median(np.diff(sample_x))) if len(sample_x) > 1 else 0.001
        res = max(res, 1e-5)

        nx = max(2, int(round((xmax - xmin) / res)) + 1)
        ny = max(2, int(round((ymax - ymin) / res)) + 1)
        xi = np.linspace(xmin, xmax, nx)
        yi = np.linspace(ymax, ymin, ny)  # north-up
        gx, gy = np.meshgrid(xi, yi)

        grid = griddata(pts_arr, vals_arr, (gx, gy), method="nearest").astype(np.float32)

        transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
        profile = {
            "driver":    "GTiff",
            "dtype":     rasterio.float32,
            "width":     nx,
            "height":    ny,
            "count":     1,
            "crs":       CRS.from_epsg(epsg),
            "transform": transform,
            "compress":  "lzw",
            "nodata":    np.nan,
        }
        _safe_write_tif(out_path, profile, grid[np.newaxis, ...])

        log.info("   ✓ ASCII→GeoTIFF → %s (%dx%d px)", out_path, nx, ny)
        return out_path
    except Exception as exc:
        log.error("   ASCII→GeoTIFF failed: %s", exc)
        return None


# ===========================================================================
# 7.  Bathymetric indices (slope, TRI, BPI, curvature)
# ===========================================================================

def compute_bathy_indices(
    bathy_tif: str, output_dir: str,
) -> dict[str, str]:
    """
    Compute morphological indices from a bathymetry GeoTIFF:
      - slope_deg          : local slope in degrees
      - tri                : Terrain Ruggedness Index (Riley 1999)
      - bpi_fine           : Bathymetric Position Index, 3×3 kernel
      - bpi_broad          : Bathymetric Position Index, 15×15 kernel
      - curvature          : Laplacian (plan) curvature

    Returns dict mapping index name → output tif path.
    Written to output_dir with prefix matching input filename.
    """
    if not _validate_tiff(bathy_tif):
        log.warning("   Indices: invalid/missing input %s", bathy_tif)
        return {}

    log.info("→ Computing bathymetric indices from %s", bathy_tif)
    base = os.path.splitext(os.path.basename(bathy_tif))[0]
    results: dict[str, str] = {}

    try:
        with rasterio.open(bathy_tif) as src:
            dem = src.read(1).astype(np.float64)
            nodata = src.nodata
            profile = src.profile.copy()
            transform = src.transform
            res_x = abs(float(transform.a))  # degrees
            res_y = abs(float(transform.e))  # degrees
            lat_c = (src.bounds.bottom + src.bounds.top) / 2.0

        # Convert nodata to NaN
        if nodata is not None:
            dem = np.where(np.isnan(dem) | (dem == nodata), np.nan, dem)

        # Resolution in metres (approximate)
        dx_m = res_x * 111_320.0 * abs(np.cos(np.radians(lat_c)))
        dy_m = res_y * 111_320.0

        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        def _write(name: str, arr: np.ndarray) -> str:
            path = os.path.join(output_dir, f"{base}_{name}.tif")
            _safe_write_tif(path, profile, arr.astype(np.float32)[np.newaxis, ...])
            return path

        # --- Slope ---
        dz_dx = np.gradient(np.where(np.isfinite(dem), dem, 0.0), dx_m, axis=1)
        dz_dy = np.gradient(np.where(np.isfinite(dem), dem, 0.0), dy_m, axis=0)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        slope = np.where(np.isfinite(dem), slope, np.nan)
        results["slope_deg"] = _write("slope_deg", slope)
        log.info("   ✓ Slope → %s", results["slope_deg"])

        # --- TRI (Terrain Ruggedness Index) — fully vectorised ---
        # TRI = sqrt( sum((neighbour - centre)²) / 8 )  [Riley 1999]
        # Uses np.pad + slicing to replicate 'nearest' boundary mode,
        # avoiding Python-level per-pixel callbacks (~262k calls on 512×512).
        dem_filled = np.where(np.isfinite(dem), dem, 0.0)
        _pad = np.pad(dem_filled, 1, mode="edge")
        _centre = _pad[1:-1, 1:-1]
        _sq_sum = np.zeros_like(dem_filled, dtype=np.float64)
        for _dr in range(3):
            for _dc in range(3):
                if _dr == 1 and _dc == 1:
                    continue  # skip centre
                _nbr = _pad[_dr:_dr + dem_filled.shape[0],
                            _dc:_dc + dem_filled.shape[1]]
                _sq_sum += (_nbr - _centre) ** 2
        tri = np.sqrt(_sq_sum / 8.0)
        tri = np.where(np.isfinite(dem), tri, np.nan)
        results["tri"] = _write("tri", tri)
        log.info("   ✓ TRI → %s", results["tri"])

        # --- BPI fine (3×3) ---
        mean_fine = uniform_filter(dem_filled, size=3, mode="nearest")
        bpi_fine = dem - mean_fine
        bpi_fine = np.where(np.isfinite(dem), bpi_fine, np.nan)
        results["bpi_fine"] = _write("bpi_fine", bpi_fine)
        log.info("   ✓ BPI fine (3×3) → %s", results["bpi_fine"])

        # --- BPI broad (15×15) ---
        mean_broad = uniform_filter(dem_filled, size=15, mode="nearest")
        bpi_broad = dem - mean_broad
        bpi_broad = np.where(np.isfinite(dem), bpi_broad, np.nan)
        results["bpi_broad"] = _write("bpi_broad", bpi_broad)
        log.info("   ✓ BPI broad (15×15) → %s", results["bpi_broad"])

        # --- Curvature (Laplacian) ---
        d2z_dx2 = np.gradient(dz_dx, dx_m, axis=1)
        d2z_dy2 = np.gradient(dz_dy, dy_m, axis=0)
        curvature = d2z_dx2 + d2z_dy2
        curvature = np.where(np.isfinite(dem), curvature, np.nan)
        results["curvature"] = _write("curvature", curvature)
        log.info("   ✓ Curvature → %s", results["curvature"])

    except Exception as exc:
        log.error("   Indices computation failed: %s", exc)

    return results


# ===========================================================================
# 8.  Reef candidate detection
# ===========================================================================

def detect_reef_candidates(
    bathy_tif: str,
    indices_dir: str,
    output_dir: str,
    depth_min: float = -50.0,
    depth_max: float = -1.0,
    tri_pct: float = 70.0,
    bpi_pct: float = 60.0,
    min_area_px: int = 4,
    date_str: str = "",
) -> str | None:
    """
    Detect reef candidate polygons from bathymetry + morphological indices.

    Algorithm:
      1. Depth mask: keep cells in [depth_min, depth_max]
      2. High rugosity: TRI > tri_pct-th percentile of valid cells
      3. Positive BPI (elevated relative to surroundings): BPI_broad > bpi_pct-th percentile
      4. Connected-component labelling; remove patches < min_area_px
      5. Vectorise to polygons via rasterio.features.shapes

    Returns path to GeoJSON of candidate polygons, or None on failure.
    Requires shapely + geopandas.
    """
    if not HAS_SHAPELY:
        log.error("   detect_reef_candidates requires shapely + geopandas. "
                  "Install: pip install shapely geopandas")
        return None

    if not _validate_tiff(bathy_tif):
        log.warning("   Reef detection: invalid/missing bathy: %s", bathy_tif)
        return None
    if not date_str:
        date_str = _date_str()

    base = os.path.splitext(os.path.basename(bathy_tif))[0]
    log.info("→ Detecting reef candidates from %s", bathy_tif)

    try:
        with rasterio.open(bathy_tif) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)

        # 1. Depth mask
        depth_mask = (dem >= depth_min) & (dem <= depth_max) & np.isfinite(dem)
        if depth_mask.sum() == 0:
            log.warning("   Reef detection: no pixels in depth range [%.0f, %.0f] m",
                        depth_min, depth_max)
            return None

        # 2. TRI
        tri_path = os.path.join(indices_dir, f"{base}_tri.tif")
        if not os.path.exists(tri_path):
            log.warning("   Reef detection: TRI file missing %s (run indices first)", tri_path)
            return None
        with rasterio.open(tri_path) as src:
            tri = src.read(1).astype(np.float32)

        tri_valid = tri[depth_mask & np.isfinite(tri)]
        if len(tri_valid) == 0:
            log.warning("   Reef detection: no valid TRI values in depth window")
            return None
        tri_thresh = np.percentile(tri_valid, tri_pct)
        rugose = np.isfinite(tri) & (tri >= tri_thresh)

        # 3. Positive BPI (broad)
        bpi_path = os.path.join(indices_dir, f"{base}_bpi_broad.tif")
        if not os.path.exists(bpi_path):
            log.warning("   Reef detection: BPI file missing %s", bpi_path)
            return None
        with rasterio.open(bpi_path) as src:
            bpi = src.read(1).astype(np.float32)

        bpi_valid = bpi[depth_mask & np.isfinite(bpi)]
        bpi_thresh = np.percentile(bpi_valid, bpi_pct) if len(bpi_valid) > 0 else 0.0
        elevated = np.isfinite(bpi) & (bpi >= bpi_thresh)

        # 4. Combined mask
        candidate_mask = depth_mask & rugose & elevated
        log.info("   Candidate cells: %d / %d (depth) / %d total",
                 candidate_mask.sum(), depth_mask.sum(), dem.size)

        if candidate_mask.sum() == 0:
            log.warning("   No reef candidates found (loosen thresholds?)")
            return _empty_geojson(output_dir, date_str, crs)

        # 5. Connected components, min-area filter
        labelled, n_features = ndimage_label(candidate_mask)
        log.info("   Connected components: %d", n_features)

        sizes = np.bincount(labelled.ravel())
        small = np.where(sizes < min_area_px)[0]
        remove_mask = np.isin(labelled, small)
        filtered = np.where(remove_mask, 0, candidate_mask.astype(np.uint8))

        # 6. Vectorise
        geoms = []
        for geom_dict, val in rasterio_shapes(
            filtered.astype(np.uint8), mask=filtered, transform=transform
        ):
            if val == 1:
                geoms.append(shape(geom_dict))

        if not geoms:
            log.warning("   No polygons after filtering — writing empty GeoJSON")
            return _empty_geojson(output_dir, date_str, crs)

        # Merge adjacent polygons and build GeoDataFrame
        merged = [unary_union(geoms)] if len(geoms) > 50 else geoms
        gdf = gpd.GeoDataFrame(
            {
                "source_bathy": [base] * len(merged),
                "depth_min_m":  [depth_min] * len(merged),
                "depth_max_m":  [depth_max] * len(merged),
                "detection_date": [date_str] * len(merged),
                "area_m2":       [g.area * (111_320**2) for g in merged],
            },
            geometry=merged,
            crs=crs.to_epsg() if crs else 4326,
        )

        out_path = os.path.join(output_dir, f"reef_candidates_{date_str}.geojson")
        gdf.to_file(out_path, driver="GeoJSON")
        log.info("   ✓ Reef candidates → %s  (%d polygons)", out_path, len(gdf))
        return out_path

    except Exception as exc:
        log.error("   Reef detection failed: %s", exc)
        return None


def _empty_geojson(output_dir: str, date_str: str, crs) -> str:
    """Write an empty GeoJSON FeatureCollection."""
    path = os.path.join(output_dir, f"reef_candidates_{date_str}.geojson")
    fc = {"type": "FeatureCollection", "features": []}
    with open(path, "w") as f:
        json.dump(fc, f)
    log.info("   Empty reef candidates → %s", path)
    return path


# ===========================================================================
# 9.  Export helpers
# ===========================================================================

def export_candidates_geojson(
    candidates_path: str, output_dir: str, date_str: str = "",
) -> str | None:
    """
    Copy / re-export candidates GeoJSON with a date-stamped filename.
    Useful when the source was produced in a temp location.
    Returns output path.
    """
    if not date_str:
        date_str = _date_str()
    if not candidates_path or not os.path.exists(candidates_path):
        log.warning("   export_candidates_geojson: source not found: %s", candidates_path)
        return None

    out_path = os.path.join(output_dir, f"reef_candidates_{date_str}.geojson")
    if os.path.abspath(candidates_path) == os.path.abspath(out_path):
        return out_path  # already there

    import shutil
    shutil.copy2(candidates_path, out_path)
    log.info("   ✓ GeoJSON → %s", out_path)
    return out_path


def add_bathy_to_qgis(
    qgs_path: str,
    bathy_tif: str | None,
    candidates_geojson: str | None,
    indices: dict[str, str] | None = None,
) -> bool:
    """
    Inject bathymetry + reef candidate layers into an existing .qgs project.

    The .qgs format (QGIS 3.x) uses:
      <projectlayers>
        <maplayer type="raster"> ... </maplayer>
        ...
      </projectlayers>

    New layers are inserted just before </projectlayers>.
    Returns True on success.
    """
    if not os.path.exists(qgs_path):
        log.warning("   QGIS inject: project not found: %s", qgs_path)
        return False

    log.info("→ Injecting bathy layers into QGIS project: %s", qgs_path)

    new_layers: list[str] = []

    if bathy_tif and os.path.exists(bathy_tif):
        abs_tif = os.path.abspath(bathy_tif)
        bname = os.path.splitext(os.path.basename(bathy_tif))[0]
        new_layers.append(f"""
    <!-- Bathymetry raster (reef_bathy_module) -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>bathy_{bname}</id>
      <datasource>{abs_tif}</datasource>
      <layername>Bathymetry — {bname}</layername>
      <srs><spatialrefsys><authid>EPSG:4326</authid></spatialrefsys></srs>
    </maplayer>""")

    # Optional index layers (slope, TRI, BPI)
    if indices:
        for idx_name, idx_path in indices.items():
            if idx_path and os.path.exists(idx_path):
                abs_idx = os.path.abspath(idx_path)
                ibase = os.path.splitext(os.path.basename(idx_path))[0]
                new_layers.append(f"""
    <!-- Bathymetric index: {idx_name} -->
    <maplayer type="raster" autoRefreshEnabled="0">
      <id>bathy_idx_{ibase}</id>
      <datasource>{abs_idx}</datasource>
      <layername>Bathy index — {idx_name}</layername>
      <srs><spatialrefsys><authid>EPSG:4326</authid></spatialrefsys></srs>
    </maplayer>""")

    if candidates_geojson and os.path.exists(candidates_geojson):
        abs_geojson = os.path.abspath(candidates_geojson)
        gjbase = os.path.splitext(os.path.basename(candidates_geojson))[0]
        new_layers.append(f"""
    <!-- Reef candidates (vector, reef_bathy_module) -->
    <maplayer type="vector" autoRefreshEnabled="0">
      <id>reef_candidates_{gjbase}</id>
      <datasource>{abs_geojson}</datasource>
      <layername>Reef Candidates — {gjbase}</layername>
      <srs><spatialrefsys><authid>EPSG:4326</authid></spatialrefsys></srs>
    </maplayer>""")

    if not new_layers:
        log.info("   No new layers to inject — skipping QGS update")
        return True

    try:
        with open(qgs_path, "r", encoding="utf-8") as f:
            content = f.read()

        tag = "</projectlayers>"
        if tag not in content:
            log.warning("   QGIS inject: </projectlayers> tag not found in %s", qgs_path)
            return False

        injection = "\n".join(new_layers) + "\n\n  "
        content = content.replace(tag, injection + tag, 1)

        with open(qgs_path, "w", encoding="utf-8") as f:
            f.write(content)

        log.info("   ✓ Injected %d layer(s) into %s", len(new_layers), qgs_path)
        return True
    except Exception as exc:
        log.error("   QGIS inject failed: %s", exc)
        return False


# ===========================================================================
# 10.  Pipeline entry point
# ===========================================================================

def run_bathy_step(args) -> None:
    """
    Main entry point called by reef_imagery_pipeline_v3 when --step bathy.

    Expects args to have:
      .output_dir    : str
      .lat           : float
      .lon           : float
      .buffer_m      : float
      .bathy_source  : str  — "emodnet" | "gebco" | "geomar" | "etopo" | "s2" | "all"
      .depth_min     : float
      .depth_max     : float
      .date          : str   (YYYY-MM-DD, used to find existing S2 bands)
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    _setup_log(output_dir)

    lat      = float(getattr(args, "lat",       37.069071))
    lon      = float(getattr(args, "lon",       -8.210492))
    buf_m    = float(getattr(args, "buffer_m",  500.0))
    src      = getattr(args, "bathy_source",    "all")
    d_min    = float(getattr(args, "depth_min", -50.0))
    d_max    = float(getattr(args, "depth_max", -1.0))
    date_s   = getattr(args, "date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    date_tag = date_s.replace("-", "") if date_s else _date_str()

    log.info("=" * 60)
    log.info("BATHY STEP — source=%s, depth=[%.0f, %.0f] m", src, d_min, d_max)

    # ---- Download ----
    bathy_tifs: list[str] = []
    use_all = (src == "all")

    if use_all or src == "dgt_lidar":
        t = download_dgt_lidar(lat, lon, buf_m, output_dir, date_tag)
        if t:
            bathy_tifs.append(t)

    if use_all or src == "emodnet":
        t = download_emodnet_bathy(lat, lon, buf_m, output_dir, date_tag)
        if t:
            bathy_tifs.append(t)

    if use_all or src == "gebco":
        t = download_gebco(lat, lon, buf_m, output_dir, date_tag)
        if t:
            bathy_tifs.append(t)

    if use_all or src == "geomar":
        t = download_geomar(lat, lon, buf_m, output_dir, date_tag)
        if t:
            bathy_tifs.append(t)

    if use_all or src == "etopo":
        t = download_etopo(lat, lon, buf_m, output_dir, date_tag)
        if t:
            bathy_tifs.append(t)

    if use_all or src == "s2":
        b02 = os.path.join(output_dir, f"S2_B02_{date_tag}.tif")
        b03 = os.path.join(output_dir, f"S2_B03_{date_tag}.tif")
        b08 = os.path.join(output_dir, f"S2_B08_{date_tag}.tif")
        t = compute_s2_depth_inversion(b02, b03, output_dir, b08_path=b08, date_str=date_tag)
        if t:
            bathy_tifs.append(t)

    if not bathy_tifs:
        log.warning("   No bathymetry data was successfully downloaded.")
        log.warning("   Check network access and data source availability.")
        return

    # ---- Process: use highest-resolution (first successful) source ----
    # Priority order: DGT LiDAR > EMODnet > S2 > GEBCO > ETOPO > GEOMAR
    priority = [
        f"bathy_dgt_lidar_{date_tag}.tif",
        f"bathy_emodnet_{date_tag}.tif",
        f"bathy_s2_stumpf_{date_tag}.tif",
        f"bathy_gebco_{date_tag}.tif",
        f"bathy_geomar_{date_tag}.tif",
        f"bathy_etopo_{date_tag}.tif",
    ]
    primary = None
    for pname in priority:
        candidate = os.path.join(output_dir, pname)
        if candidate in bathy_tifs and _validate_tiff(candidate):
            primary = candidate
            break
    if primary is None:
        primary = bathy_tifs[0]

    log.info("   Primary bathymetry for analysis: %s", primary)

    # ---- Compute indices ----
    indices_dir = output_dir
    indices = compute_bathy_indices(primary, indices_dir)

    # ---- Detect candidates ----
    candidates_path = detect_reef_candidates(
        primary, indices_dir, output_dir,
        depth_min=d_min, depth_max=d_max, date_str=date_tag,
    )

    # ---- Update QGIS project ----
    import glob
    qgs_files = sorted(
        glob.glob(os.path.join(output_dir, "reef_project_*.qgs")),
        key=os.path.getmtime,
    )
    if qgs_files:
        qgs_path = qgs_files[-1]
        add_bathy_to_qgis(
            qgs_path, primary, candidates_path,
            indices={k: v for k, v in indices.items()
                     if k in ("slope_deg", "tri", "bpi_broad")},
        )
    else:
        log.info("   No .qgs file found in %s — skipping QGIS injection", output_dir)

    log.info("=" * 60)
    log.info("BATHY STEP complete.")
    log.info("  Primary bathy   : %s", primary)
    log.info("  Indices         : %s", list(indices.keys()))
    log.info("  Reef candidates : %s", candidates_path)


# ===========================================================================
# Standalone CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reef Bathymetry Module — Albufeira Reef (standalone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--step",        required=True, choices=["bathy"],
                   help="Pipeline step (only 'bathy' supported standalone)")
    p.add_argument("--output-dir",  default="reef_output_bathy",
                   help="Output directory")
    p.add_argument("--lat",         type=float, default=37.069071)
    p.add_argument("--lon",         type=float, default=-8.210492)
    p.add_argument("--buffer-m",    type=float, default=500.0,
                   dest="buffer_m", help="AOI radius in metres")
    p.add_argument("--bathy-source",
                   choices=["dgt_lidar", "emodnet", "gebco", "geomar", "etopo", "s2", "all"],
                   default="all", dest="bathy_source",
                   help="Data source(s) to use")
    p.add_argument("--depth-min",   type=float, default=-50.0, dest="depth_min",
                   help="Minimum depth (negative = below sea level, e.g. -50)")
    p.add_argument("--depth-max",   type=float, default=-1.0,  dest="depth_max",
                   help="Maximum depth (e.g. -1)")
    p.add_argument("--date",        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                   help="Target date YYYY-MM-DD (used to locate S2 bands)")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _setup_log(args.output_dir)
    run_bathy_step(args)
