"""
Coastal topography feature extraction from DGT MDT-50cm LiDAR DTM.

This module integrates with the reef imagery pipeline to extract
coastal terrain features (slope, aspect) around dive sites that
correlate with sediment resuspension, swell exposure, and water visibility.

Usage:
    from coastal_topography import CoastalTopographyAnalyzer

    analyzer = CoastalTopographyAnalyzer(
        bbox=(-8.25, 37.04, -8.17, 37.10),
        output_dir="./outputs/coastal_features"
    )

    # Extract features for specific dive sites
    features_df = analyzer.extract_features_for_sites(
        sites=[
            ("pedra_sta_eulalia", 37.069081, -8.210242),
            ("albufeira_reef", 37.0690, -8.2105),
        ],
        buffer_m=1000
    )

    # Save to CSV / GeoJSON
    features_df.to_csv("coastal_features.csv", index=False)

References:
    - DGT MDT-50cm via STAC: https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items
    - Sediment resuspension linked to coastal topography: Mahadevan & Archer (2000)
    - Stumpf et al. (2003) on optical properties of shallow water
"""

import base64
import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.merge import merge

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from rasterstats import zonal_stats

    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False

try:
    from src.dgt_cdd_auth import get_cdd_session, get_signed_url
    from src.dgt_cdd_auth import invalidate as _cdd_invalidate

    HAS_CDD_AUTH = True
except Exception:
    HAS_CDD_AUTH = False

logger = logging.getLogger(__name__)


class CoastalTopographyAnalyzer:
    """Extract terrain features from DGT MDT-50cm or Copernicus GLO-30 around dive sites."""

    # DGT STAC endpoints
    STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"
    MDS_STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDS-50cm/items"

    # Copernicus GLO-30 — public AWS bucket (no auth required, ~30 m resolution)
    GLO30_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

    # CDSE OAuth + download (uses user's Copernicus/CMEMS credentials)
    CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"

    # Credentials file written by copernicus-marine-client (base64-encoded)
    CDSE_CRED_FILE = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"

    # Native CRS of DGT MDT-50cm: ETRS89 / Portugal TM06
    NATIVE_CRS = "EPSG:3763"
    WGS84_CRS = "EPSG:4326"

    # Primary nodata used by most DGT tiles; a minority use -3.4028235e38 (float32 min)
    NODATA_VALUE = -999.0
    NODATA_ALT = -3.4028235e38  # seen in MDT-50cm-193014-04-2024 and similar tiles

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        output_dir: str = "./outputs/coastal_features",
        cache_tiles: bool = True,
        dem_source: str = "auto",
    ) -> None:
        """
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 (lon, lat)
            output_dir: where to save tiles, mosaics, and feature tables
            cache_tiles: if True, reuse downloaded tiles across analyses
            dem_source: one of "dgt" (50cm, requires DGT S3 credentials),
                        "copernicus" (GLO-30 via CDSE, uses ~/.copernicusmarine creds),
                        "srtm" (GLO-30 public AWS, no auth),
                        "auto" (tries dgt → copernicus → srtm)
        """
        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_tiles = cache_tiles
        self.dem_source = dem_source

        self.tiles_dir = self.output_dir / "mdt_tiles"
        self.tiles_dir.mkdir(exist_ok=True)

        self.dem_mosaic_path = self.output_dir / "dem_mosaic_50cm.tif"
        self.mds_mosaic_path = self.output_dir / "mds_mosaic_50cm.tif"
        self.chm_path = self.output_dir / "chm_50cm.tif"
        self.slope_path = self.output_dir / "slope_50cm.tif"
        self.aspect_path = self.output_dir / "aspect_50cm.tif"

        logger.info("CoastalTopographyAnalyzer initialized")
        logger.info(f"  BBox (WGS84): {self.bbox}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  DEM source: {self.dem_source}")

    def fetch_stac_items(self, limit: int = 50) -> list[dict]:
        """
        Query DGT STAC endpoint for MDT-50cm items in the bbox.

        Returns:
            List of STAC features
        """
        params = {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "limit": limit,
            "f": "json",
        }

        logger.info(f"Querying DGT STAC: {self.STAC_URL}")
        return self._fetch_stac_items_from(self.STAC_URL, limit=limit)

    # ── MinIO/S3 streaming (preferred over full-tile download) ────────────────

    def _minio_stream_env(self) -> Optional["rasterio.Env"]:
        """Build a rasterio Env for streaming DGT COGs directly from MinIO/S3.

        Reads MinIO credentials from the same env vars the DGT JupyterHub uses
        (AWS_ENDPOINT_URL2 / AWS_ACCESS_KEY_ID2 / AWS_SECRET_ACCESS_KEY2). Returns
        None when rasterio or the credentials are unavailable, so callers fall back
        to the signed-URL download path. MinIO needs path-style access, hence
        AWS_VIRTUAL_HOSTING=FALSE.
        """
        if not HAS_RASTERIO:
            return None
        endpoint = os.getenv("AWS_ENDPOINT_URL2")
        key = os.getenv("AWS_ACCESS_KEY_ID2")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY2")
        if not (endpoint and key and secret):
            return None
        try:
            import boto3
            from rasterio.session import AWSSession

            host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
            session = boto3.Session(aws_access_key_id=key, aws_secret_access_key=secret)
            return rasterio.Env(
                AWSSession(session),
                AWS_S3_ENDPOINT=host,
                AWS_HTTPS="YES" if endpoint.startswith("https://") else "NO",
                AWS_VIRTUAL_HOSTING="FALSE",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF,.TIFF,.ovr",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            )
        except Exception as e:
            logger.debug("MinIO streaming env unavailable: %s", e)
            return None

    @staticmethod
    def _href_to_vsis3(href: str | None) -> str | None:
        """Convert a MinIO HTTPS/s3 asset href into a /vsis3 GDAL path."""
        if not href:
            return None
        if href.startswith("/vsis3/"):
            return href
        if href.startswith("s3://"):
            return "/vsis3/" + href[len("s3://") :]
        parsed = urlparse(href)
        if parsed.scheme in ("http", "https") and parsed.path:
            return "/vsis3/" + parsed.path.lstrip("/")
        return None

    def _stream_crop_to_local(self, href: str | None, local_path: Path, env: Optional["rasterio.Env"]) -> bool:
        """Stream a windowed crop (self.bbox) of a DGT COG from MinIO to a local GeoTIFF.

        Reads only the pixels overlapping the analysis bbox via GDAL /vsis3 — no full
        tile download. Returns True on success; False signals the caller to fall back
        to the signed-URL/HTTP download. The output keeps the tile's native CRS/grid so
        the existing mosaic + slope/aspect + zonal_stats path is unchanged.
        """
        if env is None:
            return False
        vsi = self._href_to_vsis3(href)
        if vsi is None:
            return False
        try:
            from rasterio.warp import transform_bounds
            from rasterio.windows import Window
            from rasterio.windows import from_bounds as window_from_bounds

            with env, rasterio.open(vsi) as src:
                left, bottom, right, top = transform_bounds(self.WGS84_CRS, src.crs, *self.bbox, densify_pts=21)
                win = window_from_bounds(left, bottom, right, top, src.transform)
                win = win.intersection(Window(0, 0, src.width, src.height))
                win = win.round_offsets().round_lengths()
                if win.width < 1 or win.height < 1:
                    return False  # no overlap (STAC pre-filters, so unexpected)
                data = src.read(1, window=win)
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=int(win.height),
                    width=int(win.width),
                    transform=src.window_transform(win),
                )
            with rasterio.open(local_path, "w", **profile) as dst:
                dst.write(data, 1)
            logger.info("Streamed crop %s (%dx%d px) from MinIO", local_path.name, int(win.width), int(win.height))
            return True
        except Exception as e:
            logger.debug("Stream-crop failed for %s (%s) — falling back to download", href, e)
            return False

    def download_mdt_tiles(self, limit: int = 50) -> list[Path]:
        """
        Acquire MDT-50cm tiles for the bbox: stream windowed crops from MinIO when
        credentials are configured, else download full tiles via the CDD signed URL.

        Args:
            limit: max number of items to request from STAC

        Returns:
            List of local file paths
        """
        features = self.fetch_stac_items(limit=limit)
        if not features:
            logger.warning("No STAC features found; no tiles to download")
            return []

        stream_env = self._minio_stream_env()
        tile_paths = []
        for feat in features:
            item_id = feat.get("id", "unknown")
            asset = feat.get("assets", {}).get("Data", {})
            href = asset.get("href")

            if not href:
                logger.warning(f"Feature {item_id} has no Data.href")
                continue

            fname = Path(href).name
            local_path = self.tiles_dir / fname

            # Check cache
            if local_path.exists():
                logger.debug(f"Tile {item_id} already cached: {local_path}")
                tile_paths.append(local_path)
                continue

            # Preferred: stream a windowed crop directly from MinIO (no full download).
            if self._stream_crop_to_local(href, local_path, stream_env):
                tile_paths.append(local_path)
                continue

            # Fallback: download via CDD signed URL (preferred) or direct href.
            logger.info(f"Downloading {item_id} -> {fname}")
            if HAS_CDD_AUTH:
                download_url = get_signed_url("MDT-50cm", item_id)
                session = get_cdd_session() if download_url else None
            else:
                download_url, session = None, None

            if download_url is None:
                # No credentials or auth failed — try direct href (403 without auth)
                download_url = href
                session = None
                if HAS_CDD_AUTH:
                    logger.warning(
                        "DGT CDD auth unavailable for %s — "
                        "set DGT_CDD_USERNAME / DGT_CDD_PASSWORD env vars. "
                        "Falling back to GLO-30.",
                        fname,
                    )

            try:
                getter = session if session else requests
                with getter.get(download_url, stream=True, timeout=60) as r_tif:
                    if r_tif.status_code in (401, 403):
                        if session:
                            _cdd_invalidate()
                        logger.error(f"HTTP {r_tif.status_code} for {fname} — skipping tile")
                        continue
                    r_tif.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r_tif.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Downloaded {fname} ({local_path.stat().st_size / 1e6:.1f} MB)")
                tile_paths.append(local_path)
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")

        return tile_paths

    # ── MDS-50cm (Digital Surface Model) + Canopy Height Model ────────────────

    def download_mds_tiles(self, limit: int = 50) -> list[Path]:
        """Download MDS-50cm (Digital Surface Model) tiles from DGT STAC.

        MDS includes vegetation canopy and building tops; MDT is bare ground.
        CHM = MDS − MDT gives structure/vegetation height above terrain.
        """
        features = self._fetch_stac_items_from(self.MDS_STAC_URL, limit=limit)
        if not features:
            logger.warning("No MDS-50cm tiles found for bbox")
            return []

        stream_env = self._minio_stream_env()
        tile_paths = []
        for feat in features:
            item_id = feat.get("id", "unknown")
            href = feat.get("assets", {}).get("Data", {}).get("href")
            if not href:
                continue
            local_path = self.tiles_dir / Path(href).name
            if local_path.exists():
                tile_paths.append(local_path)
                continue
            # Preferred: stream a windowed crop directly from MinIO (no full download).
            if self._stream_crop_to_local(href, local_path, stream_env):
                tile_paths.append(local_path)
                continue
            logger.info(f"Downloading MDS tile {item_id}")
            download_url = get_signed_url("MDS-50cm", item_id) if HAS_CDD_AUTH else None
            session = get_cdd_session() if download_url else None
            if download_url is None:
                download_url = href
                session = None
            try:
                getter = session if session else requests
                with getter.get(download_url, stream=True, timeout=60) as r:
                    if r.status_code in (401, 403):
                        if session:
                            _cdd_invalidate()
                        logger.error(f"HTTP {r.status_code} for MDS {item_id} — skipping")
                        continue
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                tile_paths.append(local_path)
            except Exception as exc:
                logger.error(f"Failed to download MDS tile {href}: {exc}")
        return tile_paths

    def build_mds_mosaic(self, tile_paths: list[Path] | None = None) -> Path | None:
        """Merge MDS-50cm tiles into a mosaic GeoTIFF (mirrors build_dem_mosaic)."""
        if not HAS_RASTERIO:
            logger.error("rasterio required for MDS mosaic")
            return None
        if self.mds_mosaic_path.exists() and self.cache_tiles:
            return self.mds_mosaic_path

        if tile_paths is None:
            tile_paths = self.download_mds_tiles()
        if not tile_paths:
            logger.error("No MDS tiles available")
            return None

        # Reuse the same merge logic as the MDT mosaic
        import rioxarray as rxr
        from rasterio.crs import CRS as RioCRS
        from rasterio.merge import merge as _merge

        try:
            with contextlib.ExitStack() as stack:
                src_files = [stack.enter_context(rasterio.open(str(p))) for p in tile_paths]
                mosaic, mosaic_transform = _merge(src_files, nodata=self.NODATA_VALUE)
                mosaic = np.where(mosaic <= self.NODATA_ALT * 0.5, self.NODATA_VALUE, mosaic)
                src_crs = src_files[0].crs
                meta = src_files[0].meta.copy()
                meta.update(
                    height=mosaic.shape[1],
                    width=mosaic.shape[2],
                    transform=mosaic_transform,
                    dtype="float32",
                    nodata=self.NODATA_VALUE,
                )

            tmp = self.tiles_dir / "_mds_mosaic_raw.tif"
            with rasterio.open(str(tmp), "w", **meta) as dst:
                dst.write(mosaic.astype("float32"))

            native_crs = RioCRS.from_epsg(3763)
            if src_crs != native_crs:
                da = rxr.open_rasterio(str(tmp), masked=True)
                da.rio.reproject(self.NATIVE_CRS).rio.to_raster(str(self.mds_mosaic_path), dtype="float32")
                tmp.unlink(missing_ok=True)
            else:
                tmp.rename(self.mds_mosaic_path)

            logger.info(f"MDS mosaic saved: {self.mds_mosaic_path}")
            return self.mds_mosaic_path
        except Exception as exc:
            logger.error(f"MDS mosaic failed: {exc}")
            return None

    def compute_canopy_height(self, mdt_path: Path | None = None, mds_path: Path | None = None) -> Path | None:
        """Compute Canopy Height Model (CHM = MDS − MDT) and save as GeoTIFF.

        For coastal reef sites the CHM represents dune/vegetation height above
        bare ground, which is a proxy for coastal shelter and sediment trapping.
        Values are clamped to [0, 50] m; negatives (LiDAR noise) are set to 0.
        """
        if not HAS_RASTERIO:
            return None

        mdt_path = mdt_path or self.dem_mosaic_path
        mds_path = mds_path or self.mds_mosaic_path

        if not mdt_path.exists():
            logger.error(f"MDT not found: {mdt_path}")
            return None
        if not mds_path.exists():
            logger.error(f"MDS not found: {mds_path}")
            return None

        if self.chm_path.exists() and self.cache_tiles:
            return self.chm_path

        try:
            with rasterio.open(str(mdt_path)) as mdt_src, rasterio.open(str(mds_path)) as mds_src:
                mdt = mdt_src.read(1, masked=True).astype(np.float32)
                mds = mds_src.read(1, masked=True).astype(np.float32)
                profile = mdt_src.profile.copy()

            # Align if grids differ (MDS and MDT should be identical but guard anyway)
            if mdt.shape != mds.shape:
                logger.warning("MDT/MDS grids differ — reprojecting MDS to match MDT")
                import rioxarray as rxr

                mdt_da = rxr.open_rasterio(str(mdt_path), masked=True)
                mds_da = rxr.open_rasterio(str(mds_path), masked=True)
                mds_da = mds_da.rio.reproject_match(mdt_da)
                mds = mds_da.values[0].astype(np.float32)

            nodata_mask = (mdt == self.NODATA_VALUE) | (mds == self.NODATA_VALUE)
            chm = np.where(nodata_mask, np.nan, np.clip(mds - mdt, 0.0, 50.0))

            profile.update(dtype="float32", nodata=np.nan)
            with rasterio.open(str(self.chm_path), "w", **profile) as dst:
                dst.write(chm.astype("float32"), 1)

            valid = chm[~np.isnan(chm)]
            logger.info(f"CHM saved: {self.chm_path}  " f"mean={np.mean(valid):.2f}m  max={np.max(valid):.2f}m")
            return self.chm_path
        except Exception as exc:
            logger.error(f"CHM computation failed: {exc}")
            return None

    def _fetch_stac_items_from(self, url: str, limit: int = 50) -> list[dict]:
        """Generic STAC item fetch from any DGT collection URL."""
        params = {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "limit": limit,
            "f": "json",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            fc = r.json()
            n = fc.get("context", {}).get("returned", 0)
            logger.info(f"STAC {url.split('/')[-2]} returned {n} features")
            return fc.get("features", [])
        except requests.RequestException as exc:
            logger.error(f"STAC query failed ({url}): {exc}")
            return []

    # ── Copernicus GLO-30 helpers ──────────────────────────────────────────────

    @staticmethod
    def _read_cdse_credentials() -> tuple[str, str]:
        """Read username/password from ~/.copernicusmarine/.copernicusmarine-credentials."""
        cred_file = CoastalTopographyAnalyzer.CDSE_CRED_FILE
        if not cred_file.exists():
            raise FileNotFoundError(f"CDSE credentials not found: {cred_file}")
        raw = base64.b64decode(cred_file.read_text().strip().rstrip("%")).decode()
        creds: dict[str, str] = {}
        for line in raw.splitlines():
            if "=" in line and not line.startswith("["):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
        try:
            return creds["username"], creds["password"]
        except KeyError as exc:
            raise KeyError(
                f"CDSE credentials file {cred_file} is missing key {exc}. " "Expected 'username' and 'password' entries."
            ) from exc

    def _get_cdse_token(self) -> str:
        """Obtain a short-lived OAuth2 bearer token from CDSE."""
        user, pwd = self._read_cdse_credentials()
        resp = requests.post(
            self.CDSE_TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "username": user,
                "password": pwd,
                "grant_type": "password",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    @staticmethod
    def _glo30_tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
        """Return list of (lat_floor, lon_floor) pairs covering bbox for 1°×1° GLO-30 tiles."""
        minx, miny, maxx, maxy = bbox
        tiles = []
        for lat in range(int(np.floor(miny)), int(np.floor(maxy)) + 1):
            for lon in range(int(np.floor(minx)), int(np.floor(maxx)) + 1):
                tiles.append((lat, lon))
        return tiles

    @staticmethod
    def _glo30_tile_name(lat: int, lon: int) -> str:
        """Return the Copernicus GLO-30 tile stem for a given (lat_floor, lon_floor)."""
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"Copernicus_DSM_COG_10_{lat_dir}{abs(lat):02d}_00" f"_{lon_dir}{abs(lon):03d}_00_DEM"

    def download_glo30_public(self) -> list[Path]:
        """
        Download Copernicus GLO-30 tiles from the public AWS S3 bucket (no auth).

        Resolution: ~30 m.  Covers the world.  Tiles are 1°×1°.
        """
        if not HAS_RASTERIO:
            logger.error("rasterio required for GLO-30 download")
            return []

        tiles = self._glo30_tiles_for_bbox(self.bbox)
        logger.info(f"GLO-30 public: {len(tiles)} tile(s) needed for bbox")

        paths = []
        for lat, lon in tiles:
            stem = self._glo30_tile_name(lat, lon)
            url = f"{self.GLO30_BASE_URL}/{stem}/{stem}.tif"
            local = self.tiles_dir / f"{stem}.tif"

            if local.exists() and self.cache_tiles:
                logger.info(f"  Cached: {local.name}")
                paths.append(local)
                continue

            logger.info(f"  Downloading GLO-30 tile: {stem}")
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                logger.info(f"  Saved {local.name} ({local.stat().st_size / 1e6:.1f} MB)")
                paths.append(local)
            except Exception as exc:
                logger.error(f"  Failed to download {url}: {exc}")

        return paths

    def download_copernicus_cdse(self) -> list[Path]:
        """
        Download Copernicus DEM GLO-30 tiles via CDSE (authenticated).

        Uses ~/.copernicusmarine/.copernicusmarine-credentials for OAuth2.
        Falls back to public AWS bucket if a tile is not found on CDSE.
        """
        try:
            token = self._get_cdse_token()
            headers = {"Authorization": f"Bearer {token}"}
        except Exception as exc:
            logger.warning(f"CDSE auth failed ({exc}); falling back to public GLO-30")
            return self.download_glo30_public()

        tiles = self._glo30_tiles_for_bbox(self.bbox)
        logger.info(f"GLO-30 via CDSE: {len(tiles)} tile(s) needed")

        paths = []
        for lat, lon in tiles:
            stem = self._glo30_tile_name(lat, lon)
            local = self.tiles_dir / f"{stem}.tif"

            if local.exists() and self.cache_tiles:
                logger.info(f"  Cached: {local.name}")
                paths.append(local)
                continue

            # Search CDSE OData for this tile
            product_name = f"{stem}__30"
            search_url = (
                f"{self.CDSE_CATALOGUE_URL}/Products"
                f"?$filter=Collection/Name eq 'COP-DEM'"
                f" and contains(Name,'{stem}')"
                f"&$top=1"
            )
            try:
                sr = requests.get(search_url, headers=headers, timeout=30)
                sr.raise_for_status()
                products = sr.json().get("value", [])
            except Exception as exc:
                logger.warning(f"  CDSE search failed for {stem}: {exc}")
                products = []

            if products:
                pid = products[0]["Id"]
                dl_url = f"{self.CDSE_DOWNLOAD_URL}/Products({pid})/$value"
                logger.info(f"  Downloading {stem} from CDSE (id={pid})")
                try:
                    with requests.get(dl_url, headers=headers, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(local, "wb") as fh:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                if chunk:
                                    fh.write(chunk)
                    logger.info(f"  Saved {local.name} ({local.stat().st_size / 1e6:.1f} MB)")
                    paths.append(local)
                    continue
                except Exception as exc:
                    logger.warning(f"  CDSE download failed for {stem}: {exc}; trying public bucket")

            # Fallback to public AWS for this tile
            fallback = self.download_glo30_public()
            paths.extend(p for p in fallback if p not in paths)

        return paths

    def build_dem_mosaic(self, tile_paths: list[Path] | None = None) -> Path | None:
        """
        Merge MDT-50cm tiles into a single DEM GeoTIFF.

        If mosaic already exists and cache_tiles=True, skips rebuild.

        Args:
            tile_paths: list of local tile paths. If None, calls download_mdt_tiles()

        Returns:
            Path to mosaic GeoTIFF, or None on failure
        """
        if not HAS_RASTERIO:
            logger.error("rasterio not installed; cannot build mosaic")
            return None

        # Check cache
        if self.dem_mosaic_path.exists() and self.cache_tiles:
            logger.info(f"DEM mosaic already cached: {self.dem_mosaic_path}")
            return self.dem_mosaic_path

        # Download tiles if not provided — dispatch based on dem_source
        if tile_paths is None:
            source = self.dem_source
            if source == "dgt":
                tile_paths = self.download_mdt_tiles()
            elif source == "copernicus":
                tile_paths = self.download_copernicus_cdse()
            elif source == "srtm":
                tile_paths = self.download_glo30_public()
            else:  # "auto": DGT first, then CDSE, then public GLO-30
                tile_paths = self.download_mdt_tiles()
                if not tile_paths:
                    logger.info("DGT tiles unavailable; trying Copernicus CDSE...")
                    tile_paths = self.download_copernicus_cdse()
                if not tile_paths:
                    logger.info("CDSE unavailable; falling back to public GLO-30...")
                    tile_paths = self.download_glo30_public()

        if not tile_paths:
            logger.error("No tiles available for mosaicing")
            return None

        logger.info(f"Building mosaic from {len(tile_paths)} tiles...")

        try:
            import rioxarray as rxr
            from rasterio.crs import CRS as RioCRS

            with contextlib.ExitStack() as stack:
                src_files = [stack.enter_context(rasterio.open(str(p))) for p in tile_paths]

                # Normalise nodata: some tiles use -3.4e38 instead of -999.0.
                # Passing nodata= to merge ensures uninitialized output cells get -999.0
                # (not 0.0) and all source nodata values are treated consistently.
                mosaic, mosaic_transform = merge(src_files, nodata=self.NODATA_VALUE)
                # Recode any surviving -3.4e38 fill (from tiles whose rasterio metadata
                # declared NODATA_ALT) so the mosaic is uniform.
                mosaic = np.where(mosaic <= self.NODATA_ALT * 0.5, self.NODATA_VALUE, mosaic)
                src_crs = src_files[0].crs
                meta = src_files[0].meta.copy()
                meta.update(
                    {
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": mosaic_transform,
                        "dtype": "float32",
                        "nodata": self.NODATA_VALUE,
                    }
                )

            # Clip merged array to bbox + 5% margin before writing (reduces memory ~90%)
            from rasterio.transform import array_bounds
            from rasterio.windows import from_bounds as _from_bounds

            margin = 0.05  # degrees
            clip_minx = self.bbox[0] - margin
            clip_miny = self.bbox[1] - margin
            clip_maxx = self.bbox[2] + margin
            clip_maxy = self.bbox[3] + margin

            if src_crs and str(src_crs).upper() in ("EPSG:4326", "WGS 84"):
                # Tile is geographic — clip directly by bbox
                win = _from_bounds(clip_minx, clip_miny, clip_maxx, clip_maxy, mosaic_transform)
                row_start = max(0, int(win.row_off))
                row_stop = min(mosaic.shape[1], int(win.row_off + win.height) + 1)
                col_start = max(0, int(win.col_off))
                col_stop = min(mosaic.shape[2], int(win.col_off + win.width) + 1)
                mosaic = mosaic[:, row_start:row_stop, col_start:col_stop]
                h, w = mosaic.shape[1], mosaic.shape[2]
                arr_b = array_bounds(h, w, mosaic_transform)
                # Recompute transform for clipped window
                from affine import Affine

                mosaic_transform = Affine(
                    mosaic_transform.a,
                    mosaic_transform.b,
                    mosaic_transform.c + col_start * mosaic_transform.a,
                    mosaic_transform.d,
                    mosaic_transform.e,
                    mosaic_transform.f + row_start * mosaic_transform.e,
                )
                meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform)
                logger.info(f"  Clipped to bbox: {mosaic.shape[1]}×{mosaic.shape[2]} px")

            # Write clipped mosaic to a temp file, then reproject to EPSG:3763 if needed
            tmp_path = self.tiles_dir / "_mosaic_raw.tif"
            with rasterio.open(str(tmp_path), "w", **meta) as dst:
                dst.write(mosaic.astype("float32"))

            native_crs = RioCRS.from_epsg(3763)
            if src_crs != native_crs:
                logger.info(f"  Reprojecting mosaic from {src_crs} → EPSG:3763 ...")
                da = rxr.open_rasterio(str(tmp_path), masked=True)
                da_reproj = da.rio.reproject(self.NATIVE_CRS)
                da_reproj.rio.to_raster(str(self.dem_mosaic_path), dtype="float32")
                tmp_path.unlink(missing_ok=True)
                logger.info("  Reprojection complete")
            else:
                tmp_path.rename(self.dem_mosaic_path)

            with rasterio.open(str(self.dem_mosaic_path)) as chk:
                logger.info(f"Mosaic saved: {self.dem_mosaic_path}")
                logger.info(f"  Shape: {chk.shape}, CRS: {chk.crs}, res: {chk.res[0]:.1f} m")

            return self.dem_mosaic_path

        except Exception as e:
            logger.error(f"Mosaic creation failed: {e}")
            return None

    def derive_slope_aspect(self, dem_path: Path | None = None) -> tuple[Path | None, Path | None]:
        """
        Compute slope and aspect from DEM using gradient method.

        Slope is in degrees; aspect is in degrees (0=N, 90=E, 180=S, 270=W).

        Args:
            dem_path: path to DEM GeoTIFF. If None, uses self.dem_mosaic_path

        Returns:
            (slope_path, aspect_path) or (None, None) on failure
        """
        if not HAS_RASTERIO:
            logger.error("rasterio not installed")
            return None, None

        if dem_path is None:
            dem_path = self.dem_mosaic_path

        if not dem_path.exists():
            logger.error(f"DEM not found: {dem_path}")
            return None, None

        # Check cache
        if self.slope_path.exists() and self.aspect_path.exists() and self.cache_tiles:
            logger.info("Slope/aspect already cached")
            return self.slope_path, self.aspect_path

        logger.info(f"Computing slope and aspect from {dem_path}...")

        try:
            with rasterio.open(str(dem_path)) as src:
                dem = src.read(1, masked=True)
                transform = src.transform
                profile = src.profile

                # Resolution in meters
                res_x = abs(transform.a)
                res_y = abs(transform.e)

                logger.info(f"  DEM resolution: {res_x:.1f} m x {res_y:.1f} m")

            # Compute gradients (dz/dx, dz/dy in meters per meter)
            # np.gradient returns dy, dx
            dz_dy, dz_dx = np.gradient(dem.filled(np.nan), res_y, res_x)

            # Slope in degrees
            slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
            slope = np.degrees(slope_rad)

            # Aspect in degrees (0=N, 90=E, 180=S, 270=W)
            # Note: arctan2(x, -y) converts to geographic convention
            aspect_rad = np.arctan2(dz_dx, -dz_dy)
            aspect = np.degrees(aspect_rad)
            aspect = np.where(aspect < 0, 360 + aspect, aspect)

            # Save slope
            profile_out = profile.copy()
            profile_out.update(dtype="float32", nodata=np.nan)

            with rasterio.open(str(self.slope_path), "w", **profile_out) as dst:
                dst.write(slope.astype("float32"), 1)
            logger.info(f"Slope saved: {self.slope_path}")

            # Save aspect
            with rasterio.open(str(self.aspect_path), "w", **profile_out) as dst:
                dst.write(aspect.astype("float32"), 1)
            logger.info(f"Aspect saved: {self.aspect_path}")

            return self.slope_path, self.aspect_path

        except Exception as e:
            logger.error(f"Slope/aspect derivation failed: {e}")
            return None, None

    def extract_features_for_sites(
        self,
        sites: list[tuple[str, float, float]],
        buffer_m: float = 1000,
        stats: list[str] | None = None,
        include_chm: bool = True,
    ) -> pd.DataFrame | None:
        """
        Extract terrain features (slope, aspect) for dive sites.

        Args:
            sites: list of (site_name, lat, lon) tuples
            buffer_m: circular buffer radius around each site (meters)
            stats: zonal statistics to compute. Default: ["mean", "median", "std", "min", "max", "percentile_90"]

        Returns:
            GeoDataFrame with features, or None on failure
        """
        if not HAS_RASTERSTATS:
            logger.error("rasterstats not installed; cannot compute zonal stats")
            logger.info("Install with: pip install rasterstats")
            return None

        if stats is None:
            stats = ["mean", "median", "std", "min", "max", "percentile_90"]

        # Build DEM and derive slope/aspect if needed
        if not self.dem_mosaic_path.exists():
            mosaic = self.build_dem_mosaic()
            if mosaic is None:
                return None

        if not self.slope_path.exists() or not self.aspect_path.exists():
            slope_path, aspect_path = self.derive_slope_aspect()
            if slope_path is None or aspect_path is None:
                return None

        if not HAS_GEOPANDAS:
            logger.error("geopandas not installed; cannot extract site features")
            logger.info("Install with: pip install geopandas")
            return None

        logger.info(f"Extracting features for {len(sites)} sites with buffer={buffer_m}m...")

        # Convert sites to GeoDataFrame
        points = [Point(lon, lat) for _, lat, lon in sites]
        site_names = [name for name, _, _ in sites]

        gdf = gpd.GeoDataFrame({"site_name": site_names}, geometry=points, crs=self.WGS84_CRS)

        # Reproject to native CRS
        gdf_3763 = gdf.to_crs(self.NATIVE_CRS)
        gdf_3763["buffer_geometry"] = gdf_3763.geometry.buffer(buffer_m)

        # Extract zonal statistics
        features_list = []

        try:
            for idx, row in gdf_3763.iterrows():
                site_name = row["site_name"]
                buffer_geom = row["buffer_geometry"]

                logger.debug(f"Processing {site_name}...")

                # Slope/aspect rasters are written with nodata=NaN (not NODATA_VALUE)
                slope_stats = zonal_stats([buffer_geom], str(self.slope_path), stats=stats, nodata=np.nan)[0]

                aspect_stats = zonal_stats(
                    [buffer_geom], str(self.aspect_path), stats=["mean", "median", "std", "min", "max"], nodata=np.nan
                )[0]

                # Build row
                feature_row = {
                    "site_name": site_name,
                    "latitude": gdf.loc[idx, "geometry"].y,
                    "longitude": gdf.loc[idx, "geometry"].x,
                    "buffer_m": buffer_m,
                }

                # Add slope stats with prefix
                for key, val in slope_stats.items():
                    feature_row[f"slope_{key}"] = val

                # Add aspect stats with prefix
                for key, val in aspect_stats.items():
                    feature_row[f"aspect_{key}"] = val

                # CHM (Canopy Height Model) stats — skipped silently if not built
                if include_chm and HAS_RASTERSTATS and self.chm_path.exists():
                    chm_stats = zonal_stats(
                        [buffer_geom],
                        str(self.chm_path),
                        stats=["mean", "median", "std", "max", "percentile_90"],
                        nodata=np.nan,
                    )[0]
                    for key, val in chm_stats.items():
                        feature_row[f"chm_{key}"] = val

                features_list.append(feature_row)

        except Exception as e:
            logger.error(f"Zonal stats extraction failed: {e}")
            return None

        # Convert to DataFrame
        result_df = pd.DataFrame(features_list)
        result_df["timestamp"] = datetime.now().isoformat()

        logger.info(f"Extracted features for {len(result_df)} sites")

        return result_df

    def save_features(self, features_df: pd.DataFrame, output_name: str = "coastal_features") -> dict[str, str]:
        """
        Save features to CSV and GeoJSON.

        Args:
            features_df: DataFrame with site features
            output_name: base name for output files

        Returns:
            dict with paths to saved files
        """
        csv_path = self.output_dir / f"{output_name}.csv"
        json_path = self.output_dir / f"{output_name}.json"
        geojson_path = self.output_dir / f"{output_name}.geojson"

        # CSV
        features_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # JSON (full dump)
        with open(json_path, "w") as f:
            json.dump(features_df.to_dict(orient="records"), f, indent=2, default=str)
        logger.info(f"Saved JSON: {json_path}")

        saved = {"csv": str(csv_path), "json": str(json_path)}

        # GeoJSON (if geometry available)
        if not HAS_GEOPANDAS:
            logger.warning("geopandas not installed; skipping GeoJSON export")
            return saved

        if "geometry" not in features_df.columns:
            gdf = gpd.GeoDataFrame(
                features_df, geometry=gpd.points_from_xy(features_df["longitude"], features_df["latitude"]), crs=self.WGS84_CRS
            )
        else:
            gdf = gpd.GeoDataFrame(features_df, crs=self.WGS84_CRS)

        gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info(f"Saved GeoJSON: {geojson_path}")

        saved["geojson"] = str(geojson_path)
        return saved

    def run_analysis(
        self,
        sites: list[tuple[str, float, float]],
        buffer_m: float = 1000,
        output_name: str = "coastal_features",
        include_chm: bool = True,
    ) -> dict:
        """
        Full pipeline: download MDT+MDS tiles → mosaics → CHM → slope/aspect → features → save.

        Args:
            sites: list of (site_name, lat, lon) tuples
            buffer_m: buffer radius around each site (meters)
            output_name: base name for output files
            include_chm: if True, attempt to download MDS-50cm and compute the
                         Canopy Height Model (CHM = MDS − MDT).  CHM stats are
                         added as chm_* columns.  Failure is non-fatal.

        Returns:
            dict with status and file paths
        """
        logger.info("=" * 70)
        logger.info("COASTAL TOPOGRAPHY ANALYSIS PIPELINE")
        logger.info("=" * 70)

        try:
            # Step 1+2: MDT mosaic (triggers download for chosen dem_source)
            logger.info(f"\n[1/5] Acquiring MDT tiles (source={self.dem_source})...")
            mosaic_path = self.build_dem_mosaic()
            if mosaic_path is None:
                return {"status": "error", "message": "MDT mosaic creation failed"}

            # Step 3: MDS mosaic + CHM (best-effort — DGT source only)
            chm_path = None
            if include_chm and self.dem_source in ("dgt", "auto"):
                logger.info("\n[2/5] Building MDS mosaic + Canopy Height Model...")
                mds_path = self.build_mds_mosaic()
                if mds_path is not None:
                    chm_path = self.compute_canopy_height()
                    if chm_path is None:
                        logger.warning("CHM computation failed; continuing without it")
                else:
                    logger.warning("MDS tiles unavailable; skipping CHM")
            else:
                logger.info("\n[2/5] Skipping MDS/CHM (dem_source != dgt/auto)")

            # Step 4: Slope/aspect
            logger.info("\n[3/5] Deriving slope and aspect...")
            slope_path, aspect_path = self.derive_slope_aspect(mosaic_path)
            if slope_path is None or aspect_path is None:
                return {"status": "error", "message": "Slope/aspect derivation failed"}

            # Step 5: Feature extraction
            logger.info("\n[4/5] Extracting features for dive sites...")
            features_df = self.extract_features_for_sites(sites, buffer_m=buffer_m, include_chm=include_chm)
            if features_df is None:
                return {"status": "error", "message": "Feature extraction failed"}

            # Step 6: Save
            logger.info("\n[5/5] Saving results...")
            file_paths = self.save_features(features_df, output_name=output_name)

            logger.info("=" * 70)
            logger.info("ANALYSIS COMPLETE")
            logger.info("=" * 70)

            result: dict = {
                "status": "success",
                "sites_analyzed": len(features_df),
                "dem_mosaic": str(mosaic_path),
                "slope_raster": str(slope_path),
                "aspect_raster": str(aspect_path),
                "output_files": file_paths,
                "features_shape": features_df.shape,
                "features_columns": list(features_df.columns),
            }
            if chm_path:
                result["chm_raster"] = str(chm_path)
            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


def main() -> None:
    """Example usage with Algarve survey sites."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Survey sites from algarve_reef_survey.py
    survey_sites = [
        ("tavira_west", 37.1050, -7.6800),
        ("ilha_de_tavira", 37.0950, -7.7200),
        ("fuseta", 37.0600, -7.7600),
        ("olhao_offshore", 37.0350, -7.8200),
        ("faro_east", 37.0100, -7.8800),
        ("praia_de_faro", 36.9800, -7.9400),
        ("ancão_peninsula", 36.9650, -8.0000),
        ("quarteira", 37.0650, -8.1000),
        ("vilamoura", 37.0750, -8.1300),
        ("olhos_de_agua", 37.0900, -8.1600),
        ("pedra_sta_eulalia", 37.069081, -8.210242),
        ("albufeira_reef", 37.0690, -8.2105),
        ("galé", 37.0560, -8.2296),
        ("salgados", 37.0950, -8.3000),
        ("armacao_de_pera", 37.0700, -8.3600),
    ]

    # BBox covering all sites (with margin)
    lats = [s[1] for s in survey_sites]
    lons = [s[2] for s in survey_sites]
    bbox = (min(lons) - 0.05, min(lats) - 0.05, max(lons) + 0.05, max(lats) + 0.05)

    analyzer = CoastalTopographyAnalyzer(
        bbox=bbox,
        output_dir="./outputs/coastal_topography",
        dem_source="dgt",  # 50 cm DGT LiDAR (covers full Algarve); needs DGT_CDD_* creds
    )

    result = analyzer.run_analysis(sites=survey_sites, buffer_m=4000, output_name="algarve_coastal_features")

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(json.dumps(result, indent=2))


def _run_selftest() -> int:
    """Validate the live /vsis3 MinIO streaming path end-to-end on one Algarve tile.

    Streams a windowed crop of a known Albufeira MDT-50cm tile straight from MinIO.
    Returns a process exit code: 0 on success OR a graceful no-creds/fallback path,
    1 only on an unexpected crash. Runs cleanly in CI (no creds → message, exit 0).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Hardcoded Albufeira / Sta Eulália MDT-50cm tile (DGT 2024 campaign). Only the
    # bucket/key PATH of this href is used by _href_to_vsis3; the MinIO host comes
    # from AWS_ENDPOINT_URL2 in the streaming Env.
    TILE_HREF = "https://stor-002.a.acnca.pt:9000/lidar/MDT50cm/" "MDT-50cm-191013-04-2024_v01.tif"
    TILE_BBOX = (-8.234, 37.074, -8.222, 37.083)  # WGS84, overlaps the tile footprint
    # Dev/test override: COASTAL_SELFTEST_OUT redirects the throwaway file.
    # Default (CI, production) keeps the global /tmp path unchanged.
    OUT_PATH = Path(os.environ.get("COASTAL_SELFTEST_OUT", "/tmp/selftest_crop.tif"))

    analyzer = CoastalTopographyAnalyzer(
        bbox=TILE_BBOX,
        output_dir="/tmp/coastal_selftest",
        dem_source="dgt",
    )

    env = analyzer._minio_stream_env()
    if env is None:
        print(
            "No MinIO credentials found; streaming inactive "
            "(set AWS_ENDPOINT_URL2 / AWS_ACCESS_KEY_ID2 / AWS_SECRET_ACCESS_KEY2, "
            "e.g. `set -a; source .env; set +a`). Pipeline falls back to CDD download."
        )
        return 0

    try:
        if OUT_PATH.exists():
            OUT_PATH.unlink()
        ok = analyzer._stream_crop_to_local(TILE_HREF, OUT_PATH, env)
        if ok and OUT_PATH.exists():
            with rasterio.open(OUT_PATH) as src:
                w, h, crs = src.width, src.height, src.crs
            print(f"Streamed crop from MinIO: {OUT_PATH} ({w}x{h} px, {crs})")
            return 0
        # Handled streaming failure (network/auth) — _stream_crop_to_local returned
        # False without raising. The real pipeline would fall back to CDD download.
        print("Streaming failed: no crop produced; " "pipeline would fall back to CDD signed-URL download.")
        return 0
    except Exception as e:
        print(f"Streaming failed: {e}; " "pipeline would fall back to CDD signed-URL download.")
        return 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="DGT coastal topography analyzer (MDT/MDS-50cm LiDAR).")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Validate the live /vsis3 MinIO streaming path on one Algarve tile, then exit.",
    )
    args = parser.parse_args()

    if args.selftest:
        sys.exit(_run_selftest())
    main()
