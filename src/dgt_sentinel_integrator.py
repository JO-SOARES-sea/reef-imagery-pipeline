"""
DGT MDT-50cm + Sentinel-2 integrator for reef imagery analysis.

Consumes:
  - MDT-50cm (50 cm LiDAR DTM) from DGT STAC endpoint
  - Sentinel-2 MSI from Copernicus (via sentinelhub-py or rasterio)

Outputs:
  - Aligned GeoTIFF mosaic (MDT + Sentinel bands) in EPSG:3763
  - Metadata JSON with crs, bounds, resolution, source info
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

try:
    import rioxarray as rxr
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

logger = logging.getLogger(__name__)


class DGTSentinelIntegrator:
    """Download and align MDT-50cm (DGT) + Sentinel-2 for reef imagery."""

    # DGT STAC endpoints
    DGT_STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"
    DGT_MDS_STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDS-50cm/items"
    # DGT hosts annual Sentinel-2 L2A mosaics (MosaicoS2-YYYY) — useful when
    # CDSE/Planetary Computer is unavailable.
    DGT_S2_STAC_BASE = "https://dgt-be.a.incd.pt:8081/collections/MosaicoS2-{year}/items"
    DGT_S2_YEARS = list(range(2015, 2026))  # 2015 – 2025

    # Primary nodata; a minority of DGT tiles use float32-min (-3.4e38) instead
    MDT_NODATA = -999.0
    MDT_NODATA_ALT = -3.4028235e38

    # Target CRS: ETRS89 / Portugal TM06 (used by DGT MDT-50cm)
    TARGET_CRS = "EPSG:3763"

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        output_dir: str = "./data/dgt_sentinel_integrated",
        mdt_nodata: float = -999.0,
    ):
        """
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 (lon, lat)
            output_dir: where to save tiles and mosaics
            mdt_nodata: nodata value for MDT-50cm
        """
        if not HAS_XARRAY:
            raise ImportError(
                "rioxarray/xarray are required for DGTSentinelIntegrator. Install with: pip install rioxarray xarray"
            )

        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mdt_nodata = mdt_nodata

        self.mdt_dir = self.output_dir / "mdt50cm_tiles"
        self.mdt_dir.mkdir(exist_ok=True)

        logger.info(f"DGT-Sentinel Integrator initialized. Output: {self.output_dir}")
        logger.info(f"BBox (WGS84): {self.bbox}")

    def download_mdt50cm_tiles(self, limit: int = 50) -> list[str]:
        """
        Download MDT-50cm tiles from DGT STAC for the given bbox.

        Args:
            limit: max number of items to request

        Returns:
            List of local file paths (GeoTIFFs)
        """
        params = {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "limit": limit,
            "f": "json",
        }

        logger.info(f"Querying DGT STAC: {self.DGT_STAC_URL}")
        r = requests.get(self.DGT_STAC_URL, params=params, timeout=30)
        r.raise_for_status()

        fc = r.json()
        n_returned = fc.get("context", {}).get("returned", 0)
        logger.info(f"DGT STAC returned {n_returned} features")

        if n_returned == 0:
            logger.warning("No MDT-50cm tiles found for this bbox!")
            return []

        tif_paths = []
        for feat in fc.get("features", []):
            item_id = feat.get("id", "unknown")
            asset = feat.get("assets", {}).get("Data", {})
            href = asset.get("href")

            if not href:
                logger.warning(f"Feature {item_id} has no Data.href, skipping")
                continue

            fname = os.path.basename(href)
            local_path = self.mdt_dir / fname

            if local_path.exists():
                logger.info(f"Tile {item_id} already exists: {local_path}")
                tif_paths.append(str(local_path))
                continue

            logger.info(f"Downloading {item_id} -> {local_path}")
            try:
                with requests.get(href, stream=True, timeout=60) as r_tif:
                    r_tif.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r_tif.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Successfully downloaded {fname}")
                tif_paths.append(str(local_path))
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")

        return tif_paths

    def mosaic_mdt50cm(self, tif_paths: list[str]) -> xr.DataArray:
        """Load and mosaic MDT-50cm tiles into a single aligned DataArray.

        Uses rasterio.merge (not xr.concat) so that per-tile nodata values are
        correctly masked and the output fill is uniform (-999.0).  Tiles that
        declare nodata=-3.4e38 (float32-min) are also recoded to -999.0 so
        downstream consumers see a consistent nodata value.
        """
        import rasterio
        from rasterio.merge import merge as _rio_merge

        if not tif_paths:
            raise ValueError("No tiles provided for mosaicing")

        logger.info(f"Mosaicing {len(tif_paths)} MDT-50cm tiles via rasterio.merge...")

        good_paths = []
        for p in tif_paths:
            try:
                with rasterio.open(p):
                    pass  # validate the file opens cleanly
                good_paths.append(p)
            except Exception as exc:
                logger.error(f"Cannot open {p}: {exc}")

        if not good_paths:
            raise ValueError("Could not open any tile files")

        with contextlib.ExitStack() as stack:
            src_files = [stack.enter_context(rasterio.open(p)) for p in good_paths]

            # merge() uses each source's own nodata for masking; nodata= sets the
            # output fill so uninitialized pixels become -999.0 (not 0.0).
            mosaic_arr, mosaic_transform = _rio_merge(src_files, nodata=self.MDT_NODATA)
            # Recode any surviving float32-min values from legacy tiles
            mosaic_arr = np.where(mosaic_arr <= self.MDT_NODATA_ALT * 0.5, self.MDT_NODATA, mosaic_arr)

            src_crs = src_files[0].crs
            meta = src_files[0].meta.copy()
            meta.update(
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                transform=mosaic_transform,
                dtype="float32",
                nodata=self.MDT_NODATA,
            )

        # Write to a MemoryFile so we can hand an xarray DataArray back
        from rasterio.io import MemoryFile as _MemFile

        with _MemFile() as mem:
            with mem.open(**meta) as ds:
                ds.write(mosaic_arr.astype("float32"))
            da = rxr.open_rasterio(mem.name, masked=True)
            if da.sizes.get("band", 1) == 1:
                da = da.squeeze("band")

        # Reproject to EPSG:3763 if tiles came in a different CRS
        if str(src_crs) != "EPSG:3763":
            logger.info(f"Reprojecting mosaic from {src_crs} → EPSG:3763")
            da = da.rio.reproject(self.TARGET_CRS)

        logger.info(f"MDT mosaic shape: {da.shape}, CRS: {da.rio.crs}")
        return da

    def find_dgt_s2_items(self, year: int | None = None, max_cloud_cover: float = 20.0) -> list[dict]:
        """Query DGT STAC for Sentinel-2 L2A items over the bbox.

        DGT hosts annual Sentinel-2 mosaics (MosaicoS2-2015 … MosaicoS2-2025)
        as a fallback when CDSE/Planetary Computer is unavailable.

        Args:
            year: specific year to search (None → tries current year then walks back)
            max_cloud_cover: filter items above this cloud percentage

        Returns:
            list of STAC feature dicts, sorted by cloud cover ascending
        """
        from datetime import datetime as _dt

        years_to_try = [year] if year else list(range(_dt.now().year, 2014, -1))
        bbox_str = f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}"

        for yr in years_to_try:
            url = self.DGT_S2_STAC_BASE.format(year=yr)
            try:
                r = requests.get(url, params={"bbox": bbox_str, "limit": 50, "f": "json"}, timeout=20)
                r.raise_for_status()
                features = r.json().get("features", [])
            except Exception as exc:
                logger.debug(f"DGT S2 {yr}: {exc}")
                continue

            # Filter by cloud cover (property name varies)
            filtered = []
            for f in features:
                props = f.get("properties", {})
                cc = props.get("eo:cloud_cover")
                if cc is None:
                    cc = props.get("s2:cloud_cover")
                if cc is None:
                    cc = 0
                if cc <= max_cloud_cover:
                    filtered.append(f)

            if filtered:
                logger.info(f"DGT S2 {yr}: {len(filtered)} items ≤{max_cloud_cover}% cloud")
                return sorted(filtered, key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))

        logger.warning("No DGT S2 items found for bbox/cloud criteria")
        return []

    def fetch_sentinel2_copernicus(
        self, date_start: str = "2024-01-01", date_end: str = "2024-12-31", cloud_pct: int = 30
    ) -> xr.DataArray | None:
        """
        Fetch Sentinel-2 L2A from Copernicus via sentinelhub-py or Copernicus Browser.

        For this example, we recommend using sentinelhub-py with a valid Copernicus account.

        Args:
            date_start: ISO date string
            date_end: ISO date string
            cloud_pct: max cloud coverage %

        Returns:
            xarray DataArray with Sentinel-2 bands (B02, B03, B04, B08, etc.) in EPSG:3763
        """
        # This is a template; sentinelhub-py setup requires credentials
        logger.info(
            f"Sentinel-2 fetch (template): bbox={self.bbox}, " f"date_range=[{date_start}, {date_end}], cloud_max={cloud_pct}%"
        )

        try:
            from sentinelhub import CRS, BBox, DataCollection, MimeType, SentinelHubRequest
        except ImportError:
            logger.warning(
                "sentinelhub not installed. Install with: pip install sentinelhub\n"
                "Alternatively, download Sentinel-2 L2A from Copernicus Browser manually."
            )
            return None

        logger.info("Using sentinelhub-py to fetch Sentinel-2 L2A...")

        # BBox in WGS84
        bbox_wgs84 = BBox(self.bbox, CRS.WGS84)

        # Request all standard bands
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "B11", "B12"],
                output: {bands: 6}
            };
        }
        function evaluatePixel(sample) {
            return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    DataCollection.SENTINEL2_L2A,
                    from_date=date_start,
                    to_date=date_end,
                    mosaicking_order="mostRecent",
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_wgs84,
            size=(512, 512),  # Adjust resolution as needed
            config=None,  # uses ~/.sentinelhub/config.json
        )

        try:
            img = request.get_data()
            logger.info(f"Downloaded Sentinel-2 image: shape {img[0].shape}")

            from rasterio.io import MemoryFile
            from rasterio.transform import from_bounds as _from_bounds

            arr = img[0].astype("float32")  # (H, W, nbands)
            H, W, nbands = arr.shape
            bands_arr = np.transpose(arr, (2, 0, 1))  # (nbands, H, W)

            minx, miny, maxx, maxy = self.bbox
            transform = _from_bounds(minx, miny, maxx, maxy, W, H)
            band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]

            mem_meta = {
                "driver": "GTiff",
                "count": nbands,
                "height": H,
                "width": W,
                "dtype": "float32",
                "crs": "EPSG:4326",
                "transform": transform,
            }

            with MemoryFile() as mem:
                with mem.open(**mem_meta) as ds:
                    ds.write(bands_arr)
                da = rxr.open_rasterio(mem.name, masked=True)
                da = da.assign_coords(band=band_names[:nbands])

            da = da.rio.reproject(self.TARGET_CRS)
            logger.info(f"Sentinel-2 DataArray: shape={da.shape}, CRS={da.rio.crs}")
            return da
        except Exception as e:
            logger.error(f"sentinelhub request failed: {e}")
            return None

    def reproject_match(self, src_da: xr.DataArray, reference_da: xr.DataArray) -> xr.DataArray:
        """
        Reproject src_da to match reference_da's CRS and grid.
        """
        if src_da.rio.crs != reference_da.rio.crs:
            logger.info(f"Reprojecting from {src_da.rio.crs} to {reference_da.rio.crs}")
            src_da = src_da.rio.reproject(reference_da.rio.crs)

        logger.info("Resampling to reference grid...")
        src_resampled = src_da.rio.reproject_match(reference_da)
        return src_resampled

    def stack_mdt_sentinel(self, mdt: xr.DataArray, sentinel: xr.DataArray | None = None) -> xr.DataArray:
        """
        Stack MDT-50cm + Sentinel-2 bands into a single DataArray.

        If sentinel is None, returns MDT only.
        """
        if sentinel is None:
            logger.warning("No Sentinel-2 data; returning MDT-50cm only")
            return mdt.rename("MDT_50cm_elevation")

        # Stack along 'band' dimension
        mdt_band = mdt.assign_coords(band=0).expand_dims("band")

        # Assume sentinel has multiple bands
        stacked = xr.concat([mdt_band, sentinel], dim="band")
        stacked = stacked.assign_coords(
            band=["MDT_50cm", "S2_B02", "S2_B03", "S2_B04", "S2_B08", "S2_B11", "S2_B12"][: stacked.sizes["band"]]
        )
        return stacked

    def save_mosaic_geotiff(self, data: xr.DataArray, output_path: str, metadata: dict | None = None) -> str:
        """
        Save xarray DataArray to GeoTIFF.
        """
        output_path = Path(output_path)
        logger.info(f"Saving mosaic to {output_path}")

        data.rio.to_raster(str(output_path))

        # Save companion JSON metadata
        if metadata:
            meta_path = output_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {meta_path}")

        return str(output_path)

    def integrate(
        self, fetch_sentinel: bool = False, date_start: str = "2024-01-01", date_end: str = "2024-12-31", cloud_pct: int = 30
    ) -> dict:
        """
        Full integration pipeline:
          1. Download MDT-50cm tiles from DGT STAC
          2. Mosaic MDT-50cm
          3. Optionally fetch Sentinel-2 and reproject to match
          4. Stack both and save output GeoTIFF

        Args:
            fetch_sentinel: whether to fetch Sentinel-2
            date_start, date_end: date range for Sentinel-2
            cloud_pct: max cloud coverage for Sentinel-2

        Returns:
            dict with paths and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting DGT-Sentinel Integration Pipeline")
        logger.info("=" * 60)

        # Step 1: Download MDT-50cm
        logger.info("\n[1/4] Downloading MDT-50cm tiles from DGT STAC...")
        tif_paths = self.download_mdt50cm_tiles()
        if not tif_paths:
            logger.error("No MDT tiles downloaded; aborting")
            return {"status": "error", "message": "No MDT tiles found"}

        # Step 2: Mosaic MDT-50cm
        logger.info("\n[2/4] Mosaicing MDT-50cm tiles...")
        mdt_mosaic = self.mosaic_mdt50cm(tif_paths)
        mdt_out = self.output_dir / "MDT_50cm_mosaic_algarve.tif"
        self.save_mosaic_geotiff(
            mdt_mosaic,
            str(mdt_out),
            {
                "source": "DGT MDT-50cm STAC",
                "crs": self.TARGET_CRS,
                "bbox_wgs84": self.bbox,
                "resolution_m": 0.5,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Step 3: Fetch Sentinel-2 (optional)
        sentinel_mosaic = None
        if fetch_sentinel:
            logger.info("\n[3/4] Fetching Sentinel-2 L2A from Copernicus...")
            sentinel_mosaic = self.fetch_sentinel2_copernicus(date_start, date_end, cloud_pct)

            if sentinel_mosaic is not None:
                # Reproject to match MDT
                sentinel_mosaic = self.reproject_match(sentinel_mosaic, mdt_mosaic)
        else:
            logger.info("\n[3/4] Skipping Sentinel-2 fetch (fetch_sentinel=False)")

        # Step 4: Stack and save
        logger.info("\n[4/4] Stacking and saving integrated dataset...")
        stacked = self.stack_mdt_sentinel(mdt_mosaic, sentinel_mosaic)
        stacked_out = self.output_dir / "integrated_mdt_sentinel_algarve.tif"
        self.save_mosaic_geotiff(
            stacked,
            str(stacked_out),
            {
                "source": "DGT MDT-50cm + Sentinel-2 L2A",
                "crs": self.TARGET_CRS,
                "bbox_wgs84": self.bbox,
                "mdt_resolution_m": 0.5,
                "sentinel_resolution_m": 10,
                "bands": list(stacked.coords["band"].values),
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info("=" * 60)
        logger.info("Integration complete!")
        logger.info("=" * 60)

        return {
            "status": "success",
            "mdt_mosaic_path": str(mdt_out),
            "integrated_path": str(stacked_out),
            "mdt_shape": mdt_mosaic.shape,
            "integrated_shape": stacked.shape,
            "crs": self.TARGET_CRS,
            "bbox_wgs84": self.bbox,
        }


def main():
    """Example usage."""
    import logging as lg

    lg.basicConfig(level=lg.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Algarve, Boliqueime/Faro area
    bbox = (-8.25, 37.04, -8.17, 37.10)

    integrator = DGTSentinelIntegrator(bbox=bbox, output_dir="./outputs/dgt_sentinel")

    # Run full pipeline (fetch_sentinel=False for now, requires credentials)
    result = integrator.integrate(fetch_sentinel=False)

    print("\nIntegration Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
