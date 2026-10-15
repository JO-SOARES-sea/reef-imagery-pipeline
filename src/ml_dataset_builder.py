"""
ml_dataset_builder.py — ML Dataset generation for Reef Detection.

This module converts the outputs of the deterministic pipeline (GeoTIFFs and GeoJSONs)
into training pairs (image patches + mask patches) suitable for training a semantic
segmentation model (e.g., U-Net or SAM).

It relies on the currently generated rule-based reef polygons to bootstrap the dataset.
"""

import logging
import os

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

log = logging.getLogger(__name__)


def build_ml_dataset(
    image_tif: str, candidates_geojson: str, output_dir: str, patch_size: int = 128, stride: int = 64
) -> bool:
    """
    Generate ML training patches from a source TIFF and candidate polygons.

    Args:
        image_tif: Path to the source raster (e.g., Bathymetry or S2 band).
        candidates_geojson: Path to the GeoJSON containing reef polygons.
        output_dir: Base directory to save `images/` and `masks/` patches.
        patch_size: Size of the square patch (in pixels).
        stride: Stride for overlapping patches (e.g. 64 for 50% overlap of 128px patch).

    Returns:
        True if generation was successful, False otherwise.
    """
    if not os.path.exists(image_tif):
        log.error("ML Dataset: Source image not found: %s", image_tif)
        return False

    if not os.path.exists(candidates_geojson):
        log.error("ML Dataset: Labels GeoJSON not found: %s", candidates_geojson)
        return False

    if not HAS_GEOPANDAS:
        log.error("ML Dataset: geopandas not installed; cannot load candidate polygons")
        return False

    images_dir = os.path.join(output_dir, "ml_dataset", "images")
    masks_dir = os.path.join(output_dir, "ml_dataset", "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    try:
        # Load polygons
        gdf = gpd.read_file(candidates_geojson)

        with rasterio.open(image_tif) as src:
            meta = src.meta.copy()
            width = src.width
            height = src.height
            transform = src.transform

            # Reproject polygons to raster CRS if needed
            raster_crs = src.crs.to_epsg() if src.crs else 4326
            if gdf.crs and gdf.crs.to_epsg() != raster_crs:
                gdf = gdf.to_crs(epsg=raster_crs)

            # Rasterize the polygons to create the ground truth mask
            geometries = [geom for geom in gdf.geometry if geom.is_valid and not geom.is_empty]

            if not geometries:
                log.warning("ML Dataset: GeoJSON contains no valid geometries. Cannot create positive masks.")
                mask_arr = np.zeros((height, width), dtype=np.uint8)
            else:
                mask_arr = rasterize(
                    geometries, out_shape=(height, width), transform=transform, fill=0, default_value=1, dtype=np.uint8
                )

            # Generate patches
            patch_idx = 0
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    window = Window(x, y, patch_size, patch_size)

                    img_patch = src.read(window=window)
                    mask_patch = mask_arr[y : y + patch_size, x : x + patch_size]

                    # Filter out patches that are entirely NoData
                    # Assuming nodata is NaN for float or 0 for int
                    if src.nodata is not None and np.all(img_patch == src.nodata):
                        continue
                    if np.all(np.isnan(img_patch)):
                        continue

                    # Save patches
                    patch_meta = meta.copy()
                    patch_meta.update(
                        {"height": patch_size, "width": patch_size, "transform": rasterio.windows.transform(window, transform)}
                    )

                    # Save Image
                    img_out = os.path.join(images_dir, f"patch_{patch_idx:04d}.tif")
                    with rasterio.open(img_out, "w", **patch_meta) as dst_img:
                        dst_img.write(img_patch)

                    # Save Mask
                    mask_meta = patch_meta.copy()
                    mask_meta.update({"count": 1, "dtype": "uint8", "nodata": None})
                    mask_out = os.path.join(masks_dir, f"patch_{patch_idx:04d}.tif")
                    with rasterio.open(mask_out, "w", **mask_meta) as dst_mask:
                        dst_mask.write(mask_patch, 1)

                    patch_idx += 1

        log.info(
            "✓ ML Dataset generated: %d patches (size %dx%d) saved to %s",
            patch_idx,
            patch_size,
            patch_size,
            os.path.join(output_dir, "ml_dataset"),
        )
        return True

    except Exception as exc:
        log.error("ML Dataset generation failed: %s", exc)
        return False
