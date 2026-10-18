"""
src/s2_grid.py — Native-grid windowing helpers for Sentinel-2 COGs.

* A lon/lat bbox is projected into the raster CRS with edge densification (it is not
  axis-aligned in UTM) and snapped OUTWARD to whole pixels of the native grid, then
  clamped to the raster extent. Reads therefore never resample and the window's affine
  transform is exact (``rasterio.windows.transform``).
* ``unsigned_href`` strips query strings (e.g. Planetary Computer SAS tokens) before an
  href is written to any metadata file, so credentials never leak into provenance records.
"""

from __future__ import annotations

import math
from urllib.parse import urlsplit, urlunsplit

from rasterio.windows import Window


class WindowOutsideRaster(ValueError):
    """The requested bbox does not overlap the raster."""


def bbox_window(src, bbox_wgs84: tuple) -> Window:
    """Integer window of *src* covering *bbox_wgs84* = (lon_min, lat_min, lon_max, lat_max)."""
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84, densify_pts=21)
    inv = ~src.transform
    corners = ((left, top), (right, top), (left, bottom), (right, bottom))
    cols, rows = zip(*(inv @ c for c in corners), strict=True)
    c0 = max(0, int(math.floor(min(cols))))
    r0 = max(0, int(math.floor(min(rows))))
    c1 = min(src.width, int(math.ceil(max(cols))))
    r1 = min(src.height, int(math.ceil(max(rows))))
    if c1 <= c0 or r1 <= r0:
        raise WindowOutsideRaster(f"bbox {bbox_wgs84} does not overlap raster {getattr(src, 'name', '?')}")
    return Window(c0, r0, c1 - c0, r1 - r0)


def unsigned_href(href: str) -> str:
    parts = urlsplit(href)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
