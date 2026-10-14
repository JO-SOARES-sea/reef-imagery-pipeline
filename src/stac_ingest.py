"""STAC ingestion helpers for Sentinel-2 data.

This module is intentionally lightweight and reusable by scripts that need
Sentinel-2 scene discovery, asset extraction, and cloud-filtered scene ranking.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import planetary_computer as pc
from pystac import Item
from pystac_client import Client

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_S2_COLLECTION = "sentinel-2-l2a"
STAC_URL_PRIORITY = [EARTH_SEARCH_STAC_URL, PC_STAC_URL]

# DGT (Portugal national mapping agency) hosts annual Sentinel-2 L2A mosaics
# under collections named MosaicoS2-YYYY (2015–2025).  They use a different
# collection scheme so they get their own search function below.
DGT_STAC_URL = "https://dgt-be.a.incd.pt:8081"
DGT_S2_YEARS = list(range(2025, 2014, -1))  # newest first


def open_stac_catalog(url: str = EARTH_SEARCH_STAC_URL) -> Client:
    """Open a STAC catalog and apply signing if required."""
    modifier = pc.sign_inplace if "planetarycomputer.microsoft.com" in url else None
    return Client.open(url, modifier=modifier)


def normalize_datetime_range(date_range: str | tuple[str, str] | tuple[datetime, datetime]) -> str:
    if isinstance(date_range, str):
        return date_range
    if len(date_range) != 2:
        raise ValueError("date_range must be a string or a 2-tuple")
    start, end = date_range
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d")
    return f"{start}/{end}"


def search_sentinel2_scenes(
    lat: float,
    lon: float,
    date_range: str | tuple[str, str] | tuple[datetime, datetime],
    max_cloud_cover: float = 25.0,
    catalog_url: str | None = None,
    collection: str = DEFAULT_S2_COLLECTION,
    limit: int = 10,
) -> list[Item]:
    """Search Sentinel-2 L2A scenes for a point, date range, and cloud filter.

    If catalog_url is None, this function tries the preferred STAC endpoints in
    order and returns the first successful result.
    """
    urls = [catalog_url] if catalog_url else STAC_URL_PRIORITY
    datetime_range = normalize_datetime_range(date_range)
    last_error: Exception | None = None

    for url in urls:
        try:
            catalog = open_stac_catalog(url)
            search = catalog.search(
                collections=[collection],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            )
            items = list(search.items())
            return items[:limit]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(f"STAC scene search failed for all endpoints {urls}: {last_error}")


def choose_least_cloudy(items: Iterable[Item]) -> Item | None:
    """Return the scene with the lowest STAC cloud cover from a collection of items."""
    items = [item for item in items]
    if not items:
        return None
    return min(items, key=lambda item: item.properties.get("eo:cloud_cover", 100.0))


def get_asset_hrefs(item: Item, asset_keys: Iterable[str]) -> dict[str, str]:
    """Return a mapping of asset key -> signed asset URL for a STAC item."""
    hrefs: dict[str, str] = {}

    # Build a case-insensitive map of available asset keys
    available = {k.lower(): v for k, v in item.assets.items()}

    # Common alias mapping for Sentinel-2 band names in different STAC catalogs
    alias_map = {
        "b02": ["blue", "b02"],
        "b03": ["green", "b03"],
        "b04": ["red", "b04"],
        "b08": ["nir", "b08", "b8a"],
        "b11": ["swir16", "b11"],
        "b12": ["swir22", "b12"],
    }

    for key in asset_keys:
        k_low = key.lower()

        # Direct match
        if k_low in available:
            hrefs[key] = available[k_low].href
            continue

        # Try aliases (e.g., 'B02' -> 'blue')
        aliases = alias_map.get(k_low, [k_low])
        found = False
        for a in aliases:
            if a in available:
                hrefs[key] = available[a].href
                found = True
                break

        if found:
            continue

        # Fallback: look for any asset key that contains the band name
        for a_key, a_asset in available.items():
            if k_low in a_key:
                hrefs[key] = a_asset.href
                break

    return hrefs


def search_dgt_s2_scenes(
    lat: float,
    lon: float,
    max_cloud_cover: float = 25.0,
    year: int | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search DGT-hosted Sentinel-2 L2A mosaics (MosaicoS2-YYYY collections).

    Returns raw STAC feature dicts (not pystac Items) since the DGT STAC does
    not expose a pystac-compatible search endpoint.  Use this as a fallback when
    Earth Search and Planetary Computer are unavailable.

    Args:
        lat, lon: point of interest
        max_cloud_cover: filter items above this percentage
        year: specific year to search (None → tries all years newest-first)
        limit: max items to return

    Returns:
        list of STAC feature dicts sorted by cloud cover ascending
    """
    import requests as _req

    years = [year] if year else DGT_S2_YEARS
    bbox = f"{lon - 0.05},{lat - 0.05},{lon + 0.05},{lat + 0.05}"

    results: list[dict[str, Any]] = []
    for yr in years:
        url = f"{DGT_STAC_URL}/collections/MosaicoS2-{yr}/items"
        try:
            r = _req.get(url, params={"bbox": bbox, "limit": 50, "f": "json"}, timeout=20)
            r.raise_for_status()
            features = r.json().get("features", [])
        except Exception:
            continue

        for f in features:
            props = f.get("properties", {})
            cc = props.get("eo:cloud_cover")
            if cc is None:
                cc = props.get("s2:cloud_cover")
            if cc is None:
                cc = 0.0
            if cc <= max_cloud_cover:
                f.setdefault("_dgt_year", yr)
                results.append(f)

        if results:
            break  # found items in this year — don't walk further back

    results.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100.0))
    return results[:limit]


def scene_summary(item: Item) -> dict[str, Any]:
    """Return a lightweight metadata summary for a Sentinel-2 STAC item."""
    return {
        "id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "sun_elevation": item.properties.get("view:sun_elevation"),
        "platform": item.properties.get("platform"),
        "collection": item.collection_id,
        "assets": sorted(item.assets.keys()),
    }


def _first_number(props: dict[str, Any], *keys: str) -> float | None:
    """Return the first property among keys that holds a numeric value (catalogs may publish explicit nulls)."""
    for key in keys:
        value = props.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def metadata_from_stac_item(item: Item) -> dict[str, Any]:
    """Extract a run_predictor-compatible metadata dict from a Sentinel-2 STAC item.

    This is the production replacement for compute_metadata_stub() in utils.py.
    All fields fall back to neutral defaults when the STAC catalog does not
    expose the corresponding property or publishes it as null.

    Sentinel-2 STAC property mapping (Earth Search key first, Planetary Computer fallback):
        view:sun_elevation / s2:mean_solar_zenith → solar_zenith_deg  (zenith = 90 − elevation)
        view:sun_azimuth / s2:mean_solar_azimuth  → solar_azimuth_deg
        view:incidence_angle → satellite_zenith_deg
        view:azimuth         → satellite_azimuth_deg
        eo:cloud_cover       → cloud_cover_pct
        proj:code / proj:epsg → crs  (as "EPSG:<code>")
        s2:product_type / processing:level → level  (normalised to "L2A"/"L1C")
    """
    props = item.properties

    sun_elev = _first_number(props, "view:sun_elevation")
    pc_zenith = _first_number(props, "s2:mean_solar_zenith")
    if sun_elev is not None:
        solar_zenith = 90.0 - sun_elev
    elif pc_zenith is not None:
        solar_zenith = pc_zenith
    else:
        solar_zenith = 40.0
    solar_azimuth = _first_number(props, "view:sun_azimuth", "s2:mean_solar_azimuth")
    sat_zenith = _first_number(props, "view:incidence_angle")
    sat_azimuth = _first_number(props, "view:azimuth")
    cloud_cover = _first_number(props, "eo:cloud_cover")

    # Projection extension v2 replaced proj:epsg (int) with proj:code ("EPSG:32629"); Earth Search v1 uses the latter
    epsg = props.get("proj:epsg")
    crs = props.get("proj:code") or (f"EPSG:{epsg}" if epsg else "EPSG:32629")

    # PC: "MSIL2A"; Earth Search v1: "S2MSI2A"; generic: processing:level "L2A"
    raw_level = str(props.get("s2:product_type") or props.get("processing:level") or "L2A").upper()
    if raw_level.endswith("2A"):
        level = "L2A"
    elif raw_level.endswith("1C"):
        level = "L1C"
    else:
        level = raw_level

    if item.datetime is not None:
        date_str = item.datetime.strftime("%Y-%m-%d")
    else:
        date_str = str(props.get("datetime") or "")[:10]

    return {
        "date": date_str,
        "crs": crs,
        "datum": "WGS84",
        "level": level,
        "solar_zenith_deg": round(solar_zenith, 3),
        "solar_azimuth_deg": round(solar_azimuth if solar_azimuth is not None else 150.0, 3),
        "satellite_zenith_deg": round(sat_zenith if sat_zenith is not None else 5.0, 3),
        "satellite_azimuth_deg": round(sat_azimuth if sat_azimuth is not None else 10.0, 3),
        "cloud_cover_pct": round(cloud_cover if cloud_cover is not None else 2.0, 3),
    }
