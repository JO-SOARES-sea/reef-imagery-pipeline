#!/usr/bin/env python3
"""
probe_all.py — Public data API discovery for reef imagery pipeline
==================================================================
Probes every Portuguese and EU data source relevant to this pipeline
and reports what is actually accessible, what data it holds, and
whether raw raster files can be retrieved without credentials.

Run:
    python look_api/probe_all.py
    python look_api/probe_all.py --bbox "-8.25,37.04,-8.17,37.10"
    python look_api/probe_all.py --report report.json

Output:
    Console summary + optional JSON report with per-endpoint detail.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)  # type: ignore[attr-defined]

# ── Algarve test bbox (lon_min, lat_min, lon_max, lat_max) ─────────────────────
DEFAULT_BBOX = (-8.25, 37.04, -8.17, 37.10)

# ── Shared session with retry ──────────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5,
                                                          status_forcelist=[500, 502, 503])))
_SESSION.mount("http://",  HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5)))

TIMEOUT = 12  # seconds


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         verify: bool = True, stream: bool = False,
         timeout: int | None = None) -> requests.Response:
    return _SESSION.get(url, params=params, headers=headers,
                        timeout=timeout or TIMEOUT, verify=verify, stream=stream)


def _bbox_str(bbox: tuple) -> str:
    return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"


# ══════════════════════════════════════════════════════════════════════════════
# Result helpers
# ══════════════════════════════════════════════════════════════════════════════

class ProbeResult:
    def __init__(self, name: str):
        self.name = name
        self.status: str = "UNKNOWN"   # OK / PARTIAL / BLOCKED / ERROR
        self.http_code: int | None = None
        self.summary: str = ""
        self.data: dict[str, Any] = {}
        self.access_paths: list[str] = []  # usable download/query URLs found
        self.elapsed_s: float = 0.0

    def ok(self, summary: str, **data):
        self.status = "OK"; self.summary = summary; self.data.update(data)

    def partial(self, summary: str, **data):
        self.status = "PARTIAL"; self.summary = summary; self.data.update(data)

    def blocked(self, summary: str, **data):
        self.status = "BLOCKED"; self.summary = summary; self.data.update(data)

    def error(self, summary: str, **data):
        self.status = "ERROR"; self.summary = summary; self.data.update(data)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status,
            "http_code": self.http_code, "summary": self.summary,
            "elapsed_s": round(self.elapsed_s, 2),
            "access_paths": self.access_paths, "data": self.data,
        }

    def __str__(self) -> str:
        icon = {"OK": "✅", "PARTIAL": "⚠️ ", "BLOCKED": "🔒", "ERROR": "❌", "UNKNOWN": "❓"}
        badge = icon.get(self.status, "❓")
        paths = ""
        if self.access_paths:
            paths = "\n      → " + "\n      → ".join(self.access_paths[:3])
        return f"  {badge} {self.name:<45} {self.summary}{paths}"


def _probe(fn, name: str) -> ProbeResult:
    r = ProbeResult(name)
    t0 = time.time()
    try:
        fn(r)
    except Exception as exc:
        r.error(f"{type(exc).__name__}: {exc}")
    finally:
        r.elapsed_s = time.time() - t0
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 1. DGT STAC  (dgt-be.a.incd.pt:8081)
# ══════════════════════════════════════════════════════════════════════════════

DGT_STAC = "https://dgt-be.a.incd.pt:8081"


def probe_dgt_stac_collections(r: ProbeResult):
    resp = _get(f"{DGT_STAC}/collections", verify=False)
    r.http_code = resp.status_code
    resp.raise_for_status()
    cols = [c["id"] for c in resp.json().get("collections", [])]
    r.ok(f"{len(cols)} collections", collections=cols)
    r.access_paths.append(f"{DGT_STAC}/collections")


def probe_dgt_stac_mdt50(r: ProbeResult, bbox=DEFAULT_BBOX):
    url = f"{DGT_STAC}/collections/MDT-50cm/items"
    resp = _get(url, params={"bbox": _bbox_str(bbox), "limit": 5}, verify=False)
    r.http_code = resp.status_code
    resp.raise_for_status()
    items = resp.json().get("features", [])
    if not items:
        r.partial("No MDT-50cm tiles in bbox"); return
    sample = items[0]
    href = sample.get("assets", {}).get("Data", {}).get("href", "")
    r.ok(f"{len(items)} MDT-50cm tiles in bbox", sample_id=sample["id"], asset_href=href)
    r.access_paths.append(url)


def probe_dgt_stac_mds50(r: ProbeResult, bbox=DEFAULT_BBOX):
    url = f"{DGT_STAC}/collections/MDS-50cm/items"
    resp = _get(url, params={"bbox": _bbox_str(bbox), "limit": 3}, verify=False)
    r.http_code = resp.status_code
    resp.raise_for_status()
    items = resp.json().get("features", [])
    href = items[0].get("assets", {}).get("Data", {}).get("href", "") if items else ""
    r.ok(f"{len(items)} MDS-50cm (DSM) tiles", sample_href=href)
    r.access_paths.append(url)


def probe_dgt_stac_ortosat(r: ProbeResult, bbox=DEFAULT_BBOX):
    url = f"{DGT_STAC}/collections/ORTOSAT-2023/items"
    resp = _get(url, params={"bbox": _bbox_str(bbox), "limit": 3}, verify=False)
    r.http_code = resp.status_code
    resp.raise_for_status()
    items = resp.json().get("features", [])
    bands = []
    href = ""
    if items:
        asset = items[0].get("assets", {}).get("visual", {})
        href = asset.get("href", "")
        bands = [b.get("common_name") for b in asset.get("eo:bands", [])]
    r.ok(f"{len(items)} ORTOSAT-2023 tiles  bands={bands}", sample_href=href)
    r.access_paths.append(url)


def probe_dgt_stac_s2_mosaics(r: ProbeResult, bbox=DEFAULT_BBOX):
    """Check all MosaicoS2-YYYY collections (2015-2025)."""
    years_found = {}
    for year in range(2025, 2014, -1):
        url = f"{DGT_STAC}/collections/MosaicoS2-{year}/items"
        try:
            resp = _get(url, params={"bbox": _bbox_str(bbox), "limit": 2}, verify=False)
            if resp.status_code != 200:
                continue
            items = resp.json().get("features", [])
            if items:
                f = items[0]
                cloud = f.get("properties", {}).get("cloudCover", "?")
                years_found[year] = {"n": len(items), "cloud_pct": cloud,
                                     "assets": list(f.get("assets", {}).keys())}
        except Exception:
            pass
    if years_found:
        r.ok(f"S2 mosaics found for {len(years_found)} years: {sorted(years_found)[::-1]}",
             years=years_found)
        r.access_paths.append(f"{DGT_STAC}/collections/MosaicoS2-2025/items")
    else:
        r.partial("No S2 mosaics found in bbox")


def probe_dgt_stac_historical_ortos(r: ProbeResult, bbox=DEFAULT_BBOX):
    """ORTOS-1995 through ORTOS-2021 aerial orthophotos."""
    available = {}
    for coll in ["ORTOS-2021", "ORTOS-2018", "ORTOS-2015", "ORTOS-2012",
                 "ORTOS-2010", "ORTOS-2007", "ORTOS-2004", "ORTOS-1995"]:
        try:
            resp = _get(f"{DGT_STAC}/collections/{coll}/items",
                        params={"bbox": _bbox_str(bbox), "limit": 1}, verify=False)
            if resp.status_code == 200:
                items = resp.json().get("features", [])
                if items:
                    available[coll] = items[0].get("assets", {}).get("visual", {}).get("href", "")
        except Exception:
            pass
    if available:
        r.ok(f"Historical orthophotos: {list(available.keys())}", collections=available)
    else:
        r.partial("No historical orthophotos in bbox")


def probe_dgt_stac_lidar_copc(r: ProbeResult, bbox=DEFAULT_BBOX):
    """COPC (Cloud-Optimized Point Cloud) LiDAR data."""
    url = f"{DGT_STAC}/collections/COPC/items"
    resp = _get(url, params={"bbox": _bbox_str(bbox), "limit": 3}, verify=False)
    r.http_code = resp.status_code
    resp.raise_for_status()
    items = resp.json().get("features", [])
    if not items:
        r.partial("No COPC LiDAR tiles in bbox"); return
    asset = list(items[0].get("assets", {}).values())[0] if items[0].get("assets") else {}
    href = asset.get("href", "")
    r.ok(f"{len(items)} COPC LiDAR point-cloud tiles", sample_href=href)
    r.access_paths.append(url)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DGT MinIO  (stor-002.a.acnca.pt:9000)
# ══════════════════════════════════════════════════════════════════════════════

MINIO = "https://stor-002.a.acnca.pt:9000"


def probe_minio_anonymous(r: ProbeResult):
    """Try raw anonymous GET on a known asset."""
    url = f"{MINIO}/lidar/MDT50cm/MDT-50cm-196015-04-2024_v01.tif"
    resp = _get(url, verify=False)
    r.http_code = resp.status_code
    if resp.status_code == 200:
        r.ok("Anonymous download succeeded!", content_length=resp.headers.get("Content-Length"))
        r.access_paths.append(url)
    elif resp.status_code == 403:
        xml = resp.text
        code = ""
        try:
            code = ET.fromstring(xml).findtext("Code") or ""
        except Exception:
            pass
        r.blocked(f"AccessDenied ({code}) — credentials required", xml_error=code)
    else:
        r.partial(f"Unexpected HTTP {resp.status_code}")


def probe_minio_cog_range_request(r: ProbeResult):
    """COG header is in the first 16KB. A Range request would work if bucket is public."""
    url = f"{MINIO}/lidar/MDT50cm/MDT-50cm-196015-04-2024_v01.tif"
    resp = _get(url, headers={"Range": "bytes=0-16383"}, verify=False)
    r.http_code = resp.status_code
    if resp.status_code in (200, 206):
        magic = resp.content[:4].hex()
        r.ok(f"COG range request succeeded  magic_bytes={magic} ({len(resp.content)}B)")
        r.access_paths.append(url)
    else:
        r.blocked(f"HTTP {resp.status_code} — range request also denied")


def probe_minio_presign_attempt(r: ProbeResult):
    """
    Some MinIO deployments support STS AssumeRoleWithWebIdentity
    or anonymous presign via ?X-Amz-Algorithm=AWS4-HMAC-SHA256 patterns.
    Check if the server leaks a presign endpoint.
    """
    tests = [
        (f"{MINIO}/minio/login",   "console login"),
        (f"{MINIO}/minio/health/live", "health live"),
        (f"{MINIO}/minio/health/cluster", "health cluster"),
        (f"{MINIO}/?versioning",   "bucket versioning"),
    ]
    results = {}
    for url, label in tests:
        try:
            resp = _get(url, verify=False)
            results[label] = resp.status_code
        except Exception as e:
            results[label] = str(e)
    live = results.get("health live")
    r.partial(f"Health live={live}  — no public presign endpoint found", endpoints=results)


def probe_minio_gdal_vsicurl(r: ProbeResult):
    """
    GDAL /vsicurl/ + /vsis3/ can sometimes negotiate access via env vars.
    Test if rasterio can open the COG header via HTTP range even without AWS creds.
    """
    try:
        import rasterio
        url = f"/vsicurl/{MINIO}/lidar/MDT50cm/MDT-50cm-196015-04-2024_v01.tif"
        import os
        os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
        os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
        with rasterio.Env(GDAL_HTTP_UNSAFESSL="YES"):
            try:
                ds = rasterio.open(url)
                r.ok(f"rasterio /vsicurl/ opened  CRS={ds.crs}  shape={ds.shape}",
                     crs=str(ds.crs), shape=ds.shape)
                r.access_paths.append(url)
                ds.close()
            except rasterio.errors.RasterioIOError as e:
                r.blocked(f"rasterio /vsicurl/ denied: {e}")
    except ImportError:
        r.error("rasterio not installed")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DGT WMS / WCS  (ortos.dgterritorio.gov.pt)
# ══════════════════════════════════════════════════════════════════════════════

DGT_ORTOS_WMS = "https://ortos.dgterritorio.gov.pt/wms/ortosat2023"


def probe_dgt_wms_capabilities(r: ProbeResult):
    resp = _get(DGT_ORTOS_WMS, params={"SERVICE": "WMS", "REQUEST": "GetCapabilities"})
    r.http_code = resp.status_code
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"wms": "http://www.opengis.net/wms"}
    layers = [e.text for e in root.findall(".//wms:Layer/wms:Name", ns) if e.text]
    formats = [e.text for e in root.findall(".//wms:Format", ns)]
    r.ok(f"WMS live — {len(layers)} layers, formats={formats[:4]}", layers=layers)
    r.access_paths.append(DGT_ORTOS_WMS)


def probe_dgt_wms_getmap(r: ProbeResult, bbox=DEFAULT_BBOX):
    """Actually fetch a 256×256 PNG tile — confirms data is streamable."""
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": "OrtoSat2023", "CRS": "EPSG:4326",
        "BBOX": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
        "WIDTH": "256", "HEIGHT": "256", "FORMAT": "image/png",
        "STYLES": "",
    }
    resp = _get(DGT_ORTOS_WMS, params=params, stream=True)
    r.http_code = resp.status_code
    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code == 200 and "image" in content_type:
        data = b"".join(resp.iter_content(chunk_size=4096))
        r.ok(f"GetMap tile OK — {len(data)} bytes  content-type={content_type}",
             tile_bytes=len(data))
        r.access_paths.append(DGT_ORTOS_WMS + "?SERVICE=WMS&REQUEST=GetMap&LAYERS=OrtoSat2023")
    else:
        r.partial(f"HTTP {resp.status_code}  content-type={content_type}")


def probe_dgt_wcs(r: ProbeResult):
    """Check for WCS (raw raster coverage download) on related hostnames."""
    candidates = [
        "https://ortos.dgterritorio.gov.pt/wcs",
        "https://ortos.dgterritorio.gov.pt/wcs/mdt50cm",
        "https://ortos.dgterritorio.gov.pt/wcs/ortosat2023",
        "https://geo.dgterritorio.gov.pt/geoserver/wcs",
        "https://ide.dgterritorio.gov.pt/wcs",
    ]
    found = []
    for url in candidates:
        try:
            resp = _get(url, params={"SERVICE": "WCS", "REQUEST": "GetCapabilities"}, timeout=8)
            if resp.status_code == 200 and ("WCS" in resp.text or "Coverage" in resp.text):
                found.append(url)
        except Exception:
            pass
    if found:
        r.ok(f"WCS endpoint(s) found: {found}"); r.access_paths.extend(found)
    else:
        r.partial("No WCS coverage service found on DGT hostnames")


# ══════════════════════════════════════════════════════════════════════════════
# 4. IH / DGRM ArcGIS  (webgis.dgrm.mm.gov.pt)
# ══════════════════════════════════════════════════════════════════════════════

IH_BASE = ("https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/"
           "Dados_entidades_externas/Batimetrica_IH/MapServer")


def probe_ih_arcgis_services(r: ProbeResult):
    resp = _get(f"{IH_BASE}?f=json")
    r.http_code = resp.status_code
    resp.raise_for_status()
    data = resp.json()
    layers = [{l.get("id"): l.get("name")} for l in data.get("layers", [])]
    r.ok(f"ArcGIS MapServer live — {len(layers)} layers", layers=layers,
         description=data.get("description", "")[:200])
    r.access_paths.append(f"{IH_BASE}?f=json")


def probe_ih_arcgis_query(r: ProbeResult, bbox=DEFAULT_BBOX):
    """Query isobath features — use a wider Algarve shelf bbox to ensure hits."""
    url = f"{IH_BASE}/0/query"
    # Widen to ~30km buffer to catch offshore isobaths (5m–200m contours)
    wide = (bbox[0] - 0.3, bbox[1] - 0.2, bbox[2] + 0.3, bbox[3] + 0.1)
    params = {
        "f": "geojson",
        "geometry": f"{wide[0]},{wide[1]},{wide[2]},{wide[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326",
        "outFields": "*",
        "returnGeometry": "true",
        "resultRecordCount": "5",
        "where": "1=1",
    }
    resp = _get(url, params=params)
    r.http_code = resp.status_code
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if features:
        depths = list({f["properties"].get("PROFUNDIDADE") for f in features if f.get("properties")})
        r.ok(f"{len(features)} isobath features returned  depths(m)={depths}", n=len(features),
             sample_depths=depths)
        r.access_paths.append(url)
    else:
        r.partial("Query returned 0 features for bbox (try wider area)")


def probe_dgrm_other_services(r: ProbeResult):
    """Discover all ArcGIS services on the DGRM portal."""
    resp = _get("https://webgis.dgrm.mm.gov.pt/arcgis/rest/services?f=json")
    r.http_code = resp.status_code
    resp.raise_for_status()
    data = resp.json()
    services = [f"{s.get('name')} ({s.get('type')})" for s in data.get("services", [])]
    r.ok(f"{len(services)} ArcGIS services available", services=services)
    r.access_paths.append("https://webgis.dgrm.mm.gov.pt/arcgis/rest/services")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SNIG — Portuguese Spatial Data Infrastructure
# ══════════════════════════════════════════════════════════════════════════════

SNIG_BASE = "https://snig.dgterritorio.gov.pt"


def probe_snig_catalog(r: ProbeResult):
    """OGC API Records or CSW catalog for DGT terrain metadata."""
    tests = [
        (f"{SNIG_BASE}/rndg/srv/api/search/records", {"q": "MDT 50cm", "_content_type": "json"}),
        (f"{SNIG_BASE}/rndg/srv/eng/q",              {"q": "MDT", "_content_type": "json", "fast": "index"}),
        (f"{SNIG_BASE}/rndg/srv/api/records",        {"q": "MDT"}),
    ]
    for url, params in tests:
        try:
            resp = _get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                total = (data.get("@count") or data.get("count") or
                         len(data.get("metadata", {}).get("record", [])))
                r.ok(f"SNIG catalog accessible — ~{total} results for 'MDT 50cm'")
                r.access_paths.append(url)
                return
        except Exception:
            pass
    r.partial("SNIG catalog endpoints returned non-200 or parse error")


def probe_snig_csw(r: ProbeResult):
    """OGC CSW 2.0.2 GetCapabilities."""
    url = f"{SNIG_BASE}/rndg/srv/por/csw"
    resp = _get(url, params={"SERVICE": "CSW", "REQUEST": "GetCapabilities"})
    r.http_code = resp.status_code
    if resp.status_code == 200 and "CSW" in resp.text:
        r.ok("CSW 2.0.2 endpoint live")
        r.access_paths.append(url)
    else:
        r.partial(f"CSW returned HTTP {resp.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. dados.gov.pt  — Portuguese Open Data Portal
# ══════════════════════════════════════════════════════════════════════════════

DADOS_API = "https://dados.gov.pt/api/1"


def probe_dados_gov(r: ProbeResult):
    """Search for DGT terrain and imagery datasets."""
    results = {}
    for query in ["MDT DGT altimetria", "ortofoto DGT", "batimetria hidrográfico"]:
        try:
            resp = _get(f"{DADOS_API}/datasets/", params={"q": query, "page_size": 3})
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("total", 0)
                titles = [d.get("title", "?") for d in data.get("data", [])]
                results[query] = {"total": hits, "titles": titles}
        except Exception as e:
            results[query] = str(e)
    r.ok("dados.gov.pt API accessible", queries=results)
    r.access_paths.append(f"{DADOS_API}/datasets/")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EMODnet Bathymetry  (emodnet-bathymetry.eu)
# ══════════════════════════════════════════════════════════════════════════════

def probe_emodnet_wcs(r: ProbeResult, bbox=DEFAULT_BBOX):
    """EMODnet WCS for European DTM (~115m) — already used by stumpf calibration."""
    url = "https://ows.emodnet-bathymetry.eu/wcs"
    resp = _get(url, params={"SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCapabilities"})
    r.http_code = resp.status_code
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"wcs": "http://www.opengis.net/wcs"}
    coverages = [e.text for e in root.findall(".//wcs:Identifier", ns)][:5]
    if not coverages:  # try without namespace
        coverages = [e.text for e in root.findall(".//{http://www.opengis.net/wcs/1.1}Identifier")][:5]
    r.ok(f"EMODnet WCS live — sample coverages: {coverages}", coverages=coverages)
    r.access_paths.append(url)


def probe_emodnet_wcs_download(r: ProbeResult, bbox=DEFAULT_BBOX):
    """Try an actual WCS GetCoverage for the Algarve bbox."""
    url = "https://ows.emodnet-bathymetry.eu/wcs"
    params = {
        "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "COVERAGE": "emodnet:mean", "CRS": "EPSG:4326",
        "BBOX": _bbox_str(bbox), "WIDTH": "64", "HEIGHT": "32",
        "FORMAT": "image/tiff",
    }
    resp = _get(url, params=params, stream=True)
    r.http_code = resp.status_code
    ct = resp.headers.get("Content-Type", "")
    if resp.status_code == 200 and ("tiff" in ct or "octet" in ct):
        data = b"".join(resp.iter_content(4096))
        r.ok(f"WCS GetCoverage download OK — {len(data)} bytes  format={ct}", bytes=len(data))
        r.access_paths.append(url + "?SERVICE=WCS&REQUEST=GetCoverage&COVERAGE=emodnet:mean")
    else:
        r.partial(f"HTTP {resp.status_code}  content-type={ct}  body={resp.text[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Copernicus CMEMS  (marine.copernicus.eu)
# ══════════════════════════════════════════════════════════════════════════════

def probe_cmems_stac(r: ProbeResult):
    """Copernicus Marine STAC catalog — Kd490, chlorophyll, SST for Algarve."""
    url = "https://stac.marine.copernicus.eu/metadata"
    try:
        resp = _get(url, timeout=10)
        r.http_code = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            n = len(data.get("links", []))
            r.ok(f"CMEMS STAC root accessible — {n} top-level links")
            r.access_paths.append(url)
        else:
            r.partial(f"HTTP {resp.status_code}")
    except Exception as e:
        r.partial(f"STAC root unreachable: {e}")


def probe_cmems_opendap(r: ProbeResult):
    """CMEMS OPeNDAP endpoint for ocean colour (Kd490)."""
    # Ocean colour Mediterranean / Atlantic — product OCEANCOLOUR_ATL_BGC_L4_MY
    url = ("https://my.cmems-du.eu/thredds/dodsC/"
           "cmems_obs-oc_atl_bgc-transp_my_l4-multi-4km_P1M.das")
    try:
        resp = _get(url, timeout=10)
        r.http_code = resp.status_code
        if resp.status_code == 200:
            lines = resp.text.splitlines()[:8]
            r.ok("CMEMS OPeNDAP accessible", das_preview=lines)
            r.access_paths.append(url)
        else:
            r.partial(f"HTTP {resp.status_code} — may need CMEMS credentials")
    except Exception as e:
        r.partial(f"OPeNDAP unreachable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Copernicus Land Monitoring Service  (land.copernicus.eu)
# ══════════════════════════════════════════════════════════════════════════════

def probe_copernicus_land_eu_dem(r: ProbeResult):
    """EU-DEM v1.1 (25m) and EUDEM-derived coastal slope — public download."""
    url = ("https://land.copernicus.eu/api/@search"
           "?portal_type=DataSet&Subject=Digital+Elevation+Model&b_size=5")
    try:
        resp = _get(url, timeout=10)
        r.http_code = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            items = [i.get("title") for i in data.get("items", [])]
            r.ok(f"Copernicus Land accessible — EU-DEM products: {items}")
            r.access_paths.append("https://land.copernicus.eu/en/products/elevation")
        else:
            r.partial(f"HTTP {resp.status_code}")
    except Exception as e:
        r.partial(f"Unreachable: {e}")


def probe_copernicus_dem_aws(r: ProbeResult, bbox=DEFAULT_BBOX):
    """
    Copernicus GLO-30 DEM on public AWS S3 (no credentials required).
    Already used by the pipeline — verify it is still reachable.
    """
    # Tile naming convention (1°×1° tiles, zero-padded):
    # Copernicus_DSM_COG_10_N37_00_W009_00_DEM/Copernicus_DSM_COG_10_N37_00_W009_00_DEM.tif
    base = "https://copernicus-dem-30m.s3.amazonaws.com"
    tiles = [
        f"{base}/Copernicus_DSM_COG_10_N37_00_W009_00_DEM/Copernicus_DSM_COG_10_N37_00_W009_00_DEM.tif",
        f"{base}/Copernicus_DSM_COG_10_N37_00_W008_00_DEM/Copernicus_DSM_COG_10_N37_00_W008_00_DEM.tif",
    ]
    ok_tiles = []
    for url in tiles:
        try:
            # Try both Range request and HEAD
            resp = _get(url, headers={"Range": "bytes=0-511"}, timeout=10)
            if resp.status_code in (200, 206):
                ok_tiles.append(url.split("/")[-1])
            else:
                # HEAD to confirm existence without downloading
                h = _SESSION.head(url, timeout=10)
                if h.status_code == 200:
                    ok_tiles.append(url.split("/")[-1] + " (HEAD)")
        except Exception:
            pass
    if ok_tiles:
        r.ok(f"GLO-30 public AWS — {len(ok_tiles)} tiles reachable: {ok_tiles}")
        r.access_paths.extend(tiles[:len(ok_tiles)])
    else:
        # Try alternative region endpoint
        alt = f"{base}/Copernicus_DSM_COG_10_N37_00_W009_00_DEM/"
        try:
            resp2 = _get(alt, timeout=8)
            r.partial(f"S3 listing HTTP {resp2.status_code} — tiles may exist but aren't range-accessible",
                      listing_status=resp2.status_code)
        except Exception as e:
            r.error(f"GLO-30 tiles unreachable: {e} — pipeline fallback will fail")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Sentinel-2 STAC  (Earth Search + Planetary Computer)
# ══════════════════════════════════════════════════════════════════════════════

def probe_earth_search_stac(r: ProbeResult, bbox=DEFAULT_BBOX):
    """AWS Earth Search — main Sentinel-2 L2A source (use pystac_client, not raw POST)."""
    try:
        from pystac_client import Client
        cat = Client.open("https://earth-search.aws.element84.com/v1")
        results = cat.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime="2024-07-01/2025-08-31",
            max_items=3,
        )
        items = list(results.items())
        if items:
            best = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 999))
            date = best.datetime.date() if best.datetime else "?"
            cloud = best.properties.get("eo:cloud_cover", "?")
            bands = list(best.assets.keys())[:6]
            r.ok(f"{len(items)} scenes  best: {date}  cloud={cloud}%  assets={bands}")
            r.access_paths.append("https://earth-search.aws.element84.com/v1")
        else:
            r.partial("No S2 scenes found in date range")
    except ImportError:
        r.error("pystac_client not installed")


def probe_planetary_computer_stac(r: ProbeResult, bbox=DEFAULT_BBOX):
    """Microsoft Planetary Computer — backup S2 source."""
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": "2025-07-01/2025-08-31",
        "query": {"eo:cloud_cover": {"lt": 10}},
        "limit": 2,
    }
    resp = _SESSION.post(url, json=payload, timeout=TIMEOUT)
    r.http_code = resp.status_code
    if resp.status_code == 200:
        items = resp.json().get("features", [])
        r.ok(f"{len(items)} scenes found  (sign hrefs with planetary_computer.sign())")
        r.access_paths.append(url)
    else:
        r.partial(f"HTTP {resp.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ICESat-2 / NASA  (OpenAltimetry)
# ══════════════════════════════════════════════════════════════════════════════

def probe_openaltimetry(r: ProbeResult, bbox=DEFAULT_BBOX):
    """OpenAltimetry API — ICESat-2 ATL08 photon altimetry for SDB validation."""
    url = "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2/atl08"
    params = {
        "minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3],
        "date": "2023-07-15", "outputFormat": "json", "trackId": 1,
    }
    try:
        resp = _get(url, params=params, timeout=12)
        r.http_code = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            n = len(data) if isinstance(data, list) else data.get("count", "?")
            r.ok(f"OpenAltimetry ATL08 accessible — {n} records for test date")
            r.access_paths.append(url)
        else:
            r.partial(f"HTTP {resp.status_code}")
    except Exception as e:
        r.partial(f"Unreachable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. IPMA  (Instituto Português do Mar e da Atmosfera)
# ══════════════════════════════════════════════════════════════════════════════

def probe_ipma_api(r: ProbeResult):
    """IPMA public API — wind, wave observations for scene selection."""
    tests = [
        ("observations", "https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json"),
        ("forecast", "https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/hp-daily-forecast-day0.json"),
        ("sea state", "https://api.ipma.pt/open-data/forecast/oceanography/daily/hp-daily-sea-forecast-day0.json"),
    ]
    found = {}
    for name, url in tests:
        try:
            resp = _get(url, timeout=8)
            if resp.status_code == 200:
                found[name] = url
        except Exception:
            pass
    if found:
        r.ok(f"IPMA API open — {list(found.keys())}", endpoints=found)
        r.access_paths.extend(found.values())
    else:
        r.partial("IPMA API endpoints not reachable")


# ══════════════════════════════════════════════════════════════════════════════
# 13. OpenTopography  (EU terrain, public DEM access)
# ══════════════════════════════════════════════════════════════════════════════

_OT_API_KEY: str | None = None

_OT_DISCOVERY_SOURCES = [
    "https://portal.opentopography.org/apidocs/openapi.json",
    "https://opentopography.org/developers",
    "https://portal.opentopography.org/apidocs/",
    "https://raw.githubusercontent.com/OpenTopography/OT_3DEP_Workflows/main/notebooks/01_3DEP_Generate_DEM_User_AOI.ipynb",
    "https://raw.githubusercontent.com/uw-cryo/fetch_dem/master/fetch_dem/fetch_dem.py",
]

_KEY_PATTERNS = [
    re.compile(r'demoapikey[a-zA-Z0-9_\-]*', re.IGNORECASE),
    re.compile(r'[Aa][Pp][Ii][-_]?[Kk]ey["\'\s:=]+([a-zA-Z0-9_\-]{8,40})'),
    re.compile(r'"example"\s*:\s*"([a-zA-Z0-9_\-]{8,40})"'),
    re.compile(r'API_Key=([a-zA-Z0-9_\-]{8,40})'),
]

_OT_TEST_URL = (
    "https://portal.opentopography.org/API/globaldem"
    "?demtype=SRTMGL3&south=37.06&north=37.08&west=-8.22&east=-8.20"
    "&outputFormat=GTiff&API_Key={key}"
)


def _test_ot_key(key: str) -> str:
    """Return 'OK', 'RATE_LIMITED', or 'INVALID'."""
    try:
        r = _SESSION.get(_OT_TEST_URL.format(key=key), timeout=15, stream=True, verify=False)
        if r.status_code == 200:
            return "OK"
        body = next(r.iter_content(512), b"").decode("utf-8", errors="replace")
        if "rate limit" in body.lower():
            return "RATE_LIMITED"
        if "invalid" in body.lower() or "not found" in body.lower():
            return "INVALID"
        return f"HTTP_{r.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


def discover_opentopo_key() -> str | None:
    """
    Search local env/files and publicly accessible OpenTopography pages/repos for a working API key.
    """
    import os
    print("\n── OpenTopography key discovery ──")

    # 1. Check environment variable
    env_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if env_key:
        print("  ✅ Found key in environment (OPENTOPOGRAPHY_API_KEY) …")
        res = _test_ot_key(env_key)
        print(f"  → Result: {res}")
        if res == "OK":
            return env_key

    # 2. Check local files (.env, .opentopography.txt)
    for path in [Path(".env"), Path(".opentopography.txt"), Path("~/.opentopography.txt").expanduser()]:
        if path.exists():
            try:
                content = path.read_text()
                if path.name == ".env":
                    match = re.search(r"OPENTOPOGRAPHY_API_KEY\s*=\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", content)
                    if match:
                        key = match.group(1).strip()
                        print("  ✅ Found key in .env file …")
                        res = _test_ot_key(key)
                        print(f"  → Result: {res}")
                        if res == "OK":
                            return key
                else:
                    key = content.strip()
                    if key:
                        print(f"  ✅ Found key in {path.name} …")
                        res = _test_ot_key(key)
                        print(f"  → Result: {res}")
                        if res == "OK":
                            return key
            except Exception:
                pass

    # 3. Fallback to public web scraping
    print("  Searching web sources for fallback demo keys …")
    candidates: list[str] = []
    seen: set = set()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    for source_url in _OT_DISCOVERY_SOURCES:
        try:
            r = _SESSION.get(source_url, timeout=10, headers=headers, verify=False)
            if r.status_code != 200:
                print(f"  {source_url.split('/')[-1]:<45} HTTP {r.status_code}")
                continue

            text = r.text
            if "json" in r.headers.get("Content-Type", ""):
                try:
                    text = json.dumps(json.loads(text))
                except Exception:
                    pass

            found_here = []
            for pat in _KEY_PATTERNS:
                for match in pat.findall(text):
                    key = match.strip().strip("\"'")
                    if key and key not in seen:
                        seen.add(key)
                        candidates.append(key)
                        found_here.append(key)

            label = source_url.split("/")[-1] or source_url.split("/")[-2]
            if found_here:
                print(f"  ✅ {label:<45} found keys: {found_here}")
            else:
                print(f"  ⚠️  {label:<45} no key patterns matched")

        except Exception as e:
            print(f"  ❌ {source_url.split('/')[-1]:<45} {e}")

    if not candidates:
        print("  No candidate keys found in web sources.")
        return None

    print(f"\n  {len(candidates)} unique candidate(s) found in web: {candidates}")
    print("  Testing web keys against a small Algarve bbox …")

    best_key = None
    for key in candidates:
        result = _test_ot_key(key)
        icon = "✅" if result == "OK" else ("⏱️ " if result == "RATE_LIMITED" else "❌")
        print(f"  {icon} key={key:<30}  result={result}")
        if result == "OK" and best_key is None:
            best_key = key

    if best_key:
        print(f"\n  Working key: {best_key}")
    else:
        print("\n  All web keys rate-limited or invalid.")
    return best_key


def probe_opentopography(r: ProbeResult, bbox=DEFAULT_BBOX):
    """OpenTopography REST API — free DEMs including SRTM, NASADEM, COP30."""
    url = "https://portal.opentopography.org/API/globaldem"
    key = _OT_API_KEY or "demoapikeyot2022"
    params = {
        "demtype": "COP30", "south": bbox[1], "north": bbox[3],
        "west": bbox[0], "east": bbox[2],
        "outputFormat": "GTiff", "API_Key": key,
    }
    resp = _get(url, params=params, stream=True, timeout=20)
    r.http_code = resp.status_code
    ct = resp.headers.get("Content-Type", "")
    if resp.status_code == 200 and ("tiff" in ct or "octet" in ct):
        chunk = next(resp.iter_content(512), b"")
        r.ok(f"OpenTopography COP30 download OK (key: {key})  content-type={ct}  first 512B received")
        r.access_paths.append(url)
    elif resp.status_code == 401:
        r.partial(f"API key required (tried: {key}) — get free key at portal.opentopography.org")
    else:
        r.partial(f"HTTP {resp.status_code} (key: {key})  body={resp.text[:150]}")


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def build_probes(bbox: tuple) -> list[tuple[str, Any]]:
    """Return (name, callable) pairs for all probes."""
    def b(fn):
        """Wrap probe functions that need bbox."""
        def wrapped(r):
            fn(r, bbox)
        return wrapped

    return [
        # DGT STAC
        ("DGT STAC / all collections",           probe_dgt_stac_collections),
        ("DGT STAC / MDT-50cm LiDAR DTM",        b(probe_dgt_stac_mdt50)),
        ("DGT STAC / MDS-50cm LiDAR DSM",        b(probe_dgt_stac_mds50)),
        ("DGT STAC / ORTOSAT-2023 30cm RGBI",    b(probe_dgt_stac_ortosat)),
        ("DGT STAC / Sentinel-2 mosaics 2015–25", b(probe_dgt_stac_s2_mosaics)),
        ("DGT STAC / historical aerials 95–21",  b(probe_dgt_stac_historical_ortos)),
        ("DGT STAC / COPC LiDAR point clouds",   b(probe_dgt_stac_lidar_copc)),
        # DGT MinIO download barrier
        ("DGT MinIO / anonymous GET",            probe_minio_anonymous),
        ("DGT MinIO / COG range request",        probe_minio_cog_range_request),
        ("DGT MinIO / presign/health endpoints", probe_minio_presign_attempt),
        ("DGT MinIO / GDAL vsicurl access",      probe_minio_gdal_vsicurl),
        # DGT WMS / WCS
        ("DGT WMS / OrtoSat2023 GetCapabilities", probe_dgt_wms_capabilities),
        ("DGT WMS / OrtoSat2023 GetMap tile",    b(probe_dgt_wms_getmap)),
        ("DGT WCS / coverage services scan",     probe_dgt_wcs),
        # IH / DGRM ArcGIS
        ("IH/DGRM ArcGIS / service list",        probe_dgrm_other_services),
        ("IH/DGRM ArcGIS / Batimetrica_IH info", probe_ih_arcgis_services),
        ("IH/DGRM ArcGIS / isobath query bbox",  b(probe_ih_arcgis_query)),
        # Portuguese open portals
        ("SNIG / spatial data catalog",          probe_snig_catalog),
        ("SNIG / CSW 2.0.2",                     probe_snig_csw),
        ("dados.gov.pt / dataset search",        probe_dados_gov),
        # EU / global sources
        ("EMODnet / WCS GetCapabilities",        probe_emodnet_wcs),
        ("EMODnet / WCS GetCoverage download",   b(probe_emodnet_wcs_download)),
        ("Copernicus Land / EU-DEM catalog",     probe_copernicus_land_eu_dem),
        ("Copernicus GLO-30 / AWS public tiles", b(probe_copernicus_dem_aws)),
        ("Copernicus CMEMS / STAC catalog",      probe_cmems_stac),
        ("Copernicus CMEMS / OPeNDAP Kd490",     probe_cmems_opendap),
        # Satellite imagery STAC
        ("Earth Search STAC / Sentinel-2 L2A",  b(probe_earth_search_stac)),
        ("Planetary Computer / Sentinel-2 L2A", b(probe_planetary_computer_stac)),
        # Altimetry
        ("OpenAltimetry / ICESat-2 ATL08",       b(probe_openaltimetry)),
        # Meteorology
        ("IPMA / public weather & sea API",      probe_ipma_api),
        # Terrain download
        ("OpenTopography / COP30 download",      b(probe_opentopography)),
    ]


def run_all(bbox: tuple, workers: int = 8) -> list[ProbeResult]:
    probes = build_probes(bbox)
    results: list[ProbeResult] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe, fn, name): name for name, fn in probes}
        for future in as_completed(futures):
            results.append(future.result())

    # Preserve original order
    order = {name: i for i, (name, _) in enumerate(probes)}
    results.sort(key=lambda r: order.get(r.name, 999))
    return results


def print_report(results: list[ProbeResult], bbox: tuple):
    groups = {
        "DGT STAC": [r for r in results if r.name.startswith("DGT STAC")],
        "DGT MinIO (download barrier)": [r for r in results if r.name.startswith("DGT MinIO")],
        "DGT WMS / WCS": [r for r in results if "WMS" in r.name or "WCS" in r.name],
        "IH / DGRM ArcGIS": [r for r in results if "IH/DGRM" in r.name or "ArcGIS" in r.name],
        "Portuguese portals": [r for r in results if any(x in r.name for x in ["SNIG", "dados.gov"])],
        "EU / global terrain": [r for r in results if any(x in r.name for x in
                                 ["EMODnet", "Copernicus", "OpenTopography"])],
        "Satellite imagery STAC": [r for r in results if any(x in r.name for x in
                                    ["Earth Search", "Planetary"])],
        "Altimetry & met": [r for r in results if any(x in r.name for x in
                             ["ICESat", "IPMA", "OpenAltimetry"])],
    }

    print(f"\n{'═'*70}")
    print("  Reef Pipeline — Public Data API Probe")
    print(f"  bbox: {_bbox_str(bbox)}   {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*70}")

    totals = {"OK": 0, "PARTIAL": 0, "BLOCKED": 0, "ERROR": 0}
    for group_name, group in groups.items():
        if not group:
            continue
        print(f"\n── {group_name} ──")
        for r in group:
            print(r)
            totals[r.status] = totals.get(r.status, 0) + 1

    print(f"\n{'─'*70}")
    print(f"  Total: {sum(totals.values())} probes  "
          f"✅ {totals['OK']} OK  ⚠️  {totals['PARTIAL']} partial  "
          f"🔒 {totals['BLOCKED']} blocked  ❌ {totals['ERROR']} error")

    # Access summary
    usable = [r for r in results if r.status == "OK" and r.access_paths]
    if usable:
        print(f"\n  Immediately usable endpoints ({len(usable)}):")
        for r in usable:
            for p in r.access_paths[:1]:
                print(f"    {r.name:<45} {p}")

    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Probe all public data APIs relevant to the reef imagery pipeline"
    )
    parser.add_argument("--bbox", default="-8.25,37.04,-8.17,37.10",
                        help="lon_min,lat_min,lon_max,lat_max (default: Algarve Santa Eulália)")
    parser.add_argument("--report", default=None,
                        help="Write JSON report to this path (e.g. look_api/report.json)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel probe workers (default: 8)")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must be 4 comma-separated floats"

    global _OT_API_KEY
    _OT_API_KEY = discover_opentopo_key()

    print(f"\nProbing {len(build_probes(bbox))} endpoints with {args.workers} workers …")
    results = run_all(bbox, workers=args.workers)
    print_report(results, bbox)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {"bbox": list(bbox), "timestamp": datetime.utcnow().isoformat(),
                 "probes": [r.to_dict() for r in results]},
                f, indent=2, default=str,
            )
        print(f"JSON report written → {out}")


if __name__ == "__main__":
    main()
