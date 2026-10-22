#!/usr/bin/env python3
"""
sensor_config.py — Multi-Sensor Configuration Registry
══════════════════════════════════════════════════════════════════════════════════
Centralized sensor definitions for satellite imagery processing.
Maps sensor name → bands, wavelengths, resolutions, DN scaling, ACOLITE config.

Supported sensors:
  - sentinel-2  (ESA, 10m, free)
  - pleiades-neo (Airbus, 0.3m/1.2m, commercial/ESA TPM)
  - spot-6/7    (Airbus, 1.5m/6m, commercial/ESA TPM)
  - worldview-3  (Maxar, 0.31m/1.24m, commercial)

Usage:
  from src.sensor_config import get_sensor, list_sensors, SENSOR_REGISTRY
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

Entitlement = Literal[
    "free",          # no key needed (Sentinel-2 L2A via Planetary Computer / CDSE)
    "wms",           # DGT private-network WMS tile clipping (no auth, e.g. OrtoSat2023)
    "stac_search",   # DGT STAC catalogue search (no auth, IP-restricted to DGT VM)
    "planet_search", # Planet Data API — requires PLANET_CLIENT_ID (OAuth) + DGT contract
    "planet_orders", # Planet Orders API — requires PLANET_CLIENT_ID + download entitlement
    "sh_process",    # Sentinel Hub Process/Catalog/Statistical API (OAuth) — same shape as CDSE
]

# Planet OAuth endpoint — same client_credentials grant as Sentinel Hub (sh_downloader.py).
# The DGT-Planet / ESA-TPM contract yields a *separate* client_id/client_secret pair,
# NOT an API key (PLANET_API_KEY is a personal/developer key with search-only access).
_PLANET_TOKEN_URL = "https://api.planet.com/oauth/token"
_PLANET_CLIENT_ID_ENV = "PLANET_CLIENT_ID"
_PLANET_CLIENT_SECRET_ENV = "PLANET_CLIENT_SECRET"
# Personal/developer API key — kept as a search-only fallback for low-volume STAC
# discovery; MUST NOT be used to authenticate download or order submissions.
_PLANET_API_KEY_ENV = "PLANET_API_KEY"

log = logging.getLogger(__name__)


@dataclass
class BandConfig:
    """Single band configuration."""

    name: str  # Asset key in STAC or local file
    wavelength_nm: int  # Center wavelength in nm
    resolution_m: float  # Ground sampling distance in meters
    description: str  # Human-readable name
    reef_priority: int  # 1=critical, 2=useful, 3=optional for reef ID
    aw: float = 0.0  # Pure-water absorption coefficient (Pope & Fry 1997)


@dataclass
class SensorConfig:
    """Complete sensor configuration."""

    name: str  # e.g. "sentinel-2"
    display_name: str  # e.g. "Sentinel-2 MSI"
    stac_collection: str  # STAC collection ID
    provider: str  # "microsoft_pc", "cdse", "airbus", "maxar"
    bands: dict[str, BandConfig]  # band_key → BandConfig
    dn_scale: float  # DN-to-reflectance divisor (e.g. 10000 for S2)
    acolite_sensor: str  # ACOLITE --sensor flag value
    crs: str  # Default CRS for processing
    min_wavelength_nm: int  # Shortest useful wavelength
    max_wavelength_nm: int  # Longest useful wavelength
    has_swir: bool  # Has SWIR bands (useful for cloud/water masking)
    has_red_edge: bool  # Has red-edge bands (useful for submerged vegetation)
    notes: str = ""
    # ── Entitlement / authentication model (see resolve_sensor_credentials) ──
    entitlement: Entitlement = "free"  # which auth/access model applies
    # OAuth client_id / client_secret env-var names when entitlement is
    # "planet_*" or "sh_process". Empty for free/wms/stac_search sensors.
    oauth_id_env: str = ""
    oauth_secret_env: str = ""
    oauth_token_url: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# BAND DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Pure water absorption coefficients (m⁻¹) — Pope & Fry 1997
# These are critical for Beer-Lambert depth estimation
AW_443 = 0.0145  # Coastal aerosol / Blue
AW_490 = 0.0190  # Blue
AW_510 = 0.0590  # Green (short)
AW_555 = 0.0612  # Green
AW_560 = 0.0612  # Green (S2)
AW_640 = 0.2890  # Red (short)
AW_665 = 0.4300  # Red
AW_705 = 0.6500  # Red Edge 1
AW_740 = 1.7700  # Red Edge 2
AW_783 = 2.3100  # Red Edge 3
AW_842 = 3.0600  # NIR
AW_865 = 3.5800  # Narrow NIR
AW_1610 = 85.0  # SWIR1
AW_2190 = 440.0  # SWIR2


# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

SENTINEL_2 = SensorConfig(
    name="sentinel-2",
    display_name="Sentinel-2 MSI",
    stac_collection="sentinel-2-l2a",
    provider="microsoft_pc",
    dn_scale=10000.0,
    acolite_sensor="S2",
    crs="EPSG:32629",
    min_wavelength_nm=443,
    max_wavelength_nm=2190,
    has_swir=True,
    has_red_edge=True,
    bands={
        "B01": BandConfig("B01", 443, 60, "Coastal Aerosol", 3, AW_443),
        "B02": BandConfig("B02", 490, 10, "Blue", 1, AW_490),
        "B03": BandConfig("B03", 560, 10, "Green", 1, AW_560),
        "B04": BandConfig("B04", 665, 10, "Red", 2, AW_665),
        "B05": BandConfig("B05", 705, 20, "Red Edge 1", 3, AW_705),
        "B06": BandConfig("B06", 740, 20, "Red Edge 2", 3, AW_740),
        "B07": BandConfig("B07", 783, 20, "Red Edge 3", 3, AW_783),
        "B08": BandConfig("B08", 842, 10, "NIR", 2, AW_842),
        "B8A": BandConfig("B8A", 865, 20, "Narrow NIR", 3, AW_865),
        "B11": BandConfig("B11", 1610, 20, "SWIR1", 3, AW_1610),
        "B12": BandConfig("B12", 2190, 20, "SWIR2", 3, AW_2190),
    },
    notes="Baseline sensor. 10m blue/green/red/NIR. Free via Copernicus/Planetary Computer.",
    entitlement="free",
)


PLEIADES_NEO = SensorConfig(
    name="pleiades-neo",
    display_name="Pléiades Neo (HR)",
    stac_collection="pleiades-neo",
    provider="airbus",
    dn_scale=10000.0,  # L2A reflectance product (same scaling as S2)
    acolite_sensor="PLEIADES",
    crs="EPSG:32629",
    min_wavelength_nm=400,
    max_wavelength_nm=1000,
    has_swir=False,
    has_red_edge=False,
    bands={
        "PAN": BandConfig("PAN", 520, 0.3, "Panchromatic", 2, 0.0),
        "B02": BandConfig("B02", 490, 1.2, "Blue", 1, AW_490),  # Aqua band
        "B03": BandConfig("B03", 560, 1.2, "Green", 1, AW_560),
        "B04": BandConfig("B04", 665, 1.2, "Red", 2, AW_665),
        "B08": BandConfig("B08", 842, 1.2, "NIR", 2, AW_842),
    },
    notes=(
        "30cm PAN / 1.2m MS. Best for shallow reef mapping (<10m). "
        "No SWIR or red-edge. Available via Airbus OneAtlas or ESA TPM. "
        "For deeper reef ID, B02 at 1.2m gives 8x better spatial detail than S2."
    ),
    entitlement="wms",  # default: DGT private-network WMS clipping of pre-staged OrtoSat2023
    # For full multispectral download, escalate to DGT HPC team for a Planet
    # OAuth client_id/client_secret (Planet Orders API), then change to
    # entitlement="planet_orders" + set oauth_id_env / oauth_secret_env.
    oauth_id_env="",
    oauth_secret_env="",
    oauth_token_url="",
)

SPOT_6_7 = SensorConfig(
    name="spot-6/7",
    display_name="SPOT 6/7 (HRG)",
    stac_collection="spot-6/7",
    provider="airbus",
    dn_scale=10000.0,
    acolite_sensor="SPOT",
    crs="EPSG:32629",
    min_wavelength_nm=450,
    max_wavelength_nm=1750,
    has_swir=True,
    has_red_edge=False,
    bands={
        "PAN": BandConfig("PAN", 620, 1.5, "Panchromatic", 2, 0.0),
        "B02": BandConfig("B02", 490, 6.0, "Blue", 1, AW_490),
        "B03": BandConfig("B03", 560, 6.0, "Green", 1, AW_560),
        "B04": BandConfig("B04", 665, 6.0, "Red", 2, AW_665),
        "B08": BandConfig("B08", 842, 6.0, "NIR", 2, AW_842),
        "B11": BandConfig("B11", 1610, 6.0, "SWIR1", 3, AW_1610),
    },
    notes="1.5m PAN / 6m MS. Good balance of resolution and coverage. SWIR helps cloud masking.",
    entitlement="stac_search",  # default: DGT STAC catalogue discovery only
)

WORLDVIEW_3 = SensorConfig(
    name="worldview-3",
    display_name="WorldView-3 (SWIR)",
    stac_collection="worldview-3",
    provider="maxar",
    dn_scale=10000.0,
    acolite_sensor="WV3",
    crs="EPSG:32629",
    min_wavelength_nm=400,
    max_wavelength_nm=2500,
    has_swir=True,
    has_red_edge=False,
    bands={
        "PAN": BandConfig("PAN", 620, 0.31, "Panchromatic", 2, 0.0),
        "B02": BandConfig("B02", 480, 1.24, "Coastal Blue", 1, AW_443),
        "B03": BandConfig("B03", 545, 1.24, "Green", 1, AW_555),
        "B04": BandConfig("B04", 660, 1.24, "Red", 2, AW_665),
        "B08": BandConfig("B08", 832, 1.24, "NIR", 2, AW_842),
        "B11": BandConfig("B11", 1210, 3.70, "SWIR1", 3, AW_1610),
        "B12": BandConfig("B12", 1570, 3.70, "SWIR2", 3, AW_2190),
        "B_S1": BandConfig("B_S1", 2245, 3.70, "SWIR-7", 3, AW_2190),
    },
    notes=(
        "0.31m PAN / 1.24m MS / 3.7m SWIR. Best commercial sensor for water. "
        "SWIR bands enable sunglint correction and shallow water detection."
    ),
    entitlement="stac_search",  # default: DGT STAC catalogue discovery only
)


# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

SENSOR_REGISTRY: dict[str, SensorConfig] = {
    "sentinel-2": SENTINEL_2,
    "pleiades-neo": PLEIADES_NEO,
    "spot-6/7": SPOT_6_7,
    "worldview-3": WORLDVIEW_3,
}


def get_sensor(name: str) -> SensorConfig:
    """Get sensor config by name. Raises ValueError if not found."""
    key = name.lower().strip()
    if key not in SENSOR_REGISTRY:
        available = ", ".join(SENSOR_REGISTRY.keys())
        raise ValueError(f"Unknown sensor '{name}'. Available: {available}")
    return SENSOR_REGISTRY[key]


def list_sensors() -> list[str]:
    """Return list of available sensor names."""
    return list(SENSOR_REGISTRY.keys())


def get_reef_bands(sensor_name: str, max_priority: int = 2) -> dict[str, BandConfig]:
    """Get bands useful for reef identification (priority <= max_priority)."""
    sensor = get_sensor(sensor_name)
    return {k: v for k, v in sensor.bands.items() if v.reef_priority <= max_priority}


def get_blue_band(sensor_name: str) -> BandConfig:
    """Get the primary blue (B02) band for a sensor — the only useful band for reef ID."""
    sensor = get_sensor(sensor_name)
    return sensor.bands["B02"]


def map_band_name(from_sensor: str, to_sensor: str, band_name: str) -> str | None:
    """
    Map a band name from one sensor to another by wavelength proximity.
    Returns None if no match within 50nm.
    """
    src = get_sensor(from_sensor)
    tgt = get_sensor(to_sensor)
    if band_name not in src.bands:
        return None
    src_wl = src.bands[band_name].wavelength_nm
    best_match = None
    best_diff = 999
    for tgt_name, tgt_band in tgt.bands.items():
        diff = abs(tgt_band.wavelength_nm - src_wl)
        if diff < best_diff and diff < 50:
            best_diff = diff
            best_match = tgt_name
    return best_match


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════


def print_sensor_comparison():
    """Print a comparison table of all sensors."""
    header = (
        f"\n{'Sensor':<16} {'Resolution':>10} {'Bands':>6} {'Blue(B02)':>10} "
        f"{'SWIR':>5} {'RedEdge':>7} {'Cost':>8} {'Entitlement':>14}"
    )
    print(header)
    print("-" * len(header))
    for _name, cfg in SENSOR_REGISTRY.items():
        blue_res = cfg.bands["B02"].resolution_m
        n_bands = len(cfg.bands)
        swir = "Yes" if cfg.has_swir else "No"
        red_edge = "Yes" if cfg.has_red_edge else "No"
        cost = "Free" if cfg.provider in ("microsoft_pc", "cdse") else "Commercial"
        row = (
            f"  {cfg.display_name:<14} {blue_res:>7.1f}m {n_bands:>5} "
            f"{blue_res:>7.1f}m {swir:>5} {red_edge:>7} {cost:>8} {cfg.entitlement:>14}"
        )
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Entitlement / credential resolution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResolvedCredentials:
    """What credential material a caller actually needs for a sensor."""

    sensor: str
    entitlement: Entitlement
    # OAuth client_credentials tuple, populated when entitlement is one of the
    # OAuth-backed modes.  None otherwise.
    oauth_client_id: str | None = None
    oauth_client_secret: str | None = None
    oauth_token_url: str | None = None
    # Personal/developer Planet API key — search-only.  Populated ONLY when
    # the sensor's entitlement is `planet_search` AND the user explicitly opts
    # in by also setting PLANET_API_KEY.  Never used as a fallback for
    # `planet_orders` or `sh_process`.
    planet_api_key: str | None = None
    # True when no credential material was found and the caller cannot proceed.
    missing: bool = False
    notes: str = ""


def resolve_sensor_credentials(sensor_name: str) -> ResolvedCredentials:
    """Resolve the credentials required to access *sensor_name*.

    The function enforces the entitlement declared on the SensorConfig entry:

      * `free`         — no credentials needed.
      * `wms`          — no credentials; caller must be on the DGT private
                         network (the WMS endpoint is IP-restricted).
      * `stac_search`  — no credentials; DGT STAC catalogue is IP-restricted.
      * `planet_search`— requires PLANET_CLIENT_ID + PLANET_CLIENT_SECRET
                         (DGT-Planet / ESA-TPM contract). PLANET_API_KEY is
                         accepted only if no OAuth pair is set, with a loud
                         warning, because the API key is personal/search-only.
      * `planet_orders`— requires PLANET_CLIENT_ID + PLANET_CLIENT_SECRET.
                         PLANET_API_KEY is NEVER accepted — refusing to
                         silently fall through to a personal key is the whole
                         point of this helper.
      * `sh_process`   — requires the sensor-specific OAuth pair configured
                         on the SensorConfig (e.g. CDSE_CLIENT_ID/_SECRET).

    Raises ValueError if *sensor_name* is unknown.
    Returns a ResolvedCredentials with ``missing=True`` when the entitlement
    requires credentials that are not set in the environment.
    """
    cfg = get_sensor(sensor_name)
    e = cfg.entitlement

    if e in ("free", "wms", "stac_search"):
        return ResolvedCredentials(
            sensor=sensor_name, entitlement=e,
            notes="No credentials required.",
        )

    if e == "planet_search":
        cid = os.environ.get(_PLANET_CLIENT_ID_ENV, "")
        csec = os.environ.get(_PLANET_CLIENT_SECRET_ENV, "")
        if cid and csec:
            return ResolvedCredentials(
                sensor=sensor_name, entitlement=e,
                oauth_client_id=cid, oauth_client_secret=csec,
                oauth_token_url=_PLANET_TOKEN_URL,
                notes="Planet OAuth (DGT-Planet / ESA-TPM contract).",
            )
        # Fallback to personal API key — search only. Loud warning.
        personal = os.environ.get(_PLANET_API_KEY_ENV, "")
        if personal:
            log.warning(
                "Sensor '%s' has entitlement='planet_search' but no PLANET_CLIENT_ID/"
                "PLANET_CLIENT_SECRET pair is set. Falling back to PLANET_API_KEY "
                "(personal/developer, search-only — DO NOT use for asset download or "
                "Orders). Ask the DGT HPC team for an institutional OAuth client.",
                sensor_name,
            )
            return ResolvedCredentials(
                sensor=sensor_name, entitlement=e,
                planet_api_key=personal,
                notes="Personal Planet API key — search only, not for Orders.",
            )
        return ResolvedCredentials(
            sensor=sensor_name, entitlement=e, missing=True,
            notes=(
                "Planet OAuth credentials not set. Required for planet_search: "
                f"{_PLANET_CLIENT_ID_ENV} + {_PLANET_CLIENT_SECRET_ENV} "
                f"(preferred) or {_PLANET_API_KEY_ENV} (search-only fallback)."
            ),
        )

    if e == "planet_orders":
        cid = os.environ.get(_PLANET_CLIENT_ID_ENV, "")
        csec = os.environ.get(_PLANET_CLIENT_SECRET_ENV, "")
        if not (cid and csec):
            # CRITICAL: do NOT accept PLANET_API_KEY for orders. The personal key
            # would silently fail mid-pipeline or — worse — bill the developer's
            # account. Refuse loudly instead.
            return ResolvedCredentials(
                sensor=sensor_name, entitlement=e, missing=True,
                notes=(
                    f"Planet Orders requires OAuth ({_PLANET_CLIENT_ID_ENV} + "
                    f"{_PLANET_CLIENT_SECRET_ENV}). PLANET_API_KEY is NEVER accepted "
                    "for planet_orders — it is a personal/developer key without "
                    "download entitlement. Ask the DGT HPC team for an institutional "
                    "OAuth client issued under the DGT-Planet / ESA-TPM contract."
                ),
            )
        return ResolvedCredentials(
            sensor=sensor_name, entitlement=e,
            oauth_client_id=cid, oauth_client_secret=csec,
            oauth_token_url=_PLANET_TOKEN_URL,
            notes="Planet OAuth Orders (DGT-Planet / ESA-TPM contract).",
        )

    if e == "sh_process":
        if not (cfg.oauth_id_env and cfg.oauth_secret_env and cfg.oauth_token_url):
            return ResolvedCredentials(
                sensor=sensor_name, entitlement=e, missing=True,
                notes=(
                    f"Sensor '{sensor_name}' declares entitlement='sh_process' but its "
                    "SensorConfig is missing oauth_id_env / oauth_secret_env / "
                    "oauth_token_url. Configure them before use."
                ),
            )
        cid = os.environ.get(cfg.oauth_id_env, "")
        csec = os.environ.get(cfg.oauth_secret_env, "")
        if not (cid and csec):
            return ResolvedCredentials(
                sensor=sensor_name, entitlement=e, missing=True,
                notes=(
                    f"Sentinel Hub OAuth credentials not set. Required: "
                    f"{cfg.oauth_id_env} + {cfg.oauth_secret_env}."
                ),
            )
        return ResolvedCredentials(
            sensor=sensor_name, entitlement=e,
            oauth_client_id=cid, oauth_client_secret=csec,
            oauth_token_url=cfg.oauth_token_url,
            notes="Sentinel Hub OAuth (CDSE / commercial).",
        )

    # pragma: no cover — exhaustive Literal
    return ResolvedCredentials(sensor=sensor_name, entitlement=e, missing=True,
                               notes=f"Unknown entitlement: {e!r}")


if __name__ == "__main__":
    print_sensor_comparison()
