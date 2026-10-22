"""
ipma_sea_state.py — IPMA sea-state filter for Sentinel-2 scene selection.

Fetches wave height and wind speed from the IPMA public oceanography API
and exposes two functions:

    is_scene_usable(date, max_wave_m, max_wind_ms) -> bool
    get_conditions(date) -> dict

The sea-state check is wired into the orchestrator scene selection so that
acquisitions during storms are skipped with a logged reason, not silently
dropped.

Data source:
    https://api.ipma.pt/open-data/forecast/oceanography/daily/
    https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json

No credentials required — IPMA API is fully public.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache

import requests

log = logging.getLogger(__name__)

_WAVE_URL = "https://api.ipma.pt/open-data/forecast/oceanography/daily/" "hp-daily-sea-forecast-day0.json"
_WIND_URL = "https://api.ipma.pt/open-data/observation/meteorology/stations/" "observations.json"
_TIMEOUT = 10  # seconds

# Algarve station IDs confirmed from live API (lat ~37N, lon ~-8 to -9W)
# globalIdLocal values for the three nearest marine stations
_ALGARVE_STATION_IDS = {1080526, 1081526}  # Faro/Sagres area

# Defaults matching IPMA Beaufort scale dive-condition thresholds
_DEFAULT_MAX_WAVE_M = 1.5  # metres (significant wave height)
_DEFAULT_MAX_WIND_MS = 10.0  # m/s  (~Beaufort 5)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, returning default for None, NaN, or unparseable strings."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if f == f else default  # f != f is True only for NaN
    except (ValueError, TypeError):
        return default


@lru_cache(maxsize=4)
def _fetch_wave_forecast() -> list[dict]:
    """Fetch today's sea-state forecast. Cached per process run."""
    try:
        r = requests.get(_WAVE_URL, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        log.warning("IPMA wave forecast fetch failed: %s", e)
        return []


@lru_cache(maxsize=4)
def _fetch_wind_obs() -> dict:
    """Fetch latest wind observations. Cached per process run."""
    try:
        r = requests.get(_WIND_URL, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("IPMA wind observation fetch failed: %s", e)
        return {}


def _algarve_wave_conditions() -> dict:
    """
    Return the worst-case wave conditions among the Algarve marine stations.

    Returns a dict with keys:
        wave_high_max_m, wave_high_min_m, total_sea_max_m,
        sst_max_c, pred_wave_dir, station_id
    or an empty dict if no Algarve stations are found.
    """
    stations = _fetch_wave_forecast()
    algarve = [
        s
        for s in stations
        if s.get("globalIdLocal") in _ALGARVE_STATION_IDS
        or (
            s.get("latitude") is not None
            and s.get("longitude") is not None
            and 36.5 <= float(s["latitude"]) <= 37.6
            and -9.5 <= float(s["longitude"]) <= -7.0
        )
    ]
    if not algarve:
        log.debug("No Algarve marine stations in IPMA forecast.")
        return {}

    # Return worst-case (highest wave) across stations
    worst = max(algarve, key=lambda s: _safe_float(s.get("waveHighMax")))
    return {
        "wave_high_max_m": _safe_float(worst.get("waveHighMax")),
        "wave_high_min_m": _safe_float(worst.get("waveHighMin")),
        "total_sea_max_m": _safe_float(worst.get("totalSeaMax")),
        "wave_period_max_s": _safe_float(worst.get("wavePeriodMax")),
        "pred_wave_dir": worst.get("predWaveDir", "?"),
        "sst_max_c": _safe_float(worst.get("sstMax")),
        "station_id": worst.get("globalIdLocal"),
        "station_lat": _safe_float(worst.get("latitude")),
        "station_lon": _safe_float(worst.get("longitude")),
    }


def _algarve_wind_ms() -> float | None:
    """
    Return the maximum wind speed in m/s from the most recent observation
    at Algarve-area meteorological stations only.

    Filters by _ALGARVE_STATION_IDS so that inland or northern-Portugal stations
    (e.g. mountain sites with extreme wind) cannot reject an Algarve scene.
    Returns None if no Algarve station data is available.
    """
    obs_by_time = _fetch_wind_obs()
    if not obs_by_time:
        return None

    latest_ts = sorted(obs_by_time.keys())[-1]
    stations = obs_by_time[latest_ts]

    speeds = []
    for sid, s in stations.items():
        try:
            if int(sid) not in _ALGARVE_STATION_IDS:
                continue
        except (ValueError, TypeError):
            continue
        if s.get("intensidadeVento") is not None:
            speeds.append(float(s["intensidadeVento"]))
    return max(speeds) if speeds else None


def get_conditions(date: str | None = None) -> dict:
    """
    Return current sea-state conditions for the Algarve coast.

    Args:
        date: ISO date string (YYYY-MM-DD). Currently ignored — IPMA only
              serves day-0 forecast. Included for future multi-day support.

    Returns:
        dict with keys:
            wave_high_max_m   — significant wave height maximum (m)
            total_sea_max_m   — total sea height (m)
            pred_wave_dir     — predicted wave direction
            wind_speed_ms     — wind speed (m/s), or None if unavailable
            sst_max_c         — sea surface temperature max (°C)
            source            — "IPMA day-0 forecast"
            usable            — bool: conditions within default thresholds
    """
    wave = _algarve_wave_conditions()
    wind = _algarve_wind_ms()

    result = {
        "wave_high_max_m": wave.get("wave_high_max_m"),
        "total_sea_max_m": wave.get("total_sea_max_m"),
        "wave_period_max_s": wave.get("wave_period_max_s"),
        "pred_wave_dir": wave.get("pred_wave_dir"),
        "sst_max_c": wave.get("sst_max_c"),
        "wind_speed_ms": wind,
        "source": "IPMA day-0 forecast",
        "fetch_date": datetime.now(UTC).date().isoformat(),
    }

    # Derive usability under default thresholds
    wave_ok = _safe_float(wave.get("wave_high_max_m")) <= _DEFAULT_MAX_WAVE_M
    wind_ok = wind is None or wind <= _DEFAULT_MAX_WIND_MS
    result["usable"] = wave_ok and wind_ok

    return result


def is_scene_usable(
    date: str | None = None,
    max_wave_m: float = _DEFAULT_MAX_WAVE_M,
    max_wind_ms: float = _DEFAULT_MAX_WIND_MS,
) -> bool:
    """
    Return True if sea conditions are acceptable for a usable Sentinel-2 scene.

    A scene is rejected when significant wave height exceeds max_wave_m OR
    wind speed exceeds max_wind_ms. If IPMA is unreachable, the scene is
    accepted (fail-open) and a warning is logged.

    Args:
        date:        ISO date string (YYYY-MM-DD). Used for logging only.
        max_wave_m:  reject scenes with wave height above this (metres).
        max_wind_ms: reject scenes with wind speed above this (m/s).

    Returns:
        True  → scene acceptable
        False → scene rejected (logged with reason)
    """
    wave = _algarve_wave_conditions()
    wind = _algarve_wind_ms()

    if not wave:
        log.warning("IPMA sea-state data unavailable for %s — accepting scene (fail-open).", date)
        return True

    wave_h = wave.get("wave_high_max_m", 0.0)
    if wave_h > max_wave_m:
        log.info(
            "Scene %s SKIPPED — wave height %.1f m > %.1f m threshold " "(dir=%s, station=%s)",
            date,
            wave_h,
            max_wave_m,
            wave.get("pred_wave_dir", "?"),
            wave.get("station_id"),
        )
        return False

    if wind is not None and wind > max_wind_ms:
        log.info(
            "Scene %s SKIPPED — wind speed %.1f m/s > %.1f m/s threshold",
            date,
            wind,
            max_wind_ms,
        )
        return False

    log.debug(
        "Scene %s ACCEPTED — wave %.1f m, wind %s m/s",
        date,
        wave_h,
        f"{wind:.1f}" if wind is not None else "n/a",
    )
    return True
