#!/usr/bin/env python3
"""
forecast_visibility.py — Underwater visibility forecast for Algarve reef sites.

Combines the Open-Meteo Marine wave forecast with the project's seasonal Kd490
table and each site's coastal orientation:

  calm visibility  = 1.7 / Kd490(month) / 2           (Poole-Atkins, horizontal)
  resuspension     = near-bed orbital velocity at depth (linear wave theory),
                     with wave height reduced by the angle between swell and coast
  memory           = sediment stirred up on previous days settles at 50 %/day
  rain             = IPMA Faro precipitation probability -> runoff penalty on the
                     rainy day and the day after

This is a physics-based estimate, not validated against diver observations.

    python scripts/forecast_visibility.py               # today + next 2 days
    python scripts/forecast_visibility.py --days 3 --depth 18
    python scripts/forecast_visibility.py --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    KD490_DEFAULT,
    KD490_TABLE,
    VIS_FORECAST_DEPTH_M,
    VIS_HORIZONTAL_FACTOR,
    VIS_POOLE_ATKINS,
    VIS_RAIN_PENALTY_COAST,
    VIS_RAIN_PENALTY_RIA,
    VIS_RAIN_PROB_THRESHOLD,
    VIS_RESUSPENSION_SENSITIVITY,
    VIS_RESUSPENSION_U_CRIT,
    VIS_TURBIDITY_MEMORY,
)

SITES_CSV = ROOT / "outputs" / "coastal_topography" / "algarve_coastal_features.csv"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
IPMA_FARO_URL = "https://api.ipma.pt/open-data/forecast/meteorology/cities/daily/1080500.json"
PAST_DAYS = 2

# Open-Meteo's nearest grid cell for these sites is half on land and damps the waves
OFFSHORE_POINTS = {
    "tavira_west": (37.04, -7.66),
    "ilha_de_tavira": (37.04, -7.70),
    "olhos_de_agua": (37.04, -8.16),
    "salgados": (37.04, -8.30),
}
RIA_FORMOSA_SITES = {"tavira_west", "ilha_de_tavira", "fuseta", "olhao_offshore", "faro_east", "praia_de_faro"}
DAYTIME_HOURS = range(8, 19)


def wavenumber(period_s: float, depth_m: float) -> float:
    """Solve the linear dispersion relation w^2 = g k tanh(k h) by fixed-point iteration."""
    omega = 2 * math.pi / period_s
    k = omega * omega / 9.81
    for _ in range(100):
        k = omega * omega / (9.81 * math.tanh(k * depth_m))
    return k


def near_bed_orbital_velocity(height_m: float, period_s: float, depth_m: float) -> float:
    """Maximum near-bed orbital velocity (m/s) from linear wave theory."""
    if height_m <= 0 or period_s <= 0:
        return 0.0
    k = wavenumber(period_s, depth_m)
    return math.pi * height_m / (period_s * math.sinh(k * depth_m))


def exposure_factor(wave_dir_deg: float, coast_aspect_deg: float) -> float:
    """Wave height multiplier for swell arriving at an angle to the coast's facing direction."""
    diff = abs((wave_dir_deg - coast_aspect_deg + 180) % 360 - 180)
    return max(0.3, math.cos(math.radians(min(diff, 90.0)))) ** 0.5


def visibility_label(vis_m: float) -> str:
    if vis_m >= 12:
        return "Excelente"
    if vis_m >= 8:
        return "Boa"
    if vis_m >= 5:
        return "Razoável"
    return "Fraca"


def calm_visibility(month: int) -> float:
    return VIS_POOLE_ATKINS / KD490_TABLE.get(month, KD490_DEFAULT) * VIS_HORIZONTAL_FACTOR


def forecast_site(
    hourly: dict,
    coast_aspect_deg: float,
    rain_prob_by_day: dict[str, float],
    near_ria: bool,
    depth_m: float = VIS_FORECAST_DEPTH_M,
) -> dict[str, dict]:
    """Return {YYYY-MM-DD: {u_bed, wave_height_max, visibility_m, label}} for every day in `hourly`."""
    times = hourly["time"]
    result: dict[str, dict] = {}
    past_excess: list[float] = []
    for day in sorted({t[:10] for t in times}):
        idx = [
            i
            for i, t in enumerate(times)
            if t.startswith(day) and int(t[11:13]) in DAYTIME_HOURS and hourly["wave_height"][i] is not None
        ]
        u_max = max(
            (
                near_bed_orbital_velocity(
                    hourly["wave_height"][i] * exposure_factor(hourly["wave_direction"][i], coast_aspect_deg),
                    hourly["wave_period"][i],
                    depth_m,
                )
                for i in idx
            ),
            default=0.0,
        )
        excess = max(0.0, u_max / VIS_RESUSPENSION_U_CRIT - 1)
        carried = [e * VIS_TURBIDITY_MEMORY ** (n + 1) for n, e in enumerate(reversed(past_excess))]
        effective = max([excess, *carried])
        past_excess.append(excess)

        vis = calm_visibility(int(day[5:7])) / (1 + VIS_RESUSPENSION_SENSITIVITY * effective)
        previous_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        # runoff from ribeiras and the Ria Formosa keeps the water turbid into the day after the rain
        if max(rain_prob_by_day.get(day, 0), rain_prob_by_day.get(previous_day, 0)) >= VIS_RAIN_PROB_THRESHOLD:
            vis *= VIS_RAIN_PENALTY_RIA if near_ria else VIS_RAIN_PENALTY_COAST
        vis = round(vis, 1)

        result[day] = {
            "u_bed_ms": round(u_max, 3),
            "wave_height_max_m": round(max((hourly["wave_height"][i] for i in idx), default=0.0), 2),
            "visibility_m": vis,
            "label": visibility_label(vis),
        }
    return result


def load_sites(csv_path: Path = SITES_CSV) -> list[dict]:
    """Sites from the coastal features CSV, skipping duplicates of the same location."""
    sites, seen = [], set()
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (round(float(row["latitude"]), 3), round(float(row["longitude"]), 3))
            if key in seen:
                continue
            seen.add(key)
            lat, lon = OFFSHORE_POINTS.get(row["site_name"], (float(row["latitude"]), float(row["longitude"])))
            sites.append({"name": row["site_name"], "lat": lat, "lon": lon, "aspect": float(row["aspect_mean"])})
    return sites


def _get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    for attempt in range(4):
        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_marine(session: requests.Session, lat: float, lon: float, days: int) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wave_direction,wave_period",
        "timezone": "Europe/Lisbon",
        "forecast_days": days + 1,
        "past_days": PAST_DAYS,
    }
    return _get_json(session, MARINE_URL, params)["hourly"]


def fetch_rain_probability(session: requests.Session) -> dict[str, float]:
    try:
        data = _get_json(session, IPMA_FARO_URL)
        return {d["forecastDate"]: float(d.get("precipitaProb", 0)) for d in data["data"]}
    except (requests.RequestException, KeyError, ValueError) as err:
        print(f"WARNING: IPMA rain forecast unavailable ({err}); no rain penalty applied", file=sys.stderr)
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=2, help="Days ahead to forecast, besides today (default: 2)")
    ap.add_argument("--depth", type=float, default=VIS_FORECAST_DEPTH_M, help="Reef depth in metres (default: 12)")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = ap.parse_args()

    today = date.today()
    wanted = [(today + timedelta(days=n)).isoformat() for n in range(args.days + 1)]

    with requests.Session() as session:
        rain = fetch_rain_probability(session)
        forecast = {}
        for site in load_sites():
            hourly = fetch_marine(session, site["lat"], site["lon"], args.days)
            days = forecast_site(hourly, site["aspect"], rain, site["name"] in RIA_FORMOSA_SITES, args.depth)
            forecast[site["name"]] = {d: days[d] for d in wanted if d in days}

    if args.json:
        print(json.dumps({"depth_m": args.depth, "rain_probability": rain, "sites": forecast}, ensure_ascii=False, indent=1))
        return

    print(f"Horizontal visibility at ~{args.depth:g} m (estimate, ±25 %)\n")
    print(f"{'site':20}" + "".join(f"{d[5:]:>22}" for d in wanted))
    for name, days in forecast.items():
        cells = "".join(f"{days[d]['visibility_m']:>9.1f} m {days[d]['label']:<10}" for d in wanted if d in days)
        print(f"{name:20}{cells}")
    print("\nRain probability (IPMA Faro): " + ", ".join(f"{d[5:]} {rain.get(d, 0):.0f}%" for d in wanted))


if __name__ == "__main__":
    main()
