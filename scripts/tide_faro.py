#!/usr/bin/env python3
"""
tide_faro.py

Provides accurate, real-time tide predictions for Barra de Faro-Olhão, Portugal,
by fetching data from authoritative external sources.
"""

# 1. Imports and Constants
import argparse
import logging
import math
from datetime import datetime, timedelta, date as date_obj

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- Constants ---

PORT_MAP = {
    "faro-olhao": "11"
}
IH_URL = "https://www.hidrografico.pt/previsoes-mares"
OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"
FARO_COORDS = {"lat": 37.0147, "lon": -7.9753}

# --- Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tide_faro")

# --- Custom Exception ---

class DataSourceError(Exception):
    """Custom exception for when all data sources fail."""
    pass


# 2. Data Source Clients

def _fetch_from_ih(date_str: str, port_code: str) -> list | None:
    """
    Fetches tide data by scraping the Instituto Hidrográfico (IH) website.
    This is the preferred, most authoritative source.
    """
    log.info(f"Attempting to fetch data from Instituto Hidrográfico for {date_str}...")
    try:
        payload = {"iEstacao": port_code, "data": date_str}
        log.debug(f"POST request to {IH_URL} with payload: {payload}")
        response = requests.post(IH_URL, data=payload, timeout=10)
        response.raise_for_status()
        log.info(f"IH response status: {response.status_code}")

        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table", class_="table-previsoes-mares")

        if not table:
            log.warning("IH: Tide table not found in response HTML.")
            return None

        events = []
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            event_type_raw = cols[0].text.strip()
            time_str = cols[1].text.strip()
            height_str = cols[2].text.strip().split()[0].replace(",", ".")

            if "Preia-Mar" in event_type_raw:
                event_type = "high"
            elif "Baixa-Mar" in event_type_raw:
                event_type = "low"
            else:
                continue

            events.append({
                "type": event_type,
                "time": time_str,
                "height_m": float(height_str)
            })

        if not events:
            log.warning("IH: Found table but could not parse any tide events.")
            return None

        return events

    except requests.RequestException as e:
        log.error(f"IH fetch failed: {e}")
        return None

def _fetch_from_cmems(date_str: str) -> list | None:
    """
    Placeholder for fetching data from Copernicus Marine Service (CMEMS).
    Requires user-specific credentials and client setup.
    """
    log.warning("CMEMS client not implemented. Please add credentials and logic. Skipping.")
    return None

def _fetch_from_open_meteo(date_str: str) -> list | None:
    """
    Fetches tide data from Open-Meteo as a last resort.
    This source is generally less accurate than official hydrographic services.
    """
    log.info(f"Attempting to fetch data from Open-Meteo for {date_str}...")
    try:
        params = {
            "latitude": FARO_COORDS["lat"],
            "longitude": FARO_COORDS["lon"],
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "sea_level",
        }
        log.debug(f"GET request to {OPEN_METEO_URL} with params: {params}")
        response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        response.raise_for_status()
        log.info(f"Open-Meteo response status: {response.status_code}")

        data = response.json()
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "height": data["hourly"]["sea_level"]
        })
        df.dropna(inplace=True)

        # Find peaks (high tide) and troughs (low tide)
        df['is_peak'] = (df['height'] > df['height'].shift(1)) & (df['height'] > df['height'].shift(-1))
        df['is_trough'] = (df['height'] < df['height'].shift(1)) & (df['height'] < df['height'].shift(-1))

        peaks = df[df['is_peak']]
        troughs = df[df['is_trough']]

        events = []
        for _, row in peaks.iterrows():
            events.append({"type": "high", "time": row['time'].strftime("%H:%M"), "height_m": round(row['height'], 2)})
        for _, row in troughs.iterrows():
            events.append({"type": "low", "time": row['time'].strftime("%H:%M"), "height_m": round(row['height'], 2)})

        # Sort by time
        events.sort(key=lambda x: datetime.strptime(x['time'], "%H:%M"))
        return events

    except requests.RequestException as e:
        log.error(f"Open-Meteo fetch failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        log.error(f"Open-Meteo data parsing failed: {e}")
        return None


# 3. Core Logic

def get_tide_schedule(date: str, port: str = "faro-olhao") -> dict:
    """
    Returns tide events for a given date by trying sources in priority order.

    Raises:
        DataSourceError: If no data source is reachable or provides data.
    """
    port_code = PORT_MAP.get(port)
    if not port_code:
        raise ValueError(f"Unknown port: {port}. Available ports: {list(PORT_MAP.keys())}")

    # --- Priority 1: Instituto Hidrográfico ---
    events = _fetch_from_ih(date, port_code)
    if events:
        return {"port": port, "date": date, "source": "Instituto Hidrográfico", "events": events}

    # --- Priority 2: CMEMS (Placeholder) ---
    events = _fetch_from_cmems(date)
    if events:
        return {"port": port, "date": date, "source": "CMEMS", "events": events}

    # --- Priority 3: Open-Meteo ---
    events = _fetch_from_open_meteo(date)
    if events:
        return {"port": port, "date": date, "source": "Open-Meteo", "events": events}

    raise DataSourceError(f"Could not fetch tide data for {date} from any source.")


def get_tide_height(date: str, time: str, port: str = "faro-olhao") -> dict:
    """
    Interpolates tide height at a specific moment using cosine interpolation.
    """
    target_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")

    # Fetch schedule for today and tomorrow to ensure we have surrounding events
    schedule_today = get_tide_schedule(date, port)
    
    tomorrow_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_tomorrow = get_tide_schedule(tomorrow_date, port)

    all_events = schedule_today["events"] + schedule_tomorrow["events"]

    # Add full datetime objects for sorting
    for i, event in enumerate(all_events):
        event_dt = datetime.strptime(f"{date} {event['time']}", "%Y-%m-%d %H:%M")
        if i >= len(schedule_today["events"]):
            event_dt += timedelta(days=1)
        event["datetime"] = event_dt

    all_events.sort(key=lambda x: x["datetime"])

    # Find surrounding events
    prev_event = next((e for e in reversed(all_events) if e["datetime"] <= target_dt), None)
    next_event = next((e for e in all_events if e["datetime"] > target_dt), None)

    if not prev_event or not next_event:
        raise ValueError("Could not find surrounding tide events for interpolation.")

    # Cosine Interpolation
    t0, h0 = prev_event["datetime"], prev_event["height_m"]
    t1, h1 = next_event["datetime"], next_event["height_m"]

    total_seconds = (t1 - t0).total_seconds()
    elapsed_seconds = (target_dt - t0).total_seconds()

    if total_seconds == 0:
        interpolated_height = h0
    else:
        phase = math.pi * (elapsed_seconds / total_seconds)
        interpolated_height = h0 + (h1 - h0) * ((1 - math.cos(phase)) / 2)

    trend = "rising" if h1 > h0 else "falling"

    return {
        "height_m": round(interpolated_height, 2),
        "trend": trend,
        "source": schedule_today["source"],
        "next_event": {
            "type": next_event["type"],
            "time": next_event["time"],
            "height_m": next_event["height_m"]
        }
    }


# 4. CLI Entry Point

def main():
    """
    Command-line interface for the tide prediction module.
    """
    parser = argparse.ArgumentParser(
        description="Fetch tide schedule for Barra de Faro-Olhão."
    )
    parser.add_argument(
        "--date",
        help="Date to query in YYYY-MM-DD format. Defaults to today."
    )
    parser.add_argument(
        "--time",
        help="Time to query in HH:MM format. If provided, interpolates height."
    )
    args = parser.parse_args()

    query_date = args.date or date_obj.today().strftime("%Y-%m-%d")

    try:
        if args.time:
            log.info(f"Querying interpolated height for {query_date} at {args.time}...")
            height_data = get_tide_height(query_date, args.time)
            print("\n--- Tide Height Prediction ---")
            print(f"  Date & Time: {query_date} {args.time}")
            print(f"  Source: {height_data['source']}")
            print(f"  Predicted Height: {height_data['height_m']:.2f} m")
            print(f"  Trend: {height_data['trend'].capitalize()}")
            next_e = height_data['next_event']
            print(f"  Next Event: {next_e['type'].capitalize()} tide at {next_e['time']} ({next_e['height_m']:.2f} m)")
            print("----------------------------")

        else:
            log.info(f"Querying tide schedule for {query_date}...")
            schedule = get_tide_schedule(query_date)
            print(f"\n--- Tide Schedule for {schedule['port']} on {schedule['date']} ---")
            print(f"(Data from: {schedule['source']})")
            for event in schedule["events"]:
                print(f"  - {event['type'].capitalize()} Tide: {event['time']}  ({event['height_m']:.2f} m)")
            print("--------------------------------------------------")

    except (DataSourceError, ValueError) as e:
        log.error(f"Operation failed: {e}")

if __name__ == "__main__":
    main()