"""
cmems_client.py — Copernicus Marine (CMEMS) ocean-colour water-clarity client.

Pulls daily 1 km Atlantic ocean-colour observations for the Algarve reef domain
and feeds three workstreams:

  * Underwater visibility  — ZSD (Secchi depth), KD490, SPM (turbidity)
  * Algae / phytoplankton  — CHL (chlorophyll-a)
  * SDB Kd cross-check     — KD490 vs the QAA-derived Kd from Sentinel-2

Authentication
--------------
Uses the saved `copernicusmarine` credentials (default
``~/.copernicusmarine/.copernicusmarine-credentials``). No secrets in code.
Verify the stored login with::

    copernicusmarine login --check-credentials-valid

Datasets (Atlantic ocean-colour, L3, multi-sensor, 1 km, daily)
---------------------------------------------------------------
    transparency : cmems_obs-oc_atl_bgc-transp_*_l3-multi-1km_P1D   (KD490, ZSD, SPM)
    plankton     : cmems_obs-oc_atl_bgc-plankton_*_l3-multi-1km_P1D (CHL)

The reprocessed multi-year series (``my``) lags ~3-4 weeks behind real time; the
near-real-time series (``nrt``) covers the recent window. ``stream="auto"``
(default) picks ``nrt`` when the requested end date is within MY_LAG_DAYS of
today, otherwise ``my``.

Key functions
-------------
    point_timeseries(lon, lat, date_range, variables=...)
        Nearest-pixel daily time series at one site — returns a plain dict.

    fetch_water_clarity(bbox, date_range, out_dir, stream="auto")
        Download the transparency + chlorophyll NetCDF cubes for a box.

    subset_to_netcdf(dataset_id, variables, bbox, date_range, out_path)
        Thin wrapper over ``copernicusmarine.subset``.

Usage
-----
    from src.cmems_client import point_timeseries

    ts = point_timeseries(
        lon=-8.2102, lat=37.0691,                 # Pedra de Santa Eulália
        date_range=("2024-09-01", "2024-09-30"),
        variables=("KD490", "ZSD", "CHL"),
    )
    # {"KD490": {"time": [...], "value": [...]}, "ZSD": {...}, "CHL": {...}}
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy deps — guarded so importing this module never hard-fails
# (matches the try/import pattern used across src/).
# ─────────────────────────────────────────────────────────────────────────────
try:
    import copernicusmarine as _cm  # type: ignore

    _HAS_CM = True
except ImportError:  # pragma: no cover - exercised only without the client
    _cm = None  # type: ignore
    _HAS_CM = False

try:
    import xarray as _xr  # type: ignore

    _HAS_XR = True
except ImportError:  # pragma: no cover
    _xr = None  # type: ignore
    _HAS_XR = False


Stream = Literal["my", "nrt", "auto"]
ConcreteStream = Literal["my", "nrt"]
Product = Literal["transp", "plankton"]
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

# CMEMS multi-year reprocessed ocean-colour typically lags ~3-4 weeks; below this
# horizon the near-real-time stream is the only one with coverage.
MY_LAG_DAYS = 30

# Atlantic ocean-colour L3 datasets, by stream and product.
DATASETS: dict[ConcreteStream, dict[Product, str]] = {
    "my": {
        "transp": "cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D",
        "plankton": "cmems_obs-oc_atl_bgc-plankton_my_l3-multi-1km_P1D",
    },
    "nrt": {
        "transp": "cmems_obs-oc_atl_bgc-transp_nrt_l3-multi-1km_P1D",
        "plankton": "cmems_obs-oc_atl_bgc-plankton_nrt_l3-multi-1km_P1D",
    },
}

# Which product (dataset family) each variable lives in.
VAR_PRODUCT: dict[str, Product] = {
    "KD490": "transp",  # diffuse attenuation at 490 nm  → SDB Kd cross-check
    "ZSD": "transp",  # Secchi disk depth (m)          → visibility metric
    "SPM": "transp",  # suspended particulate matter   → turbidity
    "CHL": "plankton",  # chlorophyll-a                  → algae proxy
}

DEFAULT_VARIABLES: tuple[str, ...] = ("KD490", "ZSD", "SPM", "CHL")

# Faro → Carvoeiro reef domain (the 15 Algarve sites span -8.36..-7.68 / 36.97..37.11;
# padded slightly so the 1 km grid fully covers every site buffer).
ALGARVE_BBOX: BBox = (-8.50, 36.90, -7.60, 37.15)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_cm():
    if not _HAS_CM:
        raise RuntimeError(
            "copernicusmarine is not installed. `pip install copernicusmarine` and "
            "log in with `copernicusmarine login` (free Copernicus Marine account)."
        )
    return _cm


def _iso(day: str, *, end: bool = False) -> str:
    """Normalise a 'YYYY-MM-DD' (or full ISO) string to an ISO datetime."""
    if len(day) == 10:  # date-only → span the whole day
        return f"{day}T23:59:59" if end else f"{day}T00:00:00"
    return day


def _resolve_stream(date_range: tuple[str, str], stream: Stream) -> ConcreteStream:
    """Pick 'my' (reprocessed) or 'nrt' from the requested window."""
    if stream != "auto":
        return stream  # type: ignore[return-value]
    end = datetime.fromisoformat(_iso(date_range[1], end=True)).replace(tzinfo=UTC)
    horizon = datetime.now(UTC) - timedelta(days=MY_LAG_DAYS)
    return "nrt" if end >= horizon else "my"


def _group_by_product(variables: Iterable[str]) -> dict[Product, list]:
    """Split requested variables by the dataset family that serves them."""
    groups: dict[Product, list] = {}
    for var in variables:
        try:
            product = VAR_PRODUCT[var]
        except KeyError as exc:
            raise ValueError(f"Unknown variable {var!r}. Known: {sorted(VAR_PRODUCT)}") from exc
        groups.setdefault(product, []).append(var)
    return groups


def _coord_name(obj, *candidates: str) -> str:
    """Return the first coordinate/dim name present on an xarray object."""
    names = set(getattr(obj, "coords", {})) | set(getattr(obj, "dims", ()))
    for cand in candidates:
        if cand in names:
            return cand
    raise KeyError(f"none of {candidates} found in {sorted(names)}")


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────
def subset_to_netcdf(
    dataset_id: str,
    variables: Sequence[str],
    bbox: BBox,
    date_range: tuple[str, str],
    out_path: str | Path,
) -> Path:
    """Subset one CMEMS dataset to a local NetCDF. Thin wrapper over copernicusmarine.subset."""
    cm = _require_cm()
    min_lon, min_lat, max_lon, max_lat = bbox
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("CMEMS subset %s %s → %s", dataset_id, list(variables), out)
    cm.subset(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=_iso(date_range[0]),
        end_datetime=_iso(date_range[1], end=True),
        output_filename=out.name,
        output_directory=str(out.parent),
        overwrite=True,
        disable_progress_bar=True,
    )
    return out


def fetch_water_clarity(
    bbox: BBox = ALGARVE_BBOX,
    date_range: tuple[str, str] = ("2024-09-01", "2024-09-30"),
    out_dir: str | Path = "outputs/cmems",
    *,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    stream: Stream = "auto",
) -> dict[Product, Path]:
    """Download transparency (KD490/ZSD/SPM) and/or chlorophyll (CHL) cubes for a box.

    Returns a {product: netcdf_path} mapping for whichever products the requested
    variables touch.
    """
    st = _resolve_stream(date_range, stream)
    out_dir = Path(out_dir)
    tag = f"{st}_{date_range[0]}_{date_range[1]}"
    paths: dict[Product, Path] = {}
    for product, vars_ in _group_by_product(variables).items():
        paths[product] = subset_to_netcdf(
            DATASETS[st][product], vars_, bbox, date_range, out_dir / f"algarve_{product}_{tag}.nc"
        )
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Point time series
# ─────────────────────────────────────────────────────────────────────────────
def point_timeseries(
    lon: float,
    lat: float,
    date_range: tuple[str, str],
    variables: Sequence[str] = DEFAULT_VARIABLES,
    *,
    stream: Stream = "auto",
    half_window_deg: float = 0.02,
) -> dict[str, dict[str, list]]:
    """Nearest-pixel daily time series at one site.

    Subsets a small (~2 km) box around (lon, lat) for each needed dataset, reads
    it with xarray, selects the nearest pixel and returns finite samples only::

        {"KD490": {"time": ["2024-09-02", ...], "value": [0.07, ...]}, ...}
    """
    if not _HAS_XR:
        raise RuntimeError(
            "xarray is required for point_timeseries (`pip install xarray netCDF4`)."
        )

    st = _resolve_stream(date_range, stream)
    box: BBox = (
        lon - half_window_deg,
        lat - half_window_deg,
        lon + half_window_deg,
        lat + half_window_deg,
    )
    result: dict[str, dict[str, list]] = {}

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for product, vars_ in _group_by_product(variables).items():
            nc = subset_to_netcdf(
                DATASETS[st][product], vars_, box, date_range, Path(tmp) / f"{product}.nc"
            )
            with _xr.open_dataset(nc) as ds:
                lat_n = _coord_name(ds, "latitude", "lat", "y")
                lon_n = _coord_name(ds, "longitude", "lon", "x")
                pixel = ds.sel({lat_n: lat, lon_n: lon}, method="nearest")
                times = [str(t)[:10] for t in pixel["time"].values]
                for var in vars_:
                    values = pixel[var].values.squeeze().tolist()
                    if not isinstance(values, list):
                        values = [values]
                    # keep finite samples only (v == v drops NaN)
                    pairs = [(t, v) for t, v in zip(times, values, strict=False) if v == v]
                    result[var] = {
                        "time": [t for t, _ in pairs],
                        "value": [float(v) for _, v in pairs],
                    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="CMEMS ocean-colour water-clarity client (Algarve reefs)"
    )
    ap.add_argument("--lon", type=float, help="point longitude (time-series mode)")
    ap.add_argument("--lat", type=float, help="point latitude (time-series mode)")
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default="2024-09-30")
    ap.add_argument("--vars", default=",".join(DEFAULT_VARIABLES), help="comma-separated variables")
    ap.add_argument("--stream", choices=["auto", "my", "nrt"], default="auto")
    ap.add_argument("--out-dir", default="outputs/cmems", help="download mode: NetCDF output dir")
    ap.add_argument("--json", action="store_true", help="print time series as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    variables = tuple(v.strip() for v in args.vars.split(",") if v.strip())
    date_range = (args.start, args.end)

    if args.lon is not None and args.lat is not None:
        ts = point_timeseries(args.lon, args.lat, date_range, variables, stream=args.stream)
        if args.json:
            print(json.dumps(ts, indent=2))
        else:
            for var, series in ts.items():
                vals = series["value"]
                n = len(vals)
                mean = sum(vals) / n if n else float("nan")
                lo = min(vals, default=float("nan"))
                hi = max(vals, default=float("nan"))
                print(f"{var:6s}  n={n:3d}  mean={mean:.4f}  range=[{lo:.4f}, {hi:.4f}]")
    else:
        paths = fetch_water_clarity(
            date_range=date_range, out_dir=args.out_dir, variables=variables, stream=args.stream
        )
        for product, path in paths.items():
            print(f"{product:9s} → {path}")


if __name__ == "__main__":
    _main()
