#!/usr/bin/env python3
"""
Orchestrator — Reef Benthic Visibility Pipeline
================================================
Orquestra: ACOLITE (ou fallback L2A) → run_predictor() → JSON report

Refactor v2.0 — physics delegation:
  - Removed 5 duplicate physics fns (snell, beer-lambert, sunglint, gdal-extract, analyse_band)
  - All physics now handled by reef_ml_predictor_acolite.run_predictor()
  - Added load_config() for YAML-driven paths (no more hardcoded paths)
  - Glint penalty sourced from GLINT_PENALTY constant (not hardcoded 0.60)

Uso: python3 -m src.orchestrator_run [--depth 16.0] [--config config.yaml]
"""

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio

from src.constants import (
    CLOUD_THRESHOLD,
    GLINT_PENALTY,
    GLINT_PENALTY_DEFAULT,
    KD490_DEFAULT,
    KD490_TABLE,
    N_WATER,
)

try:
    from src.cmems_kd490 import get_kd490 as _get_kd490_live
    from src.cmems_kd490 import refresh_from_cmems as _cmems_refresh  # type: ignore[attr-defined]

    HAS_CMEMS_KD = True
except Exception:
    _get_kd490_live = None  # type: ignore[assignment]
    _cmems_refresh = None  # type: ignore[assignment]
    HAS_CMEMS_KD = False

try:
    from src.ipma_sea_state import get_conditions as _ipma_get_conditions
    from src.ipma_sea_state import is_scene_usable as _ipma_is_scene_usable

    HAS_IPMA = True
except Exception:
    HAS_IPMA = False
from src.reef_ml_predictor_acolite import run_predictor

try:
    from src.ih_bathy_features import get_bathy_features_for_summary

    HAS_IH_BATHY = True
except ImportError:
    HAS_IH_BATHY = False

# Drift monitoring (shadow mode — never blocks pipeline)
try:
    from src.drift_export import export_to_file as drift_export_file
    from src.drift_export import export_to_webhook as drift_export_webhook
    from src.drift_history import export_history_csv as drift_history_csv
    from src.drift_history import export_history_json as drift_history_json
    from src.drift_monitor import log_summary as drift_log_summary
    from src.drift_monitor import reset as drift_reset
    from src.drift_report import export_html as drift_export_html

    HAS_DRIFT_MONITOR = True
except ImportError:
    HAS_DRIFT_MONITOR = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Wall-clock cap for the CMEMS live fetch. Without this, a slow or
# unreachable Copernicus Marine endpoint hangs the orchestrator for 60+
# seconds before falling back to the static Kd490 table.
_CMEMS_REFRESH_TIMEOUT_S = 10.0


def _activate_cmems_live() -> None:
    """Refresh the live CMEMS Kd490/ZSD table (shadow mode — static fallback on failure).

    Called at the START of a pipeline run, NOT at import time: the refresh does a
    `copernicusmarine.open_dataset()` + monthly-median over a 1 km daily product,
    which is a heavy network operation. Doing it at import made every `import
    src.orchestrator_run` (including the test suite) pay that cost. Keep it in main().

    Bounded by `_CMEMS_REFRESH_TIMEOUT_S` seconds via a thread pool: if the
    Copernicus Marine endpoint is slow or unreachable we silently fall back
    to the static Kd490 table rather than blocking the orchestrator.
    """
    if not (HAS_CMEMS_KD and _cmems_refresh is not None):
        return
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FutTimeout

    # Note: we use explicit shutdown(wait=False) instead of `with` — the context
    # manager's __exit__ calls shutdown(wait=True), which would block the
    # orchestrator for the full worker duration even after a timeout.
    _ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = _ex.submit(_cmems_refresh)
        try:
            ok = fut.result(timeout=_CMEMS_REFRESH_TIMEOUT_S)
        except _FutTimeout:
            log.warning(
                "CMEMS refresh exceeded %.0fs timeout — using static Kd490 fallback.",
                _CMEMS_REFRESH_TIMEOUT_S,
            )
            return
    except Exception as _e:
        log.debug("CMEMS Kd490 refresh skipped: %s", _e)
        return
    finally:
        _ex.shutdown(wait=False)
    if ok:
        log.info("CMEMS Kd490 live table loaded.")
    else:
        log.info("CMEMS Kd490 using static fallback (credentials not set or unreachable).")


# ── Config defaults ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent

# Coastal terrain features (shadow mode — never blocks pipeline)
try:
    import pandas as _pd

    _COASTAL_CSV = PROJECT_DIR / "outputs" / "coastal_topography" / "algarve_coastal_features.csv"
    _TERRAIN_DF = _pd.read_csv(_COASTAL_CSV) if _COASTAL_CSV.exists() else None
    HAS_TERRAIN = _TERRAIN_DF is not None and not _TERRAIN_DF.empty
except Exception:
    _TERRAIN_DF = None
    HAS_TERRAIN = False

IMAGE_A_B02 = PROJECT_DIR / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B02_20250925.tif"
IMAGE_A_B03 = PROJECT_DIR / "reef_Output_Master/reef_output_pedra_to_gale_20250925/S2_B03_20250925.tif"
IMAGE_B_B02 = PROJECT_DIR / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B02_20231001.tif"
IMAGE_B_B03 = PROJECT_DIR / "reef_Output_Master/reef_output_ai_prediction_spot_2023/S2_B03_20231001.tif"
OUTPUT_DIR = PROJECT_DIR / "reef_output_acolite_comparison"

METADATA = {
    "A": {"date": "2025-09-25", "sza": 40.498, "saa": 158.883, "cloud": 1.245, "level": "L2A", "month": 9},
    "B": {"date": "2023-10-01", "sza": 42.413, "saa": 160.459, "cloud": 0.007, "level": "L2A", "month": 10},
}

TARGET_LAT, TARGET_LON = 37.05815, -8.20982


# ── Config loader (YAML-driven, optional) ────────────────────────────────────
def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load YAML config file if present. Falls back to module-level defaults.
    Expected keys: image_a_b02, image_a_b03, image_b_b02, image_b_b03,
                   output_dir, target_lat, target_lon, metadata.
    """
    if config_path is None:
        return {}
    try:
        import yaml  # type: ignore

        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        log.info("Loaded config from %s", config_path)
        return cfg
    except ImportError:
        log.warning("PyYAML not installed — ignoring config file %s", config_path)
        return {}
    except FileNotFoundError:
        log.warning("Config file not found: %s — using defaults", config_path)
        return {}
    except Exception as e:
        log.warning("Config file parse error (%s) — using defaults: %s", config_path, e)
        return {}


# ── Shell helpers ─────────────────────────────────────────────────────────────
def run_shell(cmd, check=True):
    """Run a command safely. Accepts str or list of args. Uses shell=False."""
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)
    log.info("$ %s", " ".join(shlex.quote(c) for c in cmd_list))
    result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, timeout=600)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{result.stderr}")
    return result


def acolite_available() -> bool:
    return shutil.which("acolite") is not None or shutil.which("acolite_cli") is not None


def snap_gpt_available() -> bool:
    return shutil.which("gpt") is not None and str(shutil.which("gpt")) != "/usr/sbin/gpt"


def run_acolite(input_path: Path, output_dir: Path):
    """Run ACOLITE BOA correction with sunglint removal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "acolite_cli",
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--product",
        "BOA",
        "--sensor",
        "S2",
        "--proc",
        "water",
        "--sunglint",
        "true",
        "--aot-method",
        "image",
        "--output-format",
        "GeoTIFF",
    ]
    run_shell(cmd)
    boa = next(output_dir.glob("*BOA*.tif"), None)
    if not boa:
        raise FileNotFoundError(f"ACOLITE BOA output not found in {output_dir}")
    return boa


# ── Band extractor (merged gdal_extract_b02 + gdal_extract_b03) ──────────────
def extract_band(boa_tif: Path, band_num: int, out_path: Path) -> Path:
    """Extract a single band from a multi-band GeoTIFF using rasterio."""
    with rasterio.open(boa_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        data = src.read(band_num)  # rasterio is 1-based
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    log.info("Extracted band %d → %s", band_num, out_path)
    return out_path


# ── Result normaliser: run_predictor() → legacy key aliases ──────────────────
def _normalise_result(pred: dict, meta: dict) -> dict:
    """
    Thin adapter: maps run_predictor() output keys to the legacy aliases
    expected by _build_justification() and the JSON report builder.
    run_predictor() does not compute b02_cv directly — derive it as 1/SNR.
    """
    # Use explicit None check: dict.get(key, default) returns None when the key
    # IS present but its value is None, so a truthy/falsy test would be wrong.
    _snr_raw = pred.get("snr_mean_16m") if "snr_mean_16m" in pred else pred.get("SNR_mean_16m")
    snr = _snr_raw if _snr_raw is not None else 0.0
    out = dict(pred)
    # Legacy aliases
    out.setdefault("SNR_mean_16m", snr)
    _month = meta.get("month")
    if _month is not None and _get_kd490_live is not None:
        _kd_seasonal = _get_kd490_live(int(_month))
    elif _month is not None and _month in KD490_TABLE:
        _kd_seasonal = KD490_TABLE[_month]
    else:
        _kd_seasonal = pred.get("kd_seasonal_prior", KD490_DEFAULT)
    out.setdefault("kd490_seasonal", _kd_seasonal)
    _kd_est = pred.get("kd_b02_estimated") if "kd_b02_estimated" in pred else pred.get("kd490_seasonal")
    out.setdefault("kd490_estimated", _kd_est if _kd_est is not None else KD490_DEFAULT)
    out.setdefault("date", meta["date"])
    out.setdefault("cloud_cover", meta["cloud"])
    # b02_cv: coefficient of variation — proxy from SNR (CV = 1/SNR)
    out["b02_cv"] = round(1.0 / max(snr, 1e-6), 5)
    # Sentinel-1 fields (may be absent if run_predictor didn't query S1)
    out.setdefault("s1_scene_id", "none")
    out.setdefault("s1_scene_date", "none")
    out.setdefault("s1_roughness", -1.0)
    out.setdefault("s1_sea_state", "unknown")
    out.setdefault("s1_penalty_pct", 0.0)
    return out


# ── Output writers ────────────────────────────────────────────────────────────
def save_boa_copy(src_b02: Path, src_b03: Path, out_dir: Path, label: str) -> dict:
    """Copy pre-processed bands to output dir as 'BOA' equivalents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    boa_b02 = out_dir / f"BOA_B02_{label}.tif"
    snr_map = out_dir / f"SNR_map_{label}.tif"
    conf_map = out_dir / f"Confidence_map_{label}.tif"
    shutil.copy2(src_b02, boa_b02)

    with rasterio.open(src_b02) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(float) / 10000.0
        from src.reef_ml_predictor_acolite import make_snr_map

        snr_px = make_snr_map(data, window=7)
        conf_px = np.select([snr_px < 5, snr_px < 30], [0, 1], default=2).astype(np.uint8)

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(snr_map, "w", **profile) as dst:
        dst.write(snr_px, 1)
    profile.update(dtype=rasterio.uint8)
    with rasterio.open(conf_map, "w", **profile) as dst:
        dst.write(conf_px, 1)

    return {"boa_b02": str(boa_b02), "snr_map": str(snr_map), "confidence_map": str(conf_map)}


def _build_justification(winner: str, loser: str, results: dict, snr_diff: float, depth: float) -> str:
    """Construct human-readable justification safely (handles zero-division)."""
    w_cv = results[winner]["b02_cv"]
    l_cv = results[loser]["b02_cv"]
    cv_ratio = (l_cv / w_cv) if w_cv > 1e-9 else float("inf")
    snr_str = f"+{snr_diff:.0f}%" if snr_diff != float("inf") else "+inf%"
    cv_str = f"{cv_ratio:.1f}×" if cv_ratio != float("inf") else "∞×"
    return (
        f"Image {winner} ({results[winner]['date']}) chosen. "
        f"SNR {results[winner]['SNR_mean_16m']:.1f} vs {results[loser]['SNR_mean_16m']:.1f} "
        f"({snr_str}). CV {w_cv:.4f} vs {l_cv:.4f} "
        f"({cv_str} more surface noise in loser). "
        f"Kd490={results[winner]['kd490_seasonal']:.3f} m⁻¹ | "
        f"Two-way transmittance={results[winner]['water_transmittance_twoway']:.4f} at {depth:.0f}m."
    )


def save_csv(results: dict, path: Path):
    import csv

    rows = [{"image_key": k, **r} for k, r in results.items()]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log.info("CSV saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(depth: float = 16.0, config_path: str | None = None):
    # Apply YAML config overrides (if provided)
    cfg = load_config(config_path)
    b02_a_path = Path(cfg["image_a_b02"]) if cfg.get("image_a_b02") else IMAGE_A_B02
    b03_a_path = Path(cfg["image_a_b03"]) if cfg.get("image_a_b03") else IMAGE_A_B03
    b02_b_path = Path(cfg["image_b_b02"]) if cfg.get("image_b_b02") else IMAGE_B_B02
    b03_b_path = Path(cfg["image_b_b03"]) if cfg.get("image_b_b03") else IMAGE_B_B03
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else OUTPUT_DIR
    target_lat = float(cfg["target_lat"]) if cfg.get("target_lat") else TARGET_LAT
    target_lon = float(cfg["target_lon"]) if cfg.get("target_lon") else TARGET_LON

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Reef Orchestrator — depth=%.1fm ===", depth)

    # Load live CMEMS Kd490/ZSD now (runtime, not import time) so run_predictor
    # below picks up the live seasonal prior via the shared in-place table.
    _activate_cmems_live()

    if HAS_DRIFT_MONITOR:
        try:
            drift_reset()
        except Exception:
            pass

    # Step 1: ACOLITE or direct L2A fallback
    use_acolite = acolite_available()
    log.info(
        "ACOLITE: %s | SNAP/gpt: %s", "YES" if use_acolite else "NO (fallback L2A)", "YES" if snap_gpt_available() else "NO"
    )

    if use_acolite:
        boa_a = run_acolite(b02_a_path.parent, out_dir / "acolite_A")
        boa_b = run_acolite(b02_b_path.parent, out_dir / "acolite_B")
        b02_a_path = extract_band(boa_a, 2, out_dir / "BOA_B02_A_raw.tif")
        b02_b_path = extract_band(boa_b, 2, out_dir / "BOA_B02_B_raw.tif")
        b03_a_path = extract_band(boa_a, 3, out_dir / "BOA_B03_A_raw.tif")
        b03_b_path = extract_band(boa_b, 3, out_dir / "BOA_B03_B_raw.tif")
    else:
        log.info("Using L2A BOA TIFFs directly (ACOLITE not installed)")

    # Step 1b: Sea-state filter (IPMA) — skip scenes during storms
    if HAS_IPMA:
        for label, meta in [("A", METADATA["A"]), ("B", METADATA["B"])]:
            if not _ipma_is_scene_usable(meta["date"]):
                log.warning(
                    "Image %s (%s) fails IPMA sea-state check — " "BVI score may be degraded by high turbidity.",
                    label,
                    meta["date"],
                )
                # Not a hard skip: sea state affects BVI but scene may still have
                # valid cloud-free pixels. Log the warning and continue.

    # Step 1c: Multi-scene Stumpf fusion — A and B are two cloud-free scenes of the
    # same site, so calibrate both against IH isobaths and take a robust median m0/m1.
    # The fused coefficients are fed into run_predictor so both scenes share
    # consistent, artefact-resistant calibration. Best-effort: on any failure each
    # scene falls back to its own single-scene IH calibration inside run_predictor.
    fused_m0 = fused_m1 = None
    try:
        from src.stumpf_multiscene import fuse_scenes

        _pad = 0.03  # ~3 km around the target point
        _site_bbox = (target_lon - _pad, target_lat - _pad, target_lon + _pad, target_lat + _pad)
        _fused = fuse_scenes(
            scenes=[
                {"b02": str(b02_a_path), "b03": str(b03_a_path), "date": METADATA["A"]["date"]},
                {"b02": str(b02_b_path), "b03": str(b03_b_path), "date": METADATA["B"]["date"]},
            ],
            site_bbox=_site_bbox,
        )
        if _fused.get("status") == "fused":
            fused_m0, fused_m1 = _fused["m0"], _fused["m1"]
            log.info(
                "Multi-scene fusion: %d scenes used (n_rejected=%d) → " "m0=%.3f±%.3f m1=%.3f±%.3f",
                _fused["n_scenes"],
                _fused["n_rejected"],
                fused_m0,
                _fused["m0_std"],
                fused_m1,
                _fused["m1_std"],
            )
        else:
            log.info(
                "Multi-scene fusion not applied (status=%s) — " "single-scene calibration per image", _fused.get("status")
            )
    except Exception as _fuse_err:
        log.warning("Multi-scene fusion skipped: %s", _fuse_err)

    # Step 2: Physics — delegate entirely to run_predictor()
    log.info("Running run_predictor() for both images...")
    meta_a = {
        "date": METADATA["A"]["date"],
        "solar_zenith_deg": METADATA["A"]["sza"],
        "solar_azimuth_deg": METADATA["A"]["saa"],
        "cloud_cover_pct": METADATA["A"]["cloud"],
    }
    meta_b = {
        "date": METADATA["B"]["date"],
        "solar_zenith_deg": METADATA["B"]["sza"],
        "solar_azimuth_deg": METADATA["B"]["saa"],
        "cloud_cover_pct": METADATA["B"]["cloud"],
    }

    pred_a = run_predictor(
        boa_b02_path=str(b02_a_path),
        metadata=meta_a,
        output_dir=str(out_dir / "pred_A"),
        date=METADATA["A"]["date"],
        b03_path=str(b03_a_path),
        lat=target_lat,
        lon=target_lon,
        depth_target=depth,
        with_bathy_features=HAS_IH_BATHY,
        stumpf_m0_override=fused_m0,
        stumpf_m1_override=fused_m1,
    )
    pred_b = run_predictor(
        boa_b02_path=str(b02_b_path),
        metadata=meta_b,
        output_dir=str(out_dir / "pred_B"),
        date=METADATA["B"]["date"],
        b03_path=str(b03_b_path),
        lat=target_lat,
        lon=target_lon,
        depth_target=depth,
        with_bathy_features=HAS_IH_BATHY,
        stumpf_m0_override=fused_m0,
        stumpf_m1_override=fused_m1,
    )

    res_a = _normalise_result(pred_a, METADATA["A"])
    res_b = _normalise_result(pred_b, METADATA["B"])

    # Apply Sentinel-1 roughness penalty (best-effort, uses existing S1 integration)
    for key, res, meta in [("A", res_a, METADATA["A"]), ("B", res_b, METADATA["B"])]:
        try:
            from datetime import timedelta

            from src.sentinel1_roughness import (
                extract_sigma0_at_point,
                roughness_from_sigma0,
                search_stac_s1_scenes,
            )

            t_date = datetime.strptime(meta["date"], "%Y-%m-%d")
            start_dt = t_date - timedelta(days=3)
            end_dt = t_date + timedelta(days=3)
            s1_items = search_stac_s1_scenes(
                lon=target_lon,
                lat=target_lat,
                year=t_date.year,
                month_start=start_dt.month,
                month_end=end_dt.month,
                max_results=10,
            )
            valid = [i for i in s1_items if start_dt <= datetime.strptime(i["date"], "%Y-%m-%d") <= end_dt]
            if valid:
                valid.sort(key=lambda x: abs((datetime.strptime(x["date"], "%Y-%m-%d") - t_date).total_seconds()))
                best = valid[0]
                sigma0 = extract_sigma0_at_point(best, target_lon, target_lat)
                if sigma0 and sigma0.get("vv") is not None and sigma0.get("vh") is not None:
                    r_data = roughness_from_sigma0(sigma0["vv"], sigma0["vh"])
                    if r_data.get("roughness") is not None:
                        roughness = r_data["roughness"]
                        penalty_factor = float(np.clip((roughness - 0.05) / 0.20, 0.0, 1.0)) * 0.3
                        res["s1_scene_id"] = best["id"]
                        res["s1_scene_date"] = best["date"]
                        res["s1_roughness"] = roughness
                        res["s1_sea_state"] = r_data["sea_state"]
                        res["s1_penalty_pct"] = round(penalty_factor * 100.0, 2)
                        res["visibility_score"] = round(res.get("visibility_score", 0) * (1.0 - penalty_factor), 4)
                        log.info("[S1 %s] roughness=%.4f → penalty=%.2f%%", key, roughness, res["s1_penalty_pct"])
        except Exception as e:
            log.warning("Sentinel-1 penalty skipped for image %s: %s", key, e)

    # Apply coastal terrain exposure modifier (best-effort, never blocks pipeline)
    if HAS_TERRAIN:
        try:

            from src.ranking_model import terrain_exposure_modifier

            # Nearest site by Euclidean distance in degrees (fast, ~1 km precision)
            df = _TERRAIN_DF.copy()
            df["_dist"] = (df["latitude"] - target_lat) ** 2 + (df["longitude"] - target_lon) ** 2
            row = df.loc[df["_dist"].idxmin()]
            terrain_feat = {
                "slope_mean": float(np.nan_to_num(row.get("slope_mean") or 0.0, nan=0.0)),
                "aspect_mean": float(np.nan_to_num(row.get("aspect_mean") or 180.0, nan=180.0)),
            }
            mod = terrain_exposure_modifier(terrain_feat["slope_mean"], terrain_feat["aspect_mean"])
            for key, res in [("A", res_a), ("B", res_b)]:
                original = res.get("visibility_score", 0.0)
                res["visibility_score"] = round(original * mod, 4)
                res["terrain_site"] = str(row.get("site_name", "nearest"))
                res["terrain_modifier"] = round(mod, 4)
                res["terrain_slope"] = terrain_feat["slope_mean"]
                res["terrain_aspect"] = terrain_feat["aspect_mean"]
            log.info(
                "[Terrain] site=%s slope=%.2f° aspect=%.1f° modifier=%.3f",
                row.get("site_name"),
                terrain_feat["slope_mean"],
                terrain_feat["aspect_mean"],
                mod,
            )
        except Exception as e:
            log.warning("Terrain modifier skipped: %s", e)

    results = {"A": res_a, "B": res_b}

    # Step 3: ML ranking (already computed inside run_predictor via predict_score)
    score_a = res_a.get("visibility_score", 0.0)
    score_b = res_b.get("visibility_score", 0.0)

    winner = "A" if score_a >= score_b else "B"
    loser = "B" if winner == "A" else "A"
    loser_snr = results[loser]["SNR_mean_16m"]
    snr_diff = ((results[winner]["SNR_mean_16m"] - loser_snr) / loser_snr * 100) if loser_snr > 0 else float("inf")

    warnings = []
    if res_a.get("kd_high_uncertainty"):
        warnings.append("Kd high uncertainty in Image A")
    if res_b.get("kd_high_uncertainty"):
        warnings.append("Kd high uncertainty in Image B")

    # Step 4: Save BOA copies + maps
    maps_a = save_boa_copy(b02_a_path, b03_a_path, out_dir, "A_20250925")
    maps_b = save_boa_copy(b02_b_path, b03_b_path, out_dir, "B_20231001")

    # Step 5: JSON + CSV output
    csv_path = out_dir / "summary_comparison.csv"
    save_csv(results, csv_path)

    report = {
        "chosen_image": results[winner]["date"],
        "scores": {"A": score_a, "B": score_b},
        "metrics": {"A": res_a, "B": res_b},
        "outputs": {
            "boa_b02_a": maps_a["boa_b02"],
            "boa_b02_b": maps_b["boa_b02"],
            "snr_map_a": maps_a["snr_map"],
            "snr_map_b": maps_b["snr_map"],
            "confidence_map_a": maps_a["confidence_map"],
            "confidence_map_b": maps_b["confidence_map"],
            "summary_csv": str(csv_path),
        },
        "justification": _build_justification(winner, loser, results, snr_diff, depth),
        "assumptions": [
            "n_water=1.333 (Snell refraction)",
            f"depth_target={depth}m",
            "Kd490_table: Sep/Oct=0.045, Jan/Feb=0.055, Apr/May=0.200, else=0.080",
            "Sunglint: Hedley linear correction via simulate_acolite_boa() in utils.py",
            f"Cloud threshold={CLOUD_THRESHOLD}%",
            "Physics engine: reef_ml_predictor_acolite.run_predictor()",
            "Datum: WGS84/UTM Zone 29N",
            f"ACOLITE: {'used' if use_acolite else 'not installed — L2A BOA used directly'}",
        ],
        "sea_state": _ipma_get_conditions() if HAS_IPMA else None,
        "kd490_source": "CMEMS live" if _get_kd490_live is not None else "static table",
        "warnings": warnings,
        "training_inputs_reef_ml_predictor": {
            "month_9_glint_penalty": GLINT_PENALTY.get(9, GLINT_PENALTY_DEFAULT),
            "month_10_glint_penalty": GLINT_PENALTY.get(10, GLINT_PENALTY_DEFAULT),
            "kd490_sep_oct": 0.045,
            "depth_target_m": depth,
            "n_water": N_WATER,
        },
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    report_path = out_dir / "orchestrator_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    log.info("=== DONE ===")
    log.info("Winner: %s | Score A=%.4f B=%.4f", report["chosen_image"], score_a, score_b)
    log.info("JSON → %s", report_path)
    log.info("CSV  → %s", csv_path)

    # ICESat-2 validation — optional, runs after depth maps are generated
    try:
        from src.icesat2_validation import run_icesat2_validation

        winner_pred_dir = out_dir / f"pred_{winner}"
        sdb_path = winner_pred_dir / "sdb_depth_map.tif"
        val_report = run_icesat2_validation(
            depth_map_path=sdb_path,
            output_dir=out_dir / "icesat2_validation",
            scene_date=results[winner].get("date"),
        )
        if val_report and val_report.get("status") == "ok":
            report["icesat2_validation"] = {
                "rmse_m": val_report["rmse_m"],
                "bias_m": val_report["bias_m"],
                "n_points": val_report["n_colocated"],
                "report_path": str(out_dir / "icesat2_validation" / "validation_report.json"),
            }
            # Rewrite the JSON report with the validation results appended
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.debug("ICESat-2 validation (non-critical): %s", e)

    # Drift monitoring: batch-end summary + export (shadow mode)
    if HAS_DRIFT_MONITOR:
        try:
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
            batch_id = f"reef-orchestrator-{ts}-depth{int(depth)}"
            drift_log_summary()
            drift_export_file(batch_id=batch_id)
            drift_history_json()
            drift_history_csv()
            drift_export_html()
            # Optional, env-gated webhook export (shadow layer — never blocks).
            webhook_url = os.environ.get("DRIFT_WEBHOOK_URL")
            if webhook_url:
                try:
                    drift_export_webhook(webhook_url, batch_id=batch_id)
                except Exception as e:  # pragma: no cover - defensive
                    log.debug("Drift webhook export (non-critical): %s", e)
        except Exception as e:
            log.debug("Drift reporting (non-critical): %s", e)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reef Benthic Visibility Orchestrator v2")
    parser.add_argument("--depth", type=float, default=16.0, help="Target depth in metres")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--image-a-b02", type=str, help="Override Image A B02 path")
    parser.add_argument("--image-b-b02", type=str, help="Override Image B B02 path")
    args = parser.parse_args()
    if args.image_a_b02:
        IMAGE_A_B02 = Path(args.image_a_b02)
    if args.image_b_b02:
        IMAGE_B_B02 = Path(args.image_b_b02)
    main(depth=args.depth, config_path=args.config)
