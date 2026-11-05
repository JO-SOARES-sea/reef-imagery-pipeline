#!/usr/bin/env python3
"""
score_fishing_spots_mini.py — Run a trained ResNet-18 over named fishing spots.

Given a trained checkpoint (from ``train_patch_classifier_mini.py``) and a list
of named fishing-spot coordinates (from ``extract_fishing_spots_mini.py``),
extract a 128×128 Sentinel-2 patch (4 bands: B02/B03/B04/B08) at each spot
from a user-specified scene, normalise it, and predict the reef probability.

This is a more powerful follow-up to ``crossref_fishing_spots_mini.py``:
- crossref_fishing_spots_mini uses raw B04 reflectance (one number per spot).
- This script uses the trained 4-channel model (millions of learned features)
  to score the same spots.  A spot that the model ranks high but B04 ranks
  low is a candidate the simple B04 test misses.

Inputs
------
- ``--checkpoint``  Trained ResNet-18 .pth (from train_patch_classifier_mini.py)
- ``--scene-id``    STAC item id (default: 2025-09-25, the user-confirmed best scene)
- ``--spots``       Named fishing spots CSV (default: named_reef_candidates_full.csv)
- ``--top-k``       Only score the K darkest spots (B04-ranked) — useful for budget

Outputs (under ``--out-dir``, default ``outputs/reef_habitat_mini/model_scored_fishing/``)
----------------------------------------------------------------------------------------
- ``model_scores.csv``  — name, lon, lat, depth_m, b04, model_prob
- ``model_scores.md``   — top-K and bottom-K model scores for quick review
- ``model_scores_log.txt``

Example
-------
    python scripts/score_fishing_spots_mini.py \\
        --checkpoint outputs/reef_habitat_mini/model/reef_cnn_best.pth \\
        --scene-id S2B_29SNB_20250925_0_L2A \\
        --spots outputs/reef_habitat_mini/crossref_2025_09_25/named_reef_candidates_full.csv

Notes
-----
* SAFE MODE: only creates new files in ``outputs/reef_habitat_mini/``.
  Baseline ``outputs/reef_habitat/`` and the user's GPX are untouched.
* Imports ``ReefClassifier`` and ``predict_single`` from
  ``src.reef_classifier`` (single source of truth for the model).
* Reuses band alias + normalisation tables from
  ``src.s2_patch_extractor.py`` (replicated inline with attribution, ~20 lines).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from pystac_client import Client
    _HAS_PYSTAC = True
except Exception:
    _HAS_PYSTAC = False

import requests
try:
    from pyproj import Transformer as _PyprojTransformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("score_fishing_spots_mini")

_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"
_DN_SCALE = 10000.0
_PATCH_PX = 128

# Mirrors src/s2_patch_extractor.py: BANDS + _CLIP_LOW/HIGH
BANDS = ["B02", "B03", "B04", "B08"]
_BAND_ALIASES = {
    "B02": ["blue", "B02", "B2"],
    "B03": ["green", "B03", "B3"],
    "B04": ["red", "B04", "B4"],
    "B08": ["nir", "nir08", "B08", "B8", "B8A"],
}
_CLIP_LOW  = {"B02": 0,    "B03": 0,    "B04": 0,    "B08": 0}
_CLIP_HIGH = {"B02": 2500, "B03": 2000, "B04": 1800, "B08": 3000}


# ── STAC + helpers (small; attribution to src/s2_patch_extractor.py) ──────────
def _get_scene_assets(scene_id: str) -> dict:
    if _HAS_PYSTAC:
        try:
            cat = Client.open(_EARTH_SEARCH)
            item = cat.get_collection(_S2_COLLECTION).get_item(scene_id)
            if item is not None:
                return {k: v.href for k, v in item.assets.items()}
        except Exception as exc:
            log.warning("pystac-client failed (%s) — falling back to raw API", exc)
    resp = requests.get(f"{_EARTH_SEARCH}/collections/{_S2_COLLECTION}/items/{scene_id}",
                        timeout=30)
    resp.raise_for_status()
    return {k: v["href"] for k, v in resp.json()["assets"].items()}


def _band_url(assets: dict, band: str) -> Optional[str]:
    """Resolve asset URL for *band*.  Attribution: src/s2_patch_extractor.py:170."""
    aliases = _BAND_ALIASES.get(band, [band])
    for alias in aliases:
        if alias in assets:
            return assets[alias]
        for key, href in assets.items():
            if key.lower() == alias.lower():
                return href
    return None


def _normalise_band(data: np.ndarray, band: str) -> np.ndarray:
    """Clip + rescale to [0,1].  Attribution: src/s2_patch_extractor.py:238."""
    lo, hi = _CLIP_LOW[band], _CLIP_HIGH[band]
    return (np.clip(data, lo, hi) - lo) / (hi - lo)


# ── Patch extraction at (lon, lat) ─────────────────────────────────────────────
def _extract_patch(
    band_datasets: dict,
    band_to_epsg_xformer,
    lon: float,
    lat: float,
) -> Optional[np.ndarray]:
    """Return a (4, 128, 128) float32 patch, or None on any failure."""
    try:
        channels = []
        for band in BANDS:
            src = band_datasets[band]
            if band_to_epsg_xformer is not None:
                cx, cy = band_to_epsg_xformer.transform(lon, lat)
            else:
                cx, cy = lon, lat
            row_f, col_f = src.index(cx, cy)
            if not (0 <= row_f < src.height and 0 <= col_f < src.width):
                return None
            half = _PATCH_PX // 2
            col_off = max(0, col_f - half)
            row_off = max(0, row_f - half)
            col_off = min(col_off, src.width  - _PATCH_PX)
            row_off = min(row_off, src.height - _PATCH_PX)
            window = Window.from_slices(
                rows=(row_off, row_off + _PATCH_PX),
                cols=(col_off, col_off + _PATCH_PX),
            )
            data = src.read(1, window=window).astype(np.float32)
            if data.shape != (_PATCH_PX, _PATCH_PX):
                padded = np.zeros((_PATCH_PX, _PATCH_PX), dtype=np.float32)
                padded[: data.shape[0], : data.shape[1]] = data
                data = padded
            channels.append(_normalise_band(data, band))
        return np.stack(channels, axis=0).astype(np.float32)
    except Exception as exc:
        log.debug("Patch extraction failed at (%.4f, %.4f): %s", lon, lat, exc)
        return None


# ── Output writers ────────────────────────────────────────────────────────────
def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "lon", "lat", "emodnet_depth_m", "b04_reflectance",
                  "model_prob", "patch_ok"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "name":              r.get("name", ""),
                "lon":               r["lon"],
                "lat":               r["lat"],
                "emodnet_depth_m":   r.get("emodnet_depth_m", ""),
                "b04_reflectance":   f"{r['b04']:.5f}" if r.get("b04") is not None else "",
                "model_prob":        f"{r['model_prob']:.4f}" if r.get("model_prob") is not None else "",
                "patch_ok":          r.get("patch_ok", False),
            })
    log.info("Wrote CSV: %s  (%d rows)", path, len(rows))


def _write_md(rows: list[dict], path: Path, n_top: int = 20) -> None:
    """Write top-N and bottom-N (by model_prob) for quick review."""
    valid = [r for r in rows if r.get("model_prob") is not None]
    valid.sort(key=lambda r: -r["model_prob"])
    top = valid[:n_top]
    bot = valid[-n_top:] if len(valid) > n_top else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Model scores on named fishing spots\n\n")
        f.write(f"Total scored: {len(rows)}    "
                f"with patch: {sum(1 for r in rows if r.get('patch_ok'))}    "
                f"valid model_prob: {len(valid)}\n\n")
        f.write(f"## Top {len(top)} by model probability (most reef-like)\n\n")
        f.write("| name | lon | lat | depth_m | b04 | model_prob |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in top:
            f.write(f"| {r.get('name','')} | {r['lon']:.4f} | {r['lat']:.4f} | "
                    f"{r.get('emodnet_depth_m', '')} | "
                    f"{r.get('b04', 0):.5f} | {r['model_prob']:.4f} |\n")
        if bot:
            f.write(f"\n## Bottom {len(bot)} by model probability (least reef-like)\n\n")
            f.write("| name | lon | lat | depth_m | b04 | model_prob |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for r in bot:
                f.write(f"| {r.get('name','')} | {r['lon']:.4f} | {r['lat']:.4f} | "
                        f"{r.get('emodnet_depth_m', '')} | "
                        f"{r.get('b04', 0):.5f} | {r['model_prob']:.4f} |\n")
    log.info("Wrote Markdown: %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="score_fishing_spots_mini",
        description="Run a trained ResNet-18 over named fishing spots.",
    )
    p.add_argument("--checkpoint", type=str,
                   default="outputs/reef_habitat_mini/model/reef_cnn_best.pth",
                   help="Trained ResNet-18 checkpoint (.pth).")
    p.add_argument("--scene-id", type=str,
                   default="S2B_29SNB_20250925_0_L2A",
                   help="STAC item id (default: 2025-09-25 — user-confirmed best scene).")
    p.add_argument("--spots", type=str,
                   default="outputs/reef_habitat_mini/crossref_2025_09_25/named_reef_candidates_full.csv",
                   help="Fishing spots CSV (must have lon, lat, name, emodnet_depth_m, b04_reflectance).")
    p.add_argument("--top-k", type=int, default=0,
                   help="Only score the K darkest spots (B04-ranked); 0 = all (default).")
    p.add_argument("--out-dir", type=str,
                   default="outputs/reef_habitat_mini/model_scored_fishing",
                   help="Output directory (isolated from outputs/reef_habitat/).")
    p.add_argument("--device", type=str, default="auto",
                   help="cpu / cuda / mps / auto (default: auto).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")
    sys.stdout.flush()

    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        log.error("Checkpoint not found: %s", ckpt)
        return 1
    spots_csv = Path(args.spots)
    if not spots_csv.exists():
        log.error("Spots CSV not found: %s", spots_csv)
        return 1

    # Load the model
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.reef_classifier import load_model, predict_single  # type: ignore
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    log.info("Loading model: %s  device=%s", ckpt, device)
    model = load_model(ckpt, device=device)

    # Resolve scene + open the 4 band datasets
    log.info("Resolving scene: %s", args.scene_id)
    assets = _get_scene_assets(args.scene_id)
    band_datasets: dict = {}
    for band in BANDS:
        href = _band_url(assets, band)
        if href is None:
            log.error("Band %s not found in scene %s", band, args.scene_id)
            return 1
        band_datasets[band] = rasterio.open(href)
    try:
        first = band_datasets[BANDS[0]]
        if first.crs is not None and first.crs.to_epsg() != 4326 and _HAS_PYPROJ:
            xformer = _PyprojTransformer.from_crs("EPSG:4326", f"EPSG:{first.crs.to_epsg()}",
                                                  always_xy=True)
        else:
            xformer = None
    finally:
        pass

    # Load spots
    spots = []
    with open(spots_csv) as f:
        for r in csv.DictReader(f):
            try:
                lon = float(r["lon"])
                lat = float(r["lat"])
            except (KeyError, ValueError):
                continue
            d = r.get("emodnet_depth_m") or r.get("depth_m") or ""
            b04 = r.get("b04_reflectance") or ""
            spots.append({
                "name": r.get("name", ""),
                "lon": lon, "lat": lat,
                "emodnet_depth_m": d,
                "b04": float(b04) if b04 else None,
            })
    if args.top_k > 0:
        # Take K with lowest B04 (darkest = most reef-like by raw signal)
        spots = sorted([s for s in spots if s["b04"] is not None],
                       key=lambda s: s["b04"])[:args.top_k]
    log.info("Scoring %d fishing spots", len(spots))

    # Score
    results: list[dict] = []
    n_ok = 0
    for i, s in enumerate(spots, start=1):
        patch = _extract_patch(band_datasets, xformer, s["lon"], s["lat"])
        if patch is None:
            results.append({**s, "model_prob": None, "patch_ok": False})
            continue
        try:
            prob = predict_single(model, patch, device=device)
        except Exception as exc:
            log.debug("predict_single failed for %s: %s", s.get("name"), exc)
            results.append({**s, "model_prob": None, "patch_ok": True})
            continue
        results.append({**s, "model_prob": float(prob), "patch_ok": True})
        n_ok += 1
        if i % 100 == 0:
            log.info("  scored %d / %d", i, len(spots))

    for ds in band_datasets.values():
        ds.close()

    _write_csv(results, out_dir / "model_scores.csv")
    _write_md(results,  out_dir / "model_scores.md")

    log_path = out_dir / "model_scores_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.write(f"timestamp:  {ts}\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"scene_id:   {args.scene_id}\n")
        f.write(f"spots:      {spots_csv}  ({len(spots)} loaded)\n")
        f.write(f"top_k:      {args.top_k}\n")
        f.write(f"device:     {device}\n")
        f.write(f"scored_ok:  {n_ok} / {len(spots)}\n")
        if results and any(r.get("model_prob") is not None for r in results):
            probs = [r["model_prob"] for r in results if r.get("model_prob") is not None]
            f.write(f"prob min/median/max: {min(probs):.4f} / "
                    f"{float(np.median(probs)):.4f} / {max(probs):.4f}\n")

    # Quick summary
    probs = [r["model_prob"] for r in results if r.get("model_prob") is not None]
    print("\n── Model scores on fishing spots ─────────────────────────────")
    print(f"  Spots loaded    : {len(spots)}")
    print(f"  Patches ok      : {sum(1 for r in results if r.get('patch_ok'))}")
    print(f"  Valid probs     : {len(probs)}")
    if probs:
        print(f"  Prob min/med/max: {min(probs):.4f} / {float(np.median(probs)):.4f} / {max(probs):.4f}")
    print(f"  Outputs         : {out_dir}")
    if probs:
        ranked = sorted([r for r in results if r.get("model_prob") is not None],
                        key=lambda r: -r["model_prob"])
        print("\n  Top 5 by model probability:")
        for r in ranked[:5]:
            print(f"    {r.get('name',''):>16}  ({r['lon']:.4f}, {r['lat']:.4f})  "
                  f"prob={r['model_prob']:.4f}  b04={r.get('b04', 0):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
