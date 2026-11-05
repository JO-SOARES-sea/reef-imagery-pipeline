#!/usr/bin/env python3
"""
scripts/build_cosmo_truth_points.py — Gera o CSV de verdade de rocha (lon, lat, cume_id) a partir do raster COSMO.

    python scripts/build_cosmo_truth_points.py --sector LTHQUGA1 --out data/vector/quarteira_cosmo_rocha.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.cosmo_truth import cosmo_rock_points  # noqa: E402

TRUTH = PROJECT_ROOT / "data" / "processed" / "cosmo_truth"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sector", required=True, help="ex.: LTHQUGA1 (nunca LTHNOCR1: holdout v2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-px", type=int, default=2)
    args = ap.parse_args()
    if args.sector.upper() == "LTHNOCR1":
        ap.error("LTHNOCR1 é o holdout v2 (docs/HOLDOUT_PREREGISTO_v2.md): não pode gerar verdade de desenvolvimento")
    with rasterio.open(TRUTH / f"{args.sector}_truth10m.tif") as src:
        df = cosmo_rock_points(src.read(5), src.transform, src.crs, merge_px=args.merge_px)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.7f")
    print(f"{len(df)} píxeis de rocha, {df.cume_id.nunique()} cumes -> {out}")


if __name__ == "__main__":
    main()
