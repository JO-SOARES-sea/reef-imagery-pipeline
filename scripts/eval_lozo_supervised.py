#!/usr/bin/env python3
"""
eval_lozo_supervised.py — Leave-One-Zone-Out evaluation of the supervised
reef detector against the project's pre-registered ground truth.

Runs exactly the honest protocol from AGENTS.md §5:
  - collect_zone_samples(depth_matched=True)  (negatives at the same EMODnet
    depth, so the model learns rock-vs-sand, not shallow-vs-deep)
  - leave_one_zone_out()  (each zone scored by a model that never saw it)
  - AUC of a depth-only model reported alongside (the baseline to beat)

The `--gain-ci95` lower bound is the gate: <= 0 means the spectral/morpho
detector does not beat depth alone on that zone.

Usage:
    python scripts/eval_lozo_supervised.py [--imagery composite|sample|auto]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_reef_pipeline import ZONE_CONFIGS  # noqa: E402
from src.analysis.reef_segmenter import collect_zone_samples, leave_one_zone_out  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery", choices=["composite", "sample", "auto"], default="composite")
    args = parser.parse_args()

    for z_key, cfg in ZONE_CONFIGS.items():
        comp = cfg["scene_path"].with_name(cfg["scene_path"].name.replace("_sample.tif", "_composite.tif"))
        if args.imagery == "composite":
            if not comp.exists():
                raise SystemExit(f"{z_key}: composite missing ({comp}) — run scripts/build_clear_water_composite.py")
            cfg["scene_path"] = comp
        elif args.imagery == "sample":
            pass  # keep the single-scene path

    samples = collect_zone_samples(ZONE_CONFIGS, depth_matched=True)
    lozo = leave_one_zone_out(samples, n_boot=1000)

    print("\n=== Leave-One-Zone-Out (RF, depth-matched) — AUC modelo vs AUC só-profundidade ===\n")
    print(f"{'zona':<15}{'n_gt':>5}{'n_bg':>6}{'AUC':>8}{'AUC_dep':>9}{'gain':>8}{'IC95 gain':>22}")
    print("-" * 80)
    rows = {}
    for zone, r in lozo.items():
        lo, hi = r["gain_ci95"] if r["gain_ci95"] else (None, None)
        rows[zone] = r
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        ic = "—" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{zone:<15}{r['n_gt']:>5}{r['n_bg']:>6}{fmt(r['auc']):>8}{fmt(r['auc_depth_only']):>9}"
              f"{fmt(r['gain']):>8}{ic:>22}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "lozo_supervised_composite.json"
    with open(out_path, "w") as f:
        json.dump({z: {**r, "gain_ci95": list(r["gain_ci95"]) if r["gain_ci95"] else None} for z, r in rows.items()},
                  f, indent=2, default=str)
    print(f"\nGuardado em {out_path}")


if __name__ == "__main__":
    main()
