#!/usr/bin/env python3
"""
scripts/run_h3_mobility.py — Hipótese H3: estabilidade inter-anual (a rocha não migra; as formas de areia sim).

A partir da pilha de desenvolvimento (2017–2024), calcula por píxel a mediana anual do z de B03 (anos com ≥ 3
passagens válidas no píxel) e duas variantes declaradas antes da avaliação:
  - h3_yearfrac: fração dos anos em que a mediana anual é ≤ −0,5 (escuro de forma recorrente);
  - h3_yearmed:  mediana das medianas anuais (cada ano pesa o mesmo, ao contrário da mediana de todas as passagens).
Grava `<zona>_h3.npz`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PDIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "persistence"
DARK_Z = -0.5
MIN_PASSES_YEAR = 3


def main() -> None:
    p = argparse.ArgumentParser(description="H3: estabilidade inter-anual da assinatura do fundo")
    p.add_argument("--zone", required=True)
    a = p.parse_args()
    stack = np.load(PDIR / f"{a.zone}_stack_B03.npy", mmap_mode="r")
    years = pd.read_csv(PDIR / f"{a.zone}_stack_passes.csv").date.str[:4].astype(int).to_numpy()
    yr_meds = []
    for yv in np.unique(years):
        sub = np.asarray(stack[years == yv], dtype=np.float32)
        n = np.isfinite(sub).sum(0)
        with np.errstate(all="ignore"):
            m = np.nanmedian(sub, axis=0)
        yr_meds.append(np.where(n >= MIN_PASSES_YEAR, m, np.nan))
    Y = np.stack(yr_meds)
    nyr = np.isfinite(Y).sum(0)
    with np.errstate(all="ignore"):
        frac = np.where(nyr >= 3, np.nansum(Y <= DARK_Z, axis=0) / np.maximum(nyr, 1), np.nan)
        ymed = np.where(nyr >= 3, np.nanmedian(Y, axis=0), np.nan)
    np.savez_compressed(PDIR / f"{a.zone}_h3.npz", yearfrac=frac.astype(np.float32), yearmed=ymed.astype(np.float32),
                        n_years=nyr.astype(np.int8))
    print(a.zone, "anos:", sorted(set(years.tolist())), "píxeis com ≥3 anos:", int((nyr >= 3).sum()))


if __name__ == "__main__":
    main()
