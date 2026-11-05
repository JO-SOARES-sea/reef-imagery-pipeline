"""
src/analysis/teixeira_alignment.py — Alinhamento dos polígonos de rocha de Teixeira pela imagem de satélite.

A Fig. 20 (Projeto de Execução 2025, Teixeira et al. 2019) foi georreferenciada à mão e o GeoJSON não tem CRS,
logo pode estar deslocada. Estima-se a translação (dr, dc) que maximiza o resíduo de escurecimento sob a máscara
de rocha. O resíduo tira a dependência da profundidade, para o ajuste não seguir a batimetria.

Convenção: ``(dr, dc)`` é o deslocamento a aplicar à máscara (linhas crescem para sul); em metros UTM,
``dx = dc * res`` (este) e ``dy = -dr * res`` (norte).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate
from shapely import affinity
from shapely.geometry.base import BaseGeometry


def depth_detrend(score: np.ndarray, depth: np.ndarray, valid: np.ndarray, bin_m: float = 0.5) -> np.ndarray:
    """Subtrai ao ``score`` a mediana da sua classe de profundidade; píxeis inválidos ficam NaN."""
    ok = valid & np.isfinite(score) & np.isfinite(depth)
    out = np.full(score.shape, np.nan, dtype=np.float64)
    bins = np.floor(depth[ok] / bin_m).astype(np.int64)
    vals = score[ok].astype(np.float64)
    res = np.empty_like(vals)
    for b in np.unique(bins):
        sel = bins == b
        res[sel] = vals[sel] - np.median(vals[sel])
    out[ok] = res
    return out


def estimate_shift(
    rock: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
    max_shift_px: int = 30,
    min_coverage: float = 0.5,
) -> dict:
    """
    Procura o (dr, dc), com |dr|, |dc| <= ``max_shift_px``, que maximiza o resíduo médio sob ``rock`` deslocada.

    ``min_coverage``: fração mínima dos píxeis válidos sob a máscara (relativa ao deslocamento nulo), para o pico
    não ser um artefacto de a máscara sair da cena.
    """
    use = valid & np.isfinite(residual)
    r = np.where(use, residual, 0.0)
    v = use.astype(np.float64)
    m = rock.astype(np.float64)
    my, mx = m.shape
    num = correlate(r, m, mode="full", method="fft")
    den = correlate(v, m, mode="full", method="fft")

    def at(dr: int, dc: int) -> tuple[float, float]:
        return float(num[dr + my - 1, dc + mx - 1]), float(den[dr + my - 1, dc + mx - 1])

    _, den0 = at(0, 0)
    best = (None, None, -np.inf)
    for dr in range(-max_shift_px, max_shift_px + 1):
        for dc in range(-max_shift_px, max_shift_px + 1):
            n, d = at(dr, dc)
            if d < min_coverage * den0 or d <= 0:
                continue
            s = n / d
            if s > best[2]:
                best = (dr, dc, s)
    n0, d0 = at(0, 0)
    return {
        "dr": best[0],
        "dc": best[1],
        "score_at_peak": best[2],
        "score_at_zero": n0 / d0 if d0 > 0 else float("nan"),
    }


def pixel_shift_to_metres(dr: int, dc: int, res_m: float) -> tuple[float, float]:
    """(dr, dc) em píxeis -> (dx este, dy norte) em metros."""
    return float(dc * res_m), float(-dr * res_m)


def shift_polygons(geoms: list[BaseGeometry], dx_m: float, dy_m: float) -> list[BaseGeometry]:
    """Translada geometrias (coordenadas métricas) sem alterar forma nem área."""
    return [affinity.translate(g, xoff=dx_m, yoff=dy_m) for g in geoms]
