"""
src/analysis/sounder_checks.py — Verificação com a sonda (protocolo E024, docs/PROTOCOLO_VERIFICACAO_SONDA_E024.md).

Funções puras, sem I/O, usadas por scripts/build_structures_db.py (``--blind-controls``) e por
scripts/ingest_sounder_checks.py:
  - sorteio dos controlos cegos à mesma profundidade (``pick_controls``);
  - correspondência marca ↔ trilho e rótulos do observador e do SL3 (``match_track``, ``observer_label``, ``sl3_label``);
  - métricas: score máximo 3×3, AUC com bootstrap estratificado por zona, IC de Wilson, κ de Cohen;
  - regra de decisão fixada no protocolo (``decide``).
Todos os limiares vêm do protocolo aprovado; não se ajustam depois de ver resultados.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

VISIT_RADIUS_M = 25.0
ROUGH_ROCK_M = 0.15
ROUGH_SMOOTH_M = 0.04
MIN_TRACK_POINTS = 2
CONTROL_DEPTH_TOL_M = 0.5
CONTROL_MIN_DIST_M = 150.0  # emenda E024a (era 300 m: só 17 de 30 controlos)
MIN_VISITED_PER_CLASS = 20
AUC_MIN = 0.70
OBSERVER_LABELS = {"pedra": 1, "misto": 1, "areia": 0, "lodo": 0}


def pick_controls(cand_depth, depth: np.ndarray, eligible: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  avoid_xy=None, tol_m: float = CONTROL_DEPTH_TOL_M, min_dist_m: float = CONTROL_MIN_DIST_M,
                  seed: int = 42) -> list[int | None]:
    """
    Um controlo por candidata: píxel elegível com |profundidade − candidata| ≤ ``tol_m``, a ≥ ``min_dist_m`` dos pontos
    em ``avoid_xy`` e dos controlos já escolhidos. Devolve o índice linear do píxel (ou None se não houver nenhum).
    ``eligible`` já traz as restrições espaciais (zona, estruturas, holdouts, recifes artificiais, scores baixos).
    """
    rng = np.random.default_rng(seed)
    flat_ok = np.flatnonzero(eligible.ravel() & np.isfinite(depth.ravel()))
    d, x, y = depth.ravel()[flat_ok], X.ravel()[flat_ok], Y.ravel()[flat_ok]
    taken = [tuple(p) for p in (avoid_xy if avoid_xy is not None else [])]
    out: list[int | None] = []
    for cd in cand_depth:
        pool = np.flatnonzero(np.abs(d - cd) <= tol_m)
        if taken and len(pool):
            dist, _ = cKDTree(np.asarray(taken, float)).query(np.column_stack([x[pool], y[pool]]))
            pool = pool[dist >= min_dist_m]
        if not len(pool):
            out.append(None)
            continue
        k = int(rng.choice(pool))
        out.append(int(flat_ok[k]))
        taken.append((x[k], y[k]))
    return out


def match_track(mark_xy: np.ndarray, track_xy: np.ndarray, rough=None, radius_m: float = VISIT_RADIUS_M):
    """Por marca: visitada (algum ponto do trilho a ≤ raio), n de pontos no raio e rugosidade máxima nesses pontos."""
    mark_xy = np.asarray(mark_xy, float).reshape(-1, 2)
    n = len(mark_xy)
    if track_xy is None or len(track_xy) == 0:
        return np.zeros(n, bool), np.zeros(n, int), np.full(n, np.nan)
    tree = cKDTree(np.asarray(track_xy, float))
    hits = tree.query_ball_point(mark_xy, radius_m)
    npts = np.array([len(h) for h in hits])
    rmax = np.full(n, np.nan)
    if rough is not None:
        rough = np.asarray(rough, float)
        for i, h in enumerate(hits):
            v = rough[h] if h else np.array([])
            v = v[np.isfinite(v)]
            if len(v):
                rmax[i] = v.max()
    return npts > 0, npts, rmax


def observer_label(values) -> np.ndarray:
    """pedra/misto = 1, areia/lodo = 0; nao_visto, vazio ou outro = NaN."""
    return np.array([OBSERVER_LABELS.get(str(v).strip().lower(), np.nan) if v is not None else np.nan for v in values], float)


def sl3_label(max_rough, npts) -> np.ndarray:
    r, n = np.asarray(max_rough, float), np.asarray(npts)
    out = np.full(r.shape, np.nan)
    ok = n >= MIN_TRACK_POINTS
    out[ok & (r >= ROUGH_ROCK_M)] = 1
    out[ok & (r <= ROUGH_SMOOTH_M)] = 0
    return out


def max3x3(arr: np.ndarray, rows, cols) -> np.ndarray:
    """Máximo (ignorando NaN) na vizinhança 3×3 de cada píxel; NaN fora da grelha ou sem valores."""
    H, W = arr.shape
    out = np.full(len(rows), np.nan)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if not (0 <= r < H and 0 <= c < W):
            continue
        win = arr[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
        if np.isfinite(win).any():
            out[i] = np.nanmax(win)
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(c - h), float(c + h))


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan")
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def auc_zone_bootstrap(y, score, zone, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """AUC e IC 95 % com bootstrap das marcas estratificado por zona."""
    y, s, zone = np.asarray(y, int), np.asarray(score, float), np.asarray(zone)
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(zone == z) for z in np.unique(zone)]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, (2.5, 97.5))
    return auc, float(lo), float(hi)


def decide(y, score, is_candidate, zone, n_boot: int = 1000, seed: int = 42) -> dict:
    """Regra do protocolo E024 para um método, nas marcas visitadas com rótulo principal."""
    y, s, cand = np.asarray(y, float), np.asarray(score, float), np.asarray(is_candidate, bool)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s, cand, zone = y[ok].astype(int), s[ok], cand[ok], np.asarray(zone)[ok]
    nc, nk = int(cand.sum()), int((~cand).sum())
    kc, kk = int(y[cand].sum()), int(y[~cand].sum())
    out = {"n_candidates": nc, "n_controls": nk, "rock_candidates": kc, "rock_controls": kk,
           "wilson_candidates": wilson_ci(kc, nc), "wilson_controls": wilson_ci(kk, nk)}
    if len(np.unique(y)) == 2:
        out["auc"], out["auc_lo"], out["auc_hi"] = auc_zone_bootstrap(y, s, zone, n_boot, seed)
    if nc < MIN_VISITED_PER_CLASS or nk < MIN_VISITED_PER_CLASS or "auc" not in out:
        out["decision"] = "descritivo (amostra abaixo do mínimo)"
        return out
    ok_auc = out["auc"] >= AUC_MIN and out["auc_lo"] > 0.5
    ok_prec = out["wilson_candidates"][0] > out["wilson_controls"][1]
    out["decision"] = "confirmada" if (ok_auc and ok_prec) else "não confirmada"
    return out
