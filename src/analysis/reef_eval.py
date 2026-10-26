"""
src/analysis/reef_eval.py — Protocolo de avaliação comum do detetor de recifes (docs/HOLDOUT_PREREGISTO.md, §3).

Todas as comparações entre métodos passam por aqui, para que usem o mesmo protocolo:
  - AUC com IC 95 % por bootstrap de blocos espaciais (píxeis vizinhos não são independentes);
  - ganho de AUC sobre uma linha de base, emparelhado (as mesmas amostras e os mesmos blocos em cada réplica);
  - PR-AUC e recall a uma taxa de falsos positivos fixa;
  - calibração: Brier e ECE (classes de largura igual);
  - amostragem de negativos emparelhada por profundidade;
  - resultados por faixa de profundidade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEPTH_BANDS_M = ((3, 10), (10, 15), (15, 20), (20, 25))


def block_ids(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    """Identificador de bloco espacial quadrado (coordenadas métricas)."""
    bx = np.floor(np.asarray(x) / block_m).astype(np.int64)
    by = np.floor(np.asarray(y) / block_m).astype(np.int64)
    return bx * 1_000_003 + by


def _resample_blocks(y: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma réplica bootstrap: blocos de positivos e de negativos reamostrados em separado."""
    out = []
    for cls in (0, 1):
        b = blocks[y == cls]
        ub = np.unique(b)
        pick = rng.choice(ub, len(ub))
        idx_cls = np.flatnonzero(y == cls)
        by_block = {k: idx_cls[b == k] for k in ub}
        out.append(np.concatenate([by_block[k] for k in pick]))
    return np.concatenate(out)


def auc_block_ci(y, score, blocks, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """AUC e IC 95 % por bootstrap de blocos."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    blocks = np.asarray(blocks)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _resample_blocks(y, blocks, rng)
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, (2.5, 97.5))
    return float(roc_auc_score(y, score)), float(lo), float(hi)


def gain_block_ci(y, score, baseline, blocks, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Ganho de AUC (score − linha de base) emparelhado nas mesmas réplicas de blocos."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, float)
    b = np.asarray(baseline, float)
    blocks = np.asarray(blocks)
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n_boot):
        idx = _resample_blocks(y, blocks, rng)
        g.append(roc_auc_score(y[idx], s[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(g, (2.5, 97.5))
    return float(roc_auc_score(y, s) - roc_auc_score(y, b)), float(lo), float(hi)


def recall_at_fpr(y, score, fpr: float = 0.05) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(score, float)
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float(np.mean(s[y == 1] > thr))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error com classes de largura igual em [0, 1]."""
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, float), 0, 1)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        m = bins == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def depth_matched_negatives(pos_depth, neg_depth, per_pos: int, tol_m: float, seed: int = 42) -> np.ndarray:
    """Índices de negativos com profundidade a ± ``tol_m`` de cada positivo (sem repetição)."""
    rng = np.random.default_rng(seed)
    neg_depth = np.asarray(neg_depth, float)
    order = np.argsort(neg_depth)
    sd = neg_depth[order]
    chosen: set[int] = set()
    for d in np.asarray(pos_depth, float):
        lo, hi = np.searchsorted(sd, [d - tol_m, d + tol_m])
        if hi > lo:
            for i in rng.choice(np.arange(lo, hi), size=min(per_pos, hi - lo), replace=False):
                chosen.add(int(order[i]))
    return np.array(sorted(chosen), dtype=int)


@dataclass
class EvalResult:
    n_pos: int
    n_neg: int
    n_blocks: int
    auc: float
    auc_ci: tuple[float, float]
    auc_depth_only: float
    gain_vs_baseline: tuple[float, float, float] | None
    pr_auc: float
    recall_fpr5: float
    brier: float | None
    ece: float | None
    by_depth: dict

    def row(self) -> dict:
        r = {"n_pos": self.n_pos, "n_neg": self.n_neg, "n_blocks": self.n_blocks, "auc": round(self.auc, 3),
             "auc_ci95": [round(v, 3) for v in self.auc_ci], "auc_depth_only": round(self.auc_depth_only, 3),
             "pr_auc": round(self.pr_auc, 3), "recall_fpr5": round(self.recall_fpr5, 3)}
        if self.gain_vs_baseline:
            r["gain_vs_baseline"] = [round(v, 3) for v in self.gain_vs_baseline]
        if self.brier is not None:
            r["brier"] = round(self.brier, 4)
            r["ece"] = round(self.ece, 4)
        r["by_depth"] = self.by_depth
        return r


def evaluate(y, score, depth, x, y_coord, block_m: float = 500.0, baseline=None, prob=None,
             n_boot: int = 1000, seed: int = 42) -> EvalResult:
    """Avaliação completa de um score (maior = mais provável rocha) contra rótulos binários."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    depth = np.asarray(depth, float)
    blocks = block_ids(x, y_coord, block_m)
    auc, lo, hi = auc_block_ci(y, score, blocks, n_boot, seed)
    gain = gain_block_ci(y, score, baseline, blocks, n_boot, seed) if baseline is not None else None
    by = {}
    for a, b in DEPTH_BANDS_M:
        m = (depth >= a) & (depth < b)
        if m.sum() and min((y[m] == 1).sum(), (y[m] == 0).sum()) >= 10:
            by[f"{a}-{b}"] = {"auc": round(float(roc_auc_score(y[m], score[m])), 3), "n_pos": int((y[m] == 1).sum()),
                              "n_neg": int((y[m] == 0).sum())}
    brier = e = None
    if prob is not None:
        prob = np.asarray(prob, float)
        brier = float(np.mean((prob - y) ** 2))
        e = ece(y, prob)
    return EvalResult(int((y == 1).sum()), int((y == 0).sum()), int(len(np.unique(blocks))), auc, (lo, hi),
                      float(roc_auc_score(y, depth)) if len(np.unique(depth)) > 1 else 0.5, gain,
                      float(average_precision_score(y, score)), recall_at_fpr(y, score), brier, e, by)


def depth_stratified_sample(pos_depth, neg_depth, bin_m: float = 0.5, ratio: int = 5, max_pos: int | None = None,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Amostra positivos e negativos com a MESMA distribuição de profundidade (classes de ``bin_m``).

    Em cada classe ficam k positivos e ratio·k negativos, com k = min(positivos, negativos // ratio). Mais estrito
    do que o emparelhamento ± tolerância, em que os negativos podem concentrar-se num dos lados da janela e deixar
    a profundidade separar as classes. Devolve os índices (em pos_depth e em neg_depth).
    """
    rng = np.random.default_rng(seed)
    pd_, nd_ = np.asarray(pos_depth, float), np.asarray(neg_depth, float)
    pb, nb = np.floor(pd_ / bin_m).astype(int), np.floor(nd_ / bin_m).astype(int)
    pi_out, ni_out = [], []
    for b in np.unique(pb):
        pi, ni = np.flatnonzero(pb == b), np.flatnonzero(nb == b)
        k = min(len(pi), len(ni) // ratio)
        if k == 0:
            continue
        pi_out.append(rng.choice(pi, k, replace=False))
        ni_out.append(rng.choice(ni, k * ratio, replace=False))
    if not pi_out:
        return np.array([], int), np.array([], int)
    pi, ni = np.concatenate(pi_out), np.concatenate(ni_out)
    if max_pos and len(pi) > max_pos:
        keep = rng.choice(len(pi), max_pos, replace=False)
        sel_bins = pb[pi[keep]]
        pi = pi[keep]
        # reamostra os negativos para manter ratio por classe
        ni = np.concatenate([rng.choice(ni[nb[ni] == b], ratio * int((sel_bins == b).sum()), replace=False) for b in np.unique(sel_bins)])
    return pi, ni
