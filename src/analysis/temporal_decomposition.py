"""
src/analysis/temporal_decomposition.py — Hipótese H1: fundo estático × visibilidade por passagem.

A pilha de passa-altos normalizados z_p(x) (passagem p, píxel x) é modelada como

    z_p(x) = v_p · s(x) + e_p(x)

em que s(x) é a assinatura estática do fundo (o que persiste), v_p ≥ 0 a visibilidade do fundo na passagem p
(0 = turva / mar picado, 1 ≈ dia típico limpo) e e_p(x) ruído esparso e pesado (ondas, esteiras, sunglint residual).
A mediana simples trata todas as passagens por igual; aqui as passagens em que o fundo não se vê (v_p ≈ 0) deixam de
diluir o sinal, e as que o mostram bem pesam mais.

Ajuste: fatorização de posto 1 ponderada e robusta por mínimos quadrados alternados, com pesos de Huber sobre os
resíduos (k = 1,5 σ robusto) e NaN tratados como pesos nulos. Parâmetros fixados a priori (não afinados em teste):
k = 1,5, 20 iterações, v normalizado para 1 no percentil 90 das passagens (E021).

Saídas: s (mapa estático, mesma escala z de uma passagem típica), v (visibilidade por passagem) e a fração de
variância explicada.

H1' (ledger E020): a mesma decomposição por faixa de profundidade (o v_p estima-se à parte em cada faixa, porque o
fundo a 20 m deixa de se ver antes do fundo a 5 m) e a mediana ponderada por um proxy de qualidade da passagem.
Cobertura explícita: um píxel com menos de ``min_valid`` passagens válidas fica NaN, sem interpolação.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HUBER_K = 1.5
N_ITER = 20


@dataclass
class Rank1Result:
    static: np.ndarray        # s(x), forma (linhas, colunas)
    visibility: np.ndarray    # v_p, forma (passagens,)
    explained: float          # fração da variância (robusta) explicada pelo posto 1
    coverage: np.ndarray | None = None  # n de passagens válidas por píxel


@dataclass
class BandedRank1Result:
    static: np.ndarray                       # s(x) montado faixa a faixa (NaN fora das faixas ou sem cobertura)
    visibility: dict = field(default_factory=dict)   # faixa (lo, hi) -> v_p
    explained: dict = field(default_factory=dict)    # faixa (lo, hi) -> fração explicada
    coverage: np.ndarray | None = None


def _huber_w(r: np.ndarray, scale: float, k: float = HUBER_K) -> np.ndarray:
    a = np.abs(r) / max(scale, 1e-9)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-9))


def _normalize_v(v: np.ndarray) -> np.ndarray:
    """v = 1 no percentil 90 (um dia limpo). A mediana dos v > 0 (versão de E006) cai num dia cego quando a maioria das
    passagens é cega e encolhe s sem limite; a ordenação dentro de uma faixa não muda, mas as faixas deixavam de ser
    comparáveis entre si (ledger E021)."""
    ref = np.percentile(v, 90)
    return v / ref if ref > 0 else v


def coverage_count(stack: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    """Número de passagens válidas (finitas) por píxel."""
    P = stack.shape[0]
    flat = stack.reshape(P, -1)
    n = np.empty(flat.shape[1], np.int32)
    for a0 in range(0, flat.shape[1], chunk):
        n[a0:a0 + chunk] = np.isfinite(np.asarray(flat[:, a0:a0 + chunk], np.float32)).sum(0)
    return n.reshape(stack.shape[1:])


def weighted_median(stack: np.ndarray, weights: np.ndarray, min_valid: int = 1, chunk: int = 100_000) -> np.ndarray:
    """
    Mediana ponderada por píxel ao longo das passagens (peso ``weights[p]`` ≥ 0), ignorando NaN.

    Definição: o menor valor cuja soma acumulada dos pesos (por ordem crescente) atinge metade do peso total.
    Píxeis com menos de ``min_valid`` passagens válidas (ou peso total nulo) ficam NaN.
    """
    P = stack.shape[0]
    w = np.asarray(weights, np.float64)
    if w.shape != (P,) or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights: um peso finito ≥ 0 por passagem")
    flat = stack.reshape(P, -1)
    out = np.full(flat.shape[1], np.nan, np.float32)
    for a0 in range(0, flat.shape[1], chunk):
        x = np.asarray(flat[:, a0:a0 + chunk], np.float64)
        ok = np.isfinite(x)
        order = np.argsort(np.where(ok, x, np.inf), axis=0, kind="stable")
        xs = np.take_along_axis(x, order, 0)
        ws = np.where(np.take_along_axis(ok, order, 0), w[order], 0.0)
        cw = np.cumsum(ws, 0)
        tot = cw[-1]
        k = np.argmax(cw >= 0.5 * tot[None, :], axis=0)
        val = xs[k, np.arange(x.shape[1])]
        good = (ok.sum(0) >= min_valid) & (tot > 0)
        out[a0:a0 + chunk] = np.where(good, val, np.nan)
    return out.reshape(stack.shape[1:])


def rank1_decompose(stack: np.ndarray, mask: np.ndarray | None = None, n_iter: int = N_ITER,
                    sample_px: int = 200_000, seed: int = 42, exclude: np.ndarray | None = None,
                    static_mask: np.ndarray | None = None, min_valid: int = 1) -> Rank1Result:
    """
    Decomposição robusta de posto 1 de uma pilha (passagens × linhas × colunas), com NaN permitidos.

    ``mask`` limita os píxeis usados para estimar v (por exemplo, só água 3–25 m); ``exclude`` retira píxeis dessa
    estimativa (holdouts, recifes artificiais, a vizinhança dos rótulos usados para validar o v). s é calculado em
    ``static_mask`` (todos por omissão) e fica NaN onde há menos de ``min_valid`` passagens válidas.
    Para velocidade, v estima-se numa amostra de ``sample_px`` píxeis; s usa todos os píxeis com v fixo.
    """
    P = stack.shape[0]
    flat = stack.reshape(P, -1)
    cov = coverage_count(stack)
    valid_px = cov.ravel() >= max(min_valid, 1)
    if mask is not None:
        valid_px &= mask.ravel()
    if exclude is not None:
        valid_px &= ~exclude.ravel()
    if not valid_px.any():
        raise ValueError("nenhum píxel disponível para estimar v")
    idx = np.flatnonzero(valid_px)
    rng = np.random.default_rng(seed)
    if len(idx) > sample_px:
        idx = rng.choice(idx, sample_px, replace=False)
    Z = flat[:, idx].astype(np.float32)  # P × n
    W0 = np.isfinite(Z).astype(np.float32)
    Zf = np.nan_to_num(Z)

    with np.errstate(all="ignore"):
        s = np.nanmedian(Z, axis=0)
    s = np.nan_to_num(s)
    v = np.ones(P, np.float32)
    for _ in range(n_iter):
        R = Zf - v[:, None] * s[None, :]
        scale = 1.4826 * np.median(np.abs(R[W0 > 0]))
        W = W0 * _huber_w(R, scale)
        # v_p: regressão ponderada de z_p em s (sem intercepto), v ≥ 0
        v = np.clip((W * Zf * s[None, :]).sum(1) / np.maximum((W * s[None, :] ** 2).sum(1), 1e-9), 0, None)
        v = _normalize_v(v)
        # s(x): regressão ponderada de z_.(x) em v
        s = (W * Zf * v[:, None]).sum(0) / np.maximum((W * v[:, None] ** 2).sum(0), 1e-9)

    # s para todos os píxeis com v fixo (uma passagem de reponderação), em blocos para limitar a memória
    s_full = np.full(flat.shape[1], np.nan, np.float32)
    R0 = Zf - v[:, None] * s[None, :]
    scale = 1.4826 * np.median(np.abs(R0[W0 > 0]))
    todo = np.ones(flat.shape[1], bool) if static_mask is None else static_mask.ravel().astype(bool)
    for a0 in range(0, flat.shape[1], 250_000):
        if not todo[a0:a0 + 250_000].any():
            continue
        blk = flat[:, a0:a0 + 250_000].astype(np.float32)
        Wb = np.isfinite(blk).astype(np.float32)
        Fb = np.nan_to_num(blk)
        with np.errstate(all="ignore"):
            sb = np.nan_to_num(np.nanmedian(np.where(Wb > 0, blk, np.nan), axis=0))
        Wr = Wb * _huber_w(Fb - v[:, None] * sb[None, :], scale)
        num = (Wr * Fb * v[:, None]).sum(0)
        den = (Wr * v[:, None] ** 2).sum(0)
        s_full[a0:a0 + 250_000] = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    s_full[~todo | (cov.ravel() < min_valid)] = np.nan
    s_full = s_full.reshape(stack.shape[1:])

    Rs = Zf - v[:, None] * s[None, :]
    tot = np.median(np.abs(Zf[W0 > 0] - np.median(Zf[W0 > 0])))
    res = np.median(np.abs(Rs[W0 > 0]))
    explained = float(1 - (res / max(tot, 1e-9)) ** 2)
    return Rank1Result(s_full.astype(np.float32), v.astype(np.float32), explained, cov)


def banded_rank1(stack: np.ndarray, depth: np.ndarray, bands, mask: np.ndarray | None = None,
                 exclude: np.ndarray | None = None, min_valid: int = 1, n_iter: int = N_ITER,
                 sample_px: int = 200_000, seed: int = 42) -> BandedRank1Result:
    """
    Posto 1 robusto faixa a faixa de profundidade: em cada faixa [lo, hi) o v_p estima-se só com os píxeis dessa faixa
    (``mask`` e ``exclude`` aplicam-se como em ``rank1_decompose``) e s é calculado só nela. Faixas sem píxeis
    disponíveis ficam NaN e sem v.
    """
    static = np.full(stack.shape[1:], np.nan, np.float32)
    out = BandedRank1Result(static=static)
    for lo, hi in bands:
        band = np.isfinite(depth) & (depth >= lo) & (depth < hi)
        if mask is not None:
            band_v = band & mask
        else:
            band_v = band
        try:
            r = rank1_decompose(stack, mask=band_v, n_iter=n_iter, sample_px=sample_px, seed=seed, exclude=exclude,
                                static_mask=band, min_valid=min_valid)
        except ValueError:
            continue
        static[band] = r.static[band]
        out.visibility[(lo, hi)] = r.visibility
        out.explained[(lo, hi)] = r.explained
        out.coverage = r.coverage
    return out
