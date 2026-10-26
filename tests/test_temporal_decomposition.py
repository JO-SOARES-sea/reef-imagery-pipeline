"""H1 — decomposição robusta de posto 1 (src/analysis/temporal_decomposition.py), offline e sintética."""

import numpy as np

from src.analysis.temporal_decomposition import banded_rank1, rank1_decompose, weighted_median


def _stack(seed=0, P=40, H=60, W=60, frac_blind=0.5):
    rng = np.random.default_rng(seed)
    s = np.zeros((H, W), np.float32)
    s[20:25, 10:50] = -2.0                      # crista escura estreita
    v = np.where(rng.random(P) < frac_blind, 0.0, rng.uniform(0.6, 1.4, P)).astype(np.float32)
    Z = v[:, None, None] * s[None] + rng.normal(0, 1, (P, H, W)).astype(np.float32)
    spikes = rng.random(Z.shape) < 0.02          # ruído esparso (esteiras, espuma)
    Z[spikes] += rng.choice([-8, 8], spikes.sum())
    Z[rng.random(Z.shape) < 0.05] = np.nan       # nuvens / glint mascarados
    return Z, s, v


def test_recovers_static_better_than_median_when_many_blind_passes():
    Z, s, v = _stack()
    r = rank1_decompose(Z, n_iter=20)
    med = np.nanmedian(Z, axis=0)
    crest, bg = s < -1, s == 0
    snr = lambda m: (m[bg].mean() - m[crest].mean()) / m[bg].std()
    assert snr(r.static) > 1.3 * snr(med)


def test_visibility_tracks_truth():
    Z, s, v = _stack(seed=1)
    r = rank1_decompose(Z)
    assert np.corrcoef(r.visibility, v)[0, 1] > 0.9
    assert np.median(r.visibility[v == 0]) < 0.3


def test_handles_all_nan_pixels_and_shape():
    Z, s, v = _stack(seed=2, P=10, H=20, W=20)
    Z[:, 0, 0] = np.nan
    r = rank1_decompose(Z)
    assert r.static.shape == (20, 20) and r.visibility.shape == (10,)
    assert np.isnan(r.static[0, 0])


# ---- H1' (ledger E020): mediana ponderada, faixas de profundidade, cobertura


def test_weighted_median_matches_definition():
    st = np.array([1.0, 2.0, 3.0, 10.0], np.float32)[:, None, None] * np.ones((4, 2, 3), np.float32)
    eq = weighted_median(st, np.ones(4))
    assert np.allclose(eq, 2.0)                                  # menor valor que atinge metade do peso
    heavy = weighted_median(st, np.array([0.1, 0.1, 0.1, 5.0]))
    assert np.allclose(heavy, 10.0)                              # o peso dominante decide
    odd = np.array([5.0, 1.0, 3.0], np.float32)[:, None, None] * np.ones((3, 1, 1), np.float32)
    assert weighted_median(odd, np.ones(3))[0, 0] == np.median([5.0, 1.0, 3.0])


def test_weighted_median_ignores_nan_and_applies_min_valid():
    st = np.full((5, 1, 3), np.nan, np.float32)
    st[:, 0, 0] = [1, np.nan, 3, np.nan, 5]
    st[:2, 0, 1] = [7, 8]
    w = np.array([1.0, 100.0, 1.0, 100.0, 1.0])                  # pesos grandes caem em NaN e não contam
    out = weighted_median(st, w, min_valid=3)
    assert out[0, 0] == 3.0
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])           # 2 e 0 passagens válidas < 3


def test_banded_recovers_depth_dependent_visibility():
    rng = np.random.default_rng(3)
    P, H, W = 60, 80, 60
    depth = np.repeat(np.linspace(4, 24, H)[:, None], W, 1)
    s = np.zeros((H, W), np.float32)
    s[::7, 5:55] = -2.0
    s[3::7, 5:55] = 1.5
    v_sh = rng.uniform(0.5, 1.5, P)
    v_dp = np.where(rng.random(P) < 0.5, 0.0, rng.uniform(0.5, 1.5, P))   # o fundo fundo desaparece mais vezes
    v_map = np.where(depth[None] < 14, v_sh[:, None, None], v_dp[:, None, None])
    Z = (v_map * s[None] + rng.normal(0, 1, (P, H, W))).astype(np.float32)
    r = banded_rank1(Z, depth, [(3, 14), (14, 25)], seed=0)
    assert np.corrcoef(r.visibility[(3, 14)], v_sh)[0, 1] > 0.9
    assert np.corrcoef(r.visibility[(14, 25)], v_dp)[0, 1] > 0.9
    g = rank1_decompose(Z, seed=0)                                # um só v para tudo não segue as duas faixas
    assert np.corrcoef(g.visibility, v_dp)[0, 1] < np.corrcoef(r.visibility[(14, 25)], v_dp)[0, 1]
    assert np.isfinite(r.static[depth < 25]).all()


def test_exclusion_and_coverage_are_explicit():
    Z, s, v = _stack(seed=4, P=30, H=30, W=30)
    Z[:25, 0:3, :] = np.nan                                       # 5 passagens válidas nestas linhas
    excl = np.zeros((30, 30), bool)
    excl[20:25, :] = True                                         # a crista fica fora da estimativa de v
    r = rank1_decompose(Z, exclude=excl, min_valid=10)
    assert np.isnan(r.static[0:3]).all()                          # sem interpolação
    assert r.coverage[0, 0] <= 5 and r.coverage[10, 10] >= 25
    assert np.isfinite(r.static[10:20]).all()
    with np.testing.assert_raises(ValueError):
        rank1_decompose(Z, exclude=np.ones((30, 30), bool))


def test_deterministic_with_seed():
    Z, *_ = _stack(seed=5)
    depth = np.repeat(np.linspace(3, 20, Z.shape[1])[:, None], Z.shape[2], 1)
    a = banded_rank1(Z, depth, [(3, 10), (10, 20)], seed=7, sample_px=500)
    b = banded_rank1(Z, depth, [(3, 10), (10, 20)], seed=7, sample_px=500)
    assert np.array_equal(a.static, b.static, equal_nan=True)
    assert all(np.array_equal(a.visibility[k], b.visibility[k]) for k in a.visibility)


def test_majority_blind_passes():
    Z, s, v = _stack(seed=6, P=50, frac_blind=0.7)
    assert np.mean(v == 0) > 0.5
    r = rank1_decompose(Z)
    assert np.corrcoef(r.visibility, v)[0, 1] > 0.85
    crest, bg = s < -1, s == 0
    assert r.static[crest].mean() < r.static[bg].mean() - 1.0
