"""Protocolo E024 — verificação com a sonda (src/analysis/sounder_checks.py), offline e sintético."""

import numpy as np

from src.analysis.sounder_checks import (
    cohen_kappa, decide, match_track, max3x3, observer_label, pick_controls, sl3_label, wilson_ci,
)


def _grid(H=100, W=100):
    rr, cc = np.indices((H, W))
    return cc * 10.0 + 5, (H - rr) * 10.0 - 5, np.repeat(np.linspace(3, 20, H)[:, None], W, 1)


def test_controls_match_depth_and_keep_distance():
    X, Y, depth = _grid()
    eligible = np.ones(depth.shape, bool)
    eligible[:, :20] = False
    cands = [5.0, 5.0, 12.0, 30.0]                       # a última não tem profundidade possível
    avoid = [(500.0, 500.0)]
    idx = pick_controls(cands, depth, eligible, X, Y, avoid_xy=avoid, min_dist_m=300, seed=1)
    assert idx[-1] is None
    pts = [(X.ravel()[i], Y.ravel()[i]) for i in idx if i is not None]
    for i, cd in zip(idx[:3], cands[:3]):
        assert abs(depth.ravel()[i] - cd) <= 0.5 and eligible.ravel()[i]
    for a in range(len(pts)):
        assert np.hypot(pts[a][0] - 500, pts[a][1] - 500) >= 300
        for b in range(a):
            assert np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) >= 300
    assert idx == pick_controls(cands, depth, eligible, X, Y, avoid_xy=avoid, min_dist_m=300, seed=1)   # determinismo


def test_match_track_radius_and_roughness():
    marks = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]])
    track = np.array([[10.0, 0.0], [0.0, 20.0], [100.0, 24.0], [200.0, 0.0]])
    rough = np.array([0.2, 0.05, 0.03, 0.5])
    vis, n, rmax = match_track(marks, track, rough)
    assert vis.tolist() == [True, True, False]
    assert n.tolist() == [2, 1, 0]
    assert rmax[0] == 0.2 and rmax[1] == 0.03 and np.isnan(rmax[2])
    assert sl3_label(rmax, n)[0] == 1 and np.isnan(sl3_label(rmax, n)[1])   # 1 ponto só: indeterminado


def test_labels():
    lab = observer_label(["Pedra", "misto", "areia", "LODO", "nao_visto", "", None])
    assert lab[:4].tolist() == [1, 1, 0, 0] and np.isnan(lab[4:]).all()
    assert sl3_label([0.16, 0.04, 0.1], [3, 3, 3]).tolist()[:2] == [1, 0]
    assert np.isnan(sl3_label([0.1], [3])[0])


def test_max3x3_and_stats():
    a = np.zeros((5, 5)); a[2, 3] = 7; a[0, 0] = np.nan
    assert max3x3(a, [2, 0, 9], [2, 0, 0]).tolist()[:2] == [7.0, 0.0]
    assert np.isnan(max3x3(a, [9], [0])[0])
    lo, hi = wilson_ci(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.98
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_decision_rule():
    rng = np.random.default_rng(0)
    n = 30
    cand = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    y = np.r_[rng.random(n) < 0.8, rng.random(n) < 0.15].astype(float)
    s = y + rng.normal(0, 0.4, 2 * n)
    zone = np.tile(["a", "b", "c"], 2 * n // 3)
    good = decide(y, s, cand, zone, n_boot=200)
    assert good["decision"] == "confirmada"
    bad = decide(y, rng.normal(size=2 * n), cand, zone, n_boot=200)
    assert bad["decision"] == "não confirmada"
    few = decide(y[:30], s[:30], np.r_[np.ones(15, bool), np.zeros(15, bool)], zone[:30], n_boot=50)
    assert few["decision"].startswith("descritivo")
