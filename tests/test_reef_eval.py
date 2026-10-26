"""Protocolo de avaliação comum (src/analysis/reef_eval.py) — offline, dados sintéticos."""

import numpy as np
import pytest

from src.analysis.reef_eval import (
    auc_block_ci,
    block_ids,
    depth_matched_negatives,
    ece,
    evaluate,
    gain_block_ci,
    recall_at_fpr,
)


def _data(seed=0, n=2000, signal=1.5):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 5000, n), rng.uniform(0, 5000, n)
    lab = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(size=n) + signal * lab
    depth = rng.uniform(3, 25, n)
    return lab, score, depth, x, y


def test_block_ids_group_nearby_points():
    b = block_ids(np.array([10, 490, 510]), np.array([10, 20, 30]), 500)
    assert b[0] == b[1] != b[2]


def test_auc_ci_contains_point_estimate_and_is_wider_with_bigger_blocks():
    lab, s, d, x, y = _data()
    auc, lo, hi = auc_block_ci(lab, s, block_ids(x, y, 250), n_boot=200)
    assert lo <= auc <= hi
    _, lo2, hi2 = auc_block_ci(lab, s, block_ids(x, y, 2500), n_boot=200)
    assert (hi2 - lo2) > (hi - lo)


def test_gain_of_identical_scores_is_zero():
    lab, s, d, x, y = _data()
    g, lo, hi = gain_block_ci(lab, s, s, block_ids(x, y, 500), n_boot=100)
    assert g == 0 and lo == 0 and hi == 0


def test_gain_detects_better_score():
    lab, s, d, x, y = _data()
    rng = np.random.default_rng(5)
    weak = s + rng.normal(0, 3, len(s))
    g, lo, hi = gain_block_ci(lab, s, weak, block_ids(x, y, 500), n_boot=200)
    assert lo > 0


def test_recall_at_fpr_bounds():
    lab, s, d, x, y = _data(signal=10)
    assert recall_at_fpr(lab, s, 0.05) > 0.95
    lab, s, d, x, y = _data(signal=0)
    assert recall_at_fpr(lab, s, 0.05) == pytest.approx(0.05, abs=0.03)


def test_ece_perfect_and_bad_calibration():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p * 0.3, 0, 1)) > 0.2


def test_depth_matched_negatives_within_tolerance():
    pos = np.array([10.0, 20.0])
    neg = np.linspace(0, 30, 301)
    idx = depth_matched_negatives(pos, neg, per_pos=5, tol_m=1.0)
    assert len(idx) == 10
    assert all(min(abs(neg[i] - p) for p in pos) <= 1.0 for i in idx)


def test_evaluate_reports_depth_bands_and_calibration():
    lab, s, d, x, y = _data()
    p = 1 / (1 + np.exp(-s))
    r = evaluate(lab, s, d, x, y, block_m=500, baseline=d, prob=p, n_boot=100).row()
    assert {"auc", "auc_ci95", "auc_depth_only", "gain_vs_baseline", "pr_auc", "recall_fpr5", "brier", "ece", "by_depth"} <= set(r)
    assert set(r["by_depth"]) <= {"3-10", "10-15", "15-20", "20-25"}
    assert r["auc"] > 0.75


def test_depth_stratified_sample_equalises_depth():
    from sklearn.metrics import roc_auc_score
    from src.analysis.reef_eval import depth_stratified_sample
    rng = np.random.default_rng(3)
    pos = rng.uniform(15, 25, 3000)          # positivos mais fundos
    neg = rng.uniform(3, 20, 30000)          # negativos sobretudo rasos
    pi, ni = depth_stratified_sample(pos, neg, 0.5, 5)
    y = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
    d = np.r_[pos[pi], neg[ni]]
    assert len(ni) == 5 * len(pi)
    assert abs(roc_auc_score(y, d) - 0.5) < 0.03   # a profundidade deixa de separar
