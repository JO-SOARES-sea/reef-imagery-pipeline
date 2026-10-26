#!/usr/bin/env python3
"""
scripts/run_h4c_regime.py — H4c: especialistas por regime de profundidade (ledger E018, declarado antes de correr).

Modo de falha da H4 congelada (E013–E014): a 3–7 m o vermelho (B04) e o NIR (B08) veem o fundo e os controlos de
superfície invertem-se. A H4b (E016) tirou-os de todo o lado e perdeu ~0,06 no COSMO. A H4c separa os regimes:
  - d ≥ SPLIT_M: H4 congelada, inalterada (mesmas feições e zonas de treino);
  - d <  SPLIT_M: especialista raso com o desenho da H4b (feições só R037, sem B04/B08/H1, gating B08), treinado só com
    COSMO rocha estável vs liso a 3–7 m (LTHQUGA1, LTHPEAL1, LTHBAIF1; o LTHBAIF1 é holdout v1 gasto: só treino).
  - Píxeis rasos sem feições válidas no especialista (gating B08) ficam com a H4.
Avaliação leave-one-zone-out nos mesmos píxeis da H4 congelada; critério 4 do pré-registo v2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.reef_eval import block_ids, evaluate, gain_block_ci  # noqa: E402


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h4 = _load("h4", "run_h4_classifier.py")
h4b = _load("h4b", "run_h4b_classifier.py")
OUT = PROJECT_ROOT / "outputs" / "eval"
SPLIT_M = 7.0  # E014: a 3–7 m o B04 vê o fundo; fixado antes de correr, sem afinação
SHALLOW_LABELS = {"ancao": "T3_QUGA1", "se": "T3_PEAL1", "fzw": "T3_BAIF1"}
TESTS = {"ancao": ["T3_QUGA1", "T2", "T2L"], "se": ["T3_PEAL1"], "valado": ["T1"], "fzw": ["T3_BAIF1"]}
DIAGNOSTIC_ONLY = {"T3_BAIF1"}  # holdout v1 gasto


def idx_from_xy(z: dict, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    c = np.round((xs - z["T"].c) / 10 - .5).astype(int)
    r = np.round((z["T"].f - ys) / 10 - .5).astype(int)
    return np.ravel_multi_index((r, c), z["shape"])


def predict_at(m, z: dict, feats: list[str], idx: np.ndarray) -> np.ndarray:
    X = np.column_stack([z["F"][f].ravel()[idx] for f in feats])
    ok = np.isfinite(X).all(1)
    p = np.full(len(idx), np.nan)
    if ok.any():
        p[ok] = m.predict_proba(X[ok])[:, 1]
    return p


def shallow_train(zb: dict, seed: int) -> dict:
    out = {}
    for zn, name in SHALLOW_LABELS.items():
        s = h4b.samples(zb[zn], name, seed)
        keep = s["d"] < SPLIT_M
        out[zn] = (s["X"][keep], s["y"][keep])
        print(f"  raso {zn}/{name}: {int(out[zn][1].sum())} rocha / {int((out[zn][1] == 0).sum())} liso (< {SPLIT_M} m)")
    return out


def deep_models(zh: dict, seed: int) -> dict:
    data = {}
    for zn, sets in h4.ZONE_SETS.items():
        parts = []
        for s, w in sets:
            X, y, *_ = h4.samples(zh[zn], s, seed)
            parts.append((X, y, np.full(len(y), w)))
        data[zn] = tuple(np.concatenate(v) for v in zip(*parts))
    models = {}
    for held in list(h4.ZONE_SETS) + ["fzw"]:
        tr = [zn for zn in h4.ZONE_SETS if zn != held]
        models[held] = h4.model(seed).fit(np.concatenate([data[zn][0] for zn in tr]), np.concatenate([data[zn][1] for zn in tr]),
                                          sample_weight=np.concatenate([data[zn][2] for zn in tr]))
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="H4c: especialistas por regime de profundidade, leave-one-zone-out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="h4c_lozo_dev")
    ap.add_argument("--frozen-h4", type=Path, default=PROJECT_ROOT / "models" / "frozen" / "h4_2026-10-26.joblib")
    ap.add_argument("--save-model", type=Path, default=None)
    a = ap.parse_args()
    zh = {zn: h4.zone_features(zn) for zn in TESTS}
    zb = {zn: h4b.zone_features(zn) for zn in TESTS}
    sh = shallow_train(zb, a.seed)
    deep = deep_models(zh, a.seed)
    results = {}
    for held, names in TESTS.items():
        tr = [zn for zn in sh if zn != held]
        ms = h4b.model(a.seed).fit(np.concatenate([sh[zn][0] for zn in tr]), np.concatenate([sh[zn][1] for zn in tr]))
        for name in names:
            if name in DIAGNOSTIC_ONLY:
                s = h4b.samples(zb[held], name, a.seed)
                y, d, xs, ys, base, idx = s["y"], s["d"], s["x"], s["yc"], s["base"], s["idx"]
            else:
                _, y, d, xs, ys, base = h4.samples(zh[held], name, a.seed)
                idx = idx_from_xy(zh[held], xs, ys)
            p_deep = predict_at(deep[held], zh[held], h4.FEATURES, idx)
            p_sh = predict_at(ms, zb[held], h4b.FEATURES, idx)
            use_sh = (d < SPLIT_M) & np.isfinite(p_sh)
            p = np.where(use_sh, p_sh, p_deep)
            ok = np.isfinite(p)
            r = evaluate(y[ok], p[ok], d[ok], xs[ok], ys[ok], 500.0, baseline=base[ok], prob=p[ok], n_boot=1000, seed=a.seed).row()
            ok2 = ok & np.isfinite(p_deep)
            r["gain_vs_h4_frozen"] = [round(v, 3) for v in gain_block_ci(y[ok2], p[ok2], p_deep[ok2], block_ids(xs[ok2], ys[ok2], 500), 1000, a.seed)]
            r["n_shallow_expert"] = int(use_sh.sum())
            r["n_shallow_fallback_h4"] = int(((d < SPLIT_M) & ~np.isfinite(p_sh)).sum())
            r["train_shallow"] = tr
            r["diagnostic_only"] = name in DIAGNOSTIC_ONLY
            results[f"{name}@{held}"] = r
            tag = " (só diagnóstico)" if name in DIAGNOSTIC_ONLY else ""
            print(f"| {name}{tag} | h4c | AUC {r['auc']} {r['auc_ci95']} | vs H4 congelada {r['gain_vs_h4_frozen']} | "
                  f"ganho vs mediana z {r['gain_vs_baseline']} | PR-AUC {r['pr_auc']} | recall@FPR5 {r['recall_fpr5']} | ECE {r['ece']} | "
                  f"raso {r['n_shallow_expert']} (+{r['n_shallow_fallback_h4']} H4) | {r['by_depth']}")
    crit4 = {k: results[k]["gain_vs_h4_frozen"][1] >= -0.02 for k in ("T3_QUGA1@ancao", "T3_PEAL1@se")}
    print("critério 4 (limite inferior ≥ −0,02):", crit4)
    if a.save_model:
        import joblib
        ms_fin = h4b.model(a.seed).fit(np.concatenate([v[0] for v in sh.values()]), np.concatenate([v[1] for v in sh.values()]))
        frozen = joblib.load(a.frozen_h4)
        a.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": "h4c_regime", "split_m": SPLIT_M, "deep": frozen["model"], "deep_features": frozen["features"],
                     "deep_source": str(a.frozen_h4), "shallow": ms_fin, "shallow_features": h4b.FEATURES, "shallow_labels": SHALLOW_LABELS,
                     "orbit_shallow": h4b.ORBIT, "b08_gate": h4b.B08_GATE, "seed": a.seed, "version": "h4c-2026-10-26"}, a.save_model)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.label}.json").write_text(json.dumps({"split_m": SPLIT_M, "results": results, "criterion_4": crit4}, indent=2, ensure_ascii=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
