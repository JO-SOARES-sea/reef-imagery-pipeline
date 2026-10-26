#!/usr/bin/env python3
"""
scripts/run_h4_classifier.py — Hipótese H4: classificador multi-feição, validado deixando uma zona inteira de fora.

Achado que motiva as feições (ledger E008): "escuro = rocha" é só metade do sinal. Rocha baixa coberta de algas
(COSMO Quarteira/Albufeira) é escura no passa-alto B03; cristas de relevo grande (cumes do Valado) são claras porque
a coluna de água por cima é menor. As feições são, por isso, com sinal e em valor absoluto, multi-escala, e nunca
incluem a profundidade (que fica só para estratificar a amostragem).

Feições (todas da pilha de desenvolvimento 2017–2024): mediana z, |mediana z|, metades A/B, concordância de sinal das
metades, fração de dias escuros e claros (|z| ≥ 2), estabilidade inter-anual (H3), assinatura estática (H1), controlos
B04/B08, mediana z suavizada a 15 e 30 m, desvio local 5×5, DoG e laplaciano da mediana.

Rótulos por zona (verdade independente do satélite, docs/HOLDOUT_PREREGISTO.md):
  ANCAO  — T3_QUGA1 (COSMO, peso 1) e T2 Teixeira (peso 0,3, síntese generalizada);
  SE     — T3_PEAL1 (COSMO, peso 1);
  VALADO — T1 cumes CCMAR (peso 1).
Validação: treino em duas zonas, teste na terceira (leave-one-zone-out), com os conjuntos de teste dessa zona.
Calibração isotónica dentro do treino. A Greta (IF) nunca entra no treino: no fim, modelo treinado nas três zonas é
aplicado à IF e mede-se a fração da crista acima do limiar de operação (FPR 5 % no fundo da IF a 13–17 m).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.reef_eval import depth_stratified_sample, evaluate  # noqa: E402

_spec = importlib.util.spec_from_file_location("evd", PROJECT_ROOT / "scripts" / "eval_detectors.py")
evd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evd)
PDIR = evd.PDIR
OUT = PROJECT_ROOT / "outputs" / "eval"

ZONE_SETS = {"ancao": [("T3_QUGA1", 1.0), ("T2", 0.3)], "se": [("T3_PEAL1", 1.0)], "valado": [("T1", 1.0)]}
TEST_SETS = {"ancao": ["T3_QUGA1", "T2", "T2L"], "se": ["T3_PEAL1"], "valado": ["T1"]}
FEATURES = ["med", "absmed", "zA", "zB", "half_agree", "frac_dark", "frac_bright", "yearfrac", "yearmed", "h1",
            "b08", "b04", "med_s15", "med_s30", "std5", "dog13", "lap2"]


H2_FEATURES = ["h2", "h2_s30"]


def zone_features(zone: str, with_h2: bool = False) -> dict:
    z = evd.load_zone(zone, "_dev")
    z["excl"] = evd.exclusion_mask(z)
    med = z["ALL_med_B03"]
    fill = np.nan_to_num(med)
    stack = np.load(PDIR / f"{zone}_dev_stack_B03.npy", mmap_mode="r")
    n = np.zeros(med.shape, np.float32); bright = np.zeros(med.shape, np.float32)
    for i in range(stack.shape[0]):
        s = np.asarray(stack[i], np.float32)
        ok = np.isfinite(s); n += ok; bright += ok & (s >= 2.0)
    h1 = np.load(PDIR / f"{zone}_dev_h1.npz")["static"]
    h3 = np.load(PDIR / f"{zone}_dev_h3.npz")
    F = {"med": med, "absmed": np.abs(med), "zA": z["ALL_A_B03"], "zB": z["ALL_B_B03"],
         "half_agree": np.sign(z["ALL_A_B03"]) * np.sign(z["ALL_B_B03"]) * np.minimum(np.abs(z["ALL_A_B03"]), np.abs(z["ALL_B_B03"])),
         "frac_dark": z["ALL_frac_dark"], "frac_bright": bright / np.maximum(n, 1), "yearfrac": h3["yearfrac"], "yearmed": h3["yearmed"],
         "h1": h1, "b08": z["ALL_med_B08"], "b04": z["ALL_med_B04"],
         "med_s15": ndi.gaussian_filter(fill, 1.5), "med_s30": ndi.gaussian_filter(fill, 3.0),
         "std5": np.sqrt(np.maximum(ndi.uniform_filter(fill ** 2, 5) - ndi.uniform_filter(fill, 5) ** 2, 0)),
         "dog13": ndi.gaussian_filter(fill, 1) - ndi.gaussian_filter(fill, 3), "lap2": ndi.gaussian_laplace(fill, 2)}
    if with_h2:
        h2 = np.load(PDIR / f"{zone}_dev_h2.npz")["dii_anom"]
        F["h2"] = h2
        F["h2_s30"] = ndi.gaussian_filter(np.nan_to_num(h2), 3.0)
    z["F"] = F
    return z


def samples(z: dict, set_name: str, seed: int, max_pos: int = 5000):
    pos, neg = evd.truth_sets(set_name, z)
    dep = z["depth"]
    ok = z["sea"].astype(bool) & ~z["excl"] & (dep >= 3) & (dep <= 25) & np.isfinite(z["ALL_med_B03"])
    for v in z["F"].values():
        ok &= np.isfinite(v)
    pi = np.flatnonzero((pos & ok).ravel()); ni_pool = np.flatnonzero((neg & ok & ~pos).ravel())
    d = dep.ravel()
    ip, ineg = depth_stratified_sample(d[pi], d[ni_pool], 0.5, 5, max_pos=max_pos, seed=seed)
    idx = np.r_[pi[ip], ni_pool[ineg]]
    y = np.r_[np.ones(len(ip), int), np.zeros(len(ineg), int)]
    X = np.column_stack([z["F"][f].ravel()[idx] for f in feats(z)])
    rr, cc = np.unravel_index(idx, z["shape"])
    xs, ys = z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10
    return X, y, d[idx], xs, ys, -z["ALL_med_B03"].ravel()[idx]


def feats(z: dict) -> list[str]:
    return FEATURES + [f for f in H2_FEATURES if f in z["F"]]


def model(seed: int):
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def main() -> None:
    p = argparse.ArgumentParser(description="H4: classificador multi-feição leave-one-zone-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="h4_lozo_dev")
    p.add_argument("--with-h2", action="store_true")
    p.add_argument("--save-model", type=Path, default=None, help="Grava o modelo final (3 zonas) com joblib.")
    a = p.parse_args()
    zones = {zn: zone_features(zn, a.with_h2) for zn in ZONE_SETS}
    train_data = {}
    for zn, z in zones.items():
        parts = []
        for s, w in ZONE_SETS[zn]:
            X, y, *_ = samples(z, s, a.seed)
            parts.append((X, y, np.full(len(y), w)))
        train_data[zn] = tuple(np.concatenate(v) for v in zip(*parts))
    results = {}
    for held in zones:
        tr = [zn for zn in zones if zn != held]
        Xtr = np.concatenate([train_data[zn][0] for zn in tr]); ytr = np.concatenate([train_data[zn][1] for zn in tr])
        wtr = np.concatenate([train_data[zn][2] for zn in tr])
        m = model(a.seed).fit(Xtr, ytr, sample_weight=wtr)
        for s in TEST_SETS[held]:
            X, y, d, xs, ys, base = samples(zones[held], s, a.seed)
            pr = m.predict_proba(X)[:, 1]
            r = evaluate(y, pr, d, xs, ys, 500.0, baseline=base, prob=pr, n_boot=1000, seed=a.seed).row()
            r["train_zones"] = tr
            results[f"{s}@{held}"] = r
            print(f"| {s} (treino {'+'.join(tr)}) | h4 | AUC {r['auc']} {r['auc_ci95']} | ganho vs uv_median_z {r['gain_vs_baseline']} | "
                  f"PR-AUC {r['pr_auc']} | recall@FPR5 {r['recall_fpr5']} | ECE {r['ece']} | Brier {r['brier']} | {r['by_depth']}")
    # Greta: modelo nas três zonas, aplicado à IF (nunca vista)
    Xall = np.concatenate([train_data[zn][0] for zn in zones]); yall = np.concatenate([train_data[zn][1] for zn in zones])
    wall = np.concatenate([train_data[zn][2] for zn in zones])
    mfin = model(a.seed).fit(Xall, yall, sample_weight=wall)
    if a.save_model:
        import joblib
        a.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": mfin, "features": feats(zones["ancao"]), "zone_sets": ZONE_SETS, "seed": a.seed,
                     "with_h2": a.with_h2, "dev_date_range": "2017-01-01/2024-12-31"}, a.save_model)
    zif = zone_features("if", a.with_h2)
    ok = zif["sea"].astype(bool) & np.isfinite(zif["ALL_med_B03"])
    for v in zif["F"].values():
        ok &= np.isfinite(v)
    P = np.full(zif["shape"], np.nan, np.float32)
    idx = np.flatnonzero(ok.ravel())
    P.ravel()[idx] = mfin.predict_proba(np.column_stack([zif["F"][f].ravel()[idx] for f in feats(zif)]))[:, 1]
    np.save(PDIR / "if_dev_h4_prob.npy", P)
    t = Transformer.from_crs(4326, 32629, always_xy=True)
    se_ = np.array(t.transform(-7.978251, 36.972730)); nw = np.array(t.transform(-7.996629, 36.984876))
    L = np.linalg.norm(nw - se_); u = (nw - se_) / L; nrm = np.array([-u[1], u[0]])
    crest = []
    for s_ in np.arange(0, L + 1, 30):
        vals = []
        for o in np.arange(-50, 51, 10):
            x, y = se_ + u * s_ + nrm * o
            r_, c_ = int((zif["T"].f - y) // 10), int((x - zif["T"].c) // 10)
            if 0 <= r_ < P.shape[0] and 0 <= c_ < P.shape[1] and np.isfinite(P[r_, c_]):
                vals.append(P[r_, c_])
        crest.append(max(vals) if vals else np.nan)
    crest = np.array(crest)
    # limiar de operação: FPR 5 % no fundo da IF a 13–17 m, a > 300 m da crista
    dep = zif["depth"]
    far = np.ones(P.shape, bool)
    rr, cc = np.indices(P.shape)
    X_ = zif["T"].c + (cc + .5) * 10; Y_ = zif["T"].f - (rr + .5) * 10
    dist_line = np.abs((X_ - se_[0]) * nrm[0] + (Y_ - se_[1]) * nrm[1])
    along = (X_ - se_[0]) * u[0] + (Y_ - se_[1]) * u[1]
    far = ~((dist_line < 300) & (along > -500) & (along < L + 500))
    bg = P[ok & far & (dep >= 13) & (dep <= 17) & ~zif["excl"]]
    thr = float(np.nanquantile(bg, 0.95))
    greta = {"threshold_fpr5": thr, "crest_fraction_above": float(np.nanmean(crest > thr)), "n_points": int(np.isfinite(crest).sum())}
    print(f"Greta (nunca treinada): fração da crista acima do limiar FPR5 ({thr:.3f}) = {greta['crest_fraction_above']:.2f} ({greta['n_points']} pontos)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.label}.json").write_text(json.dumps({"features": feats(zones["ancao"]), "zone_sets": ZONE_SETS, "results": results, "greta": greta,
                                                     "seed": a.seed}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
