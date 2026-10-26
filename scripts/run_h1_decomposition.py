#!/usr/bin/env python3
"""
scripts/run_h1_decomposition.py — Hipótese H1 sobre uma pilha de desenvolvimento (fundo estático × visibilidade).

Lê `data/processed/persistence/<zona>_stack_B03.npy` (de build_persistence_map.py --save-stack), estima v apenas em
água de 3–25 m fora do holdout e dos recifes artificiais, e grava `<zona>_h1.npz` (static, visibility) e
`<zona>_h1_passes.csv` (visibilidade por passagem, para validar contra os rótulos de clareza do Zimbral).

Modo H1' (``--banded``, ledger E020), para as pilhas alargadas (``--min-valid 0.5``):
  - posto 1 robusto por órbita (só órbitas com ≥ ``--min-orbit-passes``) e por faixa de profundidade, com o v_p
    estimado à parte em cada faixa, fora dos recifes artificiais, das caixas dos holdouts v1 e v2 e (com
    ``--exclude-zimbral-km``) da vizinhança do Zimbral, cujos rótulos validam o v_p;
  - píxel com menos de ``--min-valid-px`` passagens válidas numa órbita → NaN nessa órbita; órbitas combinadas pela
    média ponderada pelo número de passagens válidas;
  - mediana ponderada (método A) com w_p = valid_frac / deep_noise.
  Grava `<zona>_h1p.npz` (static, wmed, coverage), `<zona>_h1p_passes.csv` (v_p por faixa) e `<zona>_h1p.json`.

Validação do v_p (``--validate-zimbral``, só avaliação): rótulos "pedra visível / não visível" do UV
(`reports/2026-09-26_zimbral_clarity/z3_labels.json`, só ≤ 2024 e só passagens da pilha) contra o v_p da faixa
10–15 m; AUC com IC por bootstrap de anos, matriz de confusão no limiar declarado e Spearman com o KD490 CMEMS da data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.temporal_decomposition import HUBER_K, N_ITER, banded_rank1, rank1_decompose, weighted_median  # noqa: E402
from src.analysis.reef_eval import DEPTH_BANDS_M  # noqa: E402

PDIR = PROJECT_ROOT / "data" / "processed" / "persistence"
ZIMBRAL_LONLAT = (-7.93774, 36.95927)  # centro da faixa (UV DL-022)
UV_ZIMBRAL = Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/reports/2026-09-26_zimbral_clarity")
VP_THRESHOLD = 1.5  # declarado em E020, antes de ver os rótulos


def validate_zimbral(a) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    lab = json.loads((UV_ZIMBRAL / "z3_labels.json").read_text())
    passes = pd.read_csv(PDIR / f"{a.zone}_h1p_passes.csv")
    col = "v_10_15"
    passes["platform"] = passes.id.str[:3]
    passes["key"] = passes.date + "_" + passes.platform + "_R" + passes.orbit.astype(int).map("{:03d}".format)
    g = passes.groupby("key", as_index=False).agg(v=(col, "mean"), date=("date", "first"), orbit=("orbit", "first"), n_items=("id", "size"))
    g["label"] = g.key.map(lambda k: lab.get(k, {}).get("label"))
    g = g[g.label.isin(["visible", "not_visible"]) & (g.date < "2025-01-01") & g.v.notna()].copy()
    g["y"] = (g.label == "visible").astype(int)
    g["year"] = g.date.str[:4].astype(int)
    out = {"zone": a.zone, "band": "10-15", "threshold": VP_THRESHOLD, "n_visible": int(g.y.sum()), "n_not_visible": int((g.y == 0).sum()),
           "n_by_orbit": {str(o): {"visible": int(d.y.sum()), "not_visible": int((d.y == 0).sum())} for o, d in g.groupby("orbit")},
           "n_dates_with_duplicate_items": int((g.n_items > 1).sum())}
    if g.y.nunique() == 2:
        auc = roc_auc_score(g.y, g.v)
        rng = np.random.default_rng(a.seed)
        years = g.year.unique()
        boots = []
        for _ in range(1000):
            pick = rng.choice(years, len(years))
            s = pd.concat([g[g.year == yv] for yv in pick])
            if s.y.nunique() == 2:
                boots.append(roc_auc_score(s.y, s.v))
        out["auc"] = round(float(auc), 3)
        out["auc_ci95_years"] = [round(float(q), 3) for q in np.percentile(boots, [2.5, 97.5])]
        out["auc_by_orbit"] = {str(o): round(float(roc_auc_score(d.y, d.v)), 3) for o, d in g.groupby("orbit") if d.y.nunique() == 2}
        pred = g.v >= VP_THRESHOLD
        out["confusion"] = {"tp": int((pred & (g.y == 1)).sum()), "fn": int((~pred & (g.y == 1)).sum()),
                            "fp": int((pred & (g.y == 0)).sum()), "tn": int((~pred & (g.y == 0)).sum())}
    cm = pd.read_csv(UV_ZIMBRAL / "z4_cmems_daily.csv")
    allp = passes[passes.date < "2025-01-01"].groupby("date", as_index=False)[col].mean().merge(cm, left_on="date", right_on="time")
    allp = allp[allp.KD490.notna() & allp[col].notna()]
    rho, pv = spearmanr(allp[col], allp.KD490)
    out["spearman_v_kd490"] = {"rho": round(float(rho), 3), "p": float(pv), "n_dates": int(len(allp))}
    dst = PROJECT_ROOT / "outputs" / "eval" / "h1prime_vp_zimbral.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


def run_banded(a) -> None:
    import importlib.util
    from pyproj import Transformer
    spec = importlib.util.spec_from_file_location("evd", PROJECT_ROOT / "scripts" / "eval_detectors.py")
    evd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evd)
    z = evd.load_zone(a.zone, "")
    excl = evd.exclusion_mask(z)  # recifes artificiais + holdout v1 + holdout v2
    if a.exclude_zimbral_km > 0:
        zx, zy = Transformer.from_crs(4326, 32629, always_xy=True).transform(*ZIMBRAL_LONLAT)
        rr, cc = np.indices(z["shape"])
        X, Y = z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10
        excl |= np.hypot(X - zx, Y - zy) <= a.exclude_zimbral_km * 1000
    sea = z["sea"].astype(bool)
    depth = z["depth"]
    passes = pd.read_csv(PDIR / f"{a.zone}_stack_passes.csv")
    stack = np.load(PDIR / f"{a.zone}_stack_B03.npy", mmap_mode="r")
    w = (passes.valid_frac / passes.deep_noise).to_numpy()
    wmed = weighted_median(stack, w, min_valid=a.min_valid_px)
    num = np.zeros(z["shape"], np.float64)
    den = np.zeros(z["shape"], np.float64)
    meta = {"zone": a.zone, "bands": DEPTH_BANDS_M, "huber_k": HUBER_K, "n_iter": N_ITER, "seed": a.seed, "min_valid_px": a.min_valid_px,
            "min_orbit_passes": a.min_orbit_passes, "exclude_zimbral_km": a.exclude_zimbral_km, "orbits": {},
            "n_excluded_px": int((excl & sea).sum())}
    for b in DEPTH_BANDS_M:
        passes[f"v_{b[0]}_{b[1]}"] = np.nan
    for orb, g in passes.groupby(passes.orbit.astype(int)):
        if len(g) < a.min_orbit_passes:
            meta["orbits"][str(orb)] = {"n_passes": int(len(g)), "used": False}
            continue
        sub = np.asarray(stack[g.index.to_numpy()], np.float32)
        r = banded_rank1(sub, depth, DEPTH_BANDS_M, mask=sea, exclude=excl, min_valid=a.min_valid_px, seed=a.seed)
        ok = np.isfinite(r.static)
        num[ok] += r.coverage[ok] * r.static[ok]
        den[ok] += r.coverage[ok]
        for b, v in r.visibility.items():
            passes.loc[g.index, f"v_{b[0]}_{b[1]}"] = v
        meta["orbits"][str(orb)] = {"n_passes": int(len(g)), "used": True,
                                    "explained": {f"{b[0]}-{b[1]}": round(e, 4) for b, e in r.explained.items()},
                                    "v_quantiles": {f"{b[0]}-{b[1]}": np.percentile(v, [10, 25, 50, 75, 90]).round(3).tolist()
                                                    for b, v in r.visibility.items()}}
        del sub
    static = np.where(den > 0, num / np.maximum(den, 1), np.nan).astype(np.float32)
    np.savez_compressed(PDIR / f"{a.zone}_h1p.npz", static=static, wmed=wmed.astype(np.float32), coverage=den.astype(np.int32))
    passes.to_csv(PDIR / f"{a.zone}_h1p_passes.csv", index=False)
    (PDIR / f"{a.zone}_h1p.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


def main() -> None:
    p = argparse.ArgumentParser(description="H1: decomposição robusta de posto 1 da pilha de passa-altos")
    p.add_argument("--zone", required=True, help="Prefixo da pilha, ex.: ancao_dev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--banded", action="store_true", help="Modo H1' (por órbita e faixa de profundidade, ledger E020).")
    p.add_argument("--min-valid-px", type=int, default=10)
    p.add_argument("--min-orbit-passes", type=int, default=20)
    p.add_argument("--exclude-zimbral-km", type=float, default=0.0)
    p.add_argument("--validate-zimbral", action="store_true", help="Só avaliação do v_p (depois de --banded).")
    a = p.parse_args()
    if a.validate_zimbral:
        validate_zimbral(a)
        return
    if a.banded:
        run_banded(a)
        return
    stack = np.load(PDIR / f"{a.zone}_stack_B03.npy", mmap_mode="r")
    z = np.load(PDIR / f"{a.zone}_maps.npz")
    mask = z["sea"].astype(bool) & (z["depth"] >= 3) & (z["depth"] <= 25)
    # nenhum píxel do holdout pré-registado entra na estimativa de v
    from pyproj import Transformer
    from src.constants import HOLDOUT_BBOX_WGS84
    meta = json.loads((PDIR / f"{a.zone}_maps.json").read_text())
    W, S, E, N = HOLDOUT_BBOX_WGS84
    t = Transformer.from_crs(4326, 32629, always_xy=True)
    (x0, y0), (x1, y1) = t.transform(W, S), t.transform(E, N)
    rr, cc = np.indices(mask.shape)
    X = meta["x0"] + (cc + .5) * 10; Y = meta["y1"] - (rr + .5) * 10
    mask &= ~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
    r = rank1_decompose(np.asarray(stack, dtype=np.float32), mask=mask, seed=a.seed)
    np.savez_compressed(PDIR / f"{a.zone}_h1.npz", static=r.static, visibility=r.visibility)
    passes = pd.read_csv(PDIR / f"{a.zone}_stack_passes.csv")
    passes["visibility_h1"] = r.visibility
    passes.to_csv(PDIR / f"{a.zone}_h1_passes.csv", index=False)
    meta = {"zone": a.zone, "huber_k": HUBER_K, "n_iter": N_ITER, "seed": a.seed, "explained": r.explained,
            "n_passes": int(len(r.visibility)), "v_quantiles": np.percentile(r.visibility, [10, 25, 50, 75, 90]).round(3).tolist(),
            "frac_v_below_0.3": float(np.mean(r.visibility < 0.3))}
    (PDIR / f"{a.zone}_h1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
