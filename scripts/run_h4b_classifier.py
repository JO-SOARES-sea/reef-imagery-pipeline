#!/usr/bin/env python3
"""
scripts/run_h4b_classifier.py — H4 revista: integração de todas as fontes de verdade (docs/HOLDOUT_PREREGISTO_v2.md).

Correções ao modo de falha da H4 congelada (ledger E013–E014):
  - B04/B08 deixam de ser feições; o B08 só exclui píxeis com perturbação de superfície persistente (|mediana z| > 1);
  - todas as feições vêm só da órbita R037 (a única comum a todas as zonas);
  - sem a feição H1 (não acrescentava, E006).
Rótulos de várias fontes, com peso por nível de evidência:
  - COSMO rocha estável vs liso (LTHQUGA1, LTHPEAL1, LTHBAIF1)          peso 1   (multifeixe, nível a)
  - cumes CCMAR do Valado                                                  peso 1   (multifeixe, nível a)
  - Teixeira 2025 (Fig. 20) rocha vs areia/lodo                            peso 0,3 (síntese, nível a ao nível de área)
  - sonar IH calibrado pelo COSMO (rocha ≥ 0,6 / liso ≤ 0,05, só ≥ 12 m)   peso 0,3 (estimativa, ledger E005)
  - sonda própria SL3 (rugosidade ≥ 0,15 m / ≤ 0,04 m ao longo do trilho)  peso 0,5 (sonda do Rui)
Validação leave-one-zone-out (ANCAO, SE, VALADO, FZW, IF); o holdout v2 (LTHNOCR1) é excluído por código de tudo.
O teste do holdout v2 faz-se à parte (scripts/final_holdout_v2.py), uma vez, com o modelo gravado aqui.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.reef_eval import depth_stratified_sample, evaluate, gain_block_ci, block_ids  # noqa: E402
from src.constants import HOLDOUT2_BBOX_WGS84  # noqa: E402

_s = importlib.util.spec_from_file_location("evd", PROJECT_ROOT / "scripts" / "eval_detectors.py")
evd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(evd)
PDIR, TRUTH = evd.PDIR, evd.TRUTH
OUT = PROJECT_ROOT / "outputs" / "eval"
SONAR = [PROJECT_ROOT / "outputs" / "sidescan" / d / "rock_probability.tif" for d in ("full_west", "full_east")]
SL3 = PROJECT_ROOT / "data" / "external" / "own_soundings" / "sl3_soundings.csv"

FEATURES = ["med", "absmed", "zA", "zB", "half_agree", "frac_dark", "frac_bright", "yearfrac_dark", "yearfrac_bright",
            "yearmed", "med_s15", "med_s30", "std5", "dog13", "lap2"]
B08_GATE = 1.0
ORBIT = 37
LABELS = {  # zona -> [(conjunto, peso)]
    "ancao": [("T3_QUGA1", 1.0), ("T2", 0.3), ("SONAR", 0.3)],
    "se": [("T3_PEAL1", 1.0)],
    "valado": [("T1", 1.0)],
    "fzw": [("T3_BAIF1", 1.0), ("SL3", 0.5)],
    "if": [("SL3", 0.5), ("SONAR", 0.3)],
}
TESTS = {"ancao": ["T3_QUGA1", "T2", "T2L"], "se": ["T3_PEAL1"], "valado": ["T1"], "fzw": ["T3_BAIF1"], "if": ["SL3"]}


def holdout2_mask(z: dict) -> np.ndarray:
    W, S, E, N = HOLDOUT2_BBOX_WGS84
    t = Transformer.from_crs(4326, 32629, always_xy=True)
    (x0, y0), (x1, y1) = t.transform(W, S), t.transform(E, N)
    rr, cc = np.indices(z["shape"])
    X, Y = z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def greta_corridor(z: dict) -> np.ndarray:
    """Corredor de 300 m à volta da crista da Greta (+1,5 km para NW): nunca entra nos rótulos de treino (critério 3)."""
    t = Transformer.from_crs(4326, 32629, always_xy=True)
    se_, nw = np.array(t.transform(-7.978251, 36.972730)), np.array(t.transform(-7.996629, 36.984876))
    L = np.linalg.norm(nw - se_)
    u = (nw - se_) / L
    nrm = np.array([-u[1], u[0]])
    rr, cc = np.indices(z["shape"])
    X, Y = z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10
    dl = np.abs((X - se_[0]) * nrm[0] + (Y - se_[1]) * nrm[1])
    al = (X - se_[0]) * u[0] + (Y - se_[1]) * u[1]
    return (dl < 300) & (al > -500) & (al < L + 1500)


def artificial_mask(z: dict) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize
    ps = gpd.read_file(evd.PSOEM).to_crs(32629)
    lith = gpd.read_file(evd.TEIXEIRA).to_crs(32629)
    art = list(ps.geometry) + list(lith[lith.layer.str.startswith("recifes artificiais")].geometry)
    return rasterize(((g, 1) for g in art), out_shape=z["shape"], transform=z["T"]).astype(bool)


def zone_features(zone: str) -> dict:
    """Feições só da órbita R037, recalculadas da pilha de desenvolvimento; B08 como gating."""
    z = evd.load_zone(zone, "_dev")
    # exclusões: recifes artificiais (PSOEM + tracejado Fig. 20) e o holdout v2; o holdout v1 (LTHBAIF1) foi gasto
    z["excl"] = artificial_mask(z) | holdout2_mask(z) | greta_corridor(z)
    passes = pd.read_csv(PDIR / f"{zone}_dev_stack_passes.csv")
    sel = passes.orbit.astype(str).str.lstrip("R").astype(int).to_numpy() == ORBIT
    years = passes.date.str[:4].astype(int).to_numpy()[sel]
    st = np.asarray(np.load(PDIR / f"{zone}_dev_stack_B03.npy", mmap_mode="r")[sel], np.float32)
    b8 = np.asarray(np.load(PDIR / f"{zone}_dev_stack_B08.npy", mmap_mode="r")[sel], np.float32)
    with np.errstate(all="ignore"):
        med = np.nanmedian(st, 0)
        A = np.nanmedian(st[years % 2 == 1], 0)
        B = np.nanmedian(st[years % 2 == 0], 0)
        n = np.isfinite(st).sum(0)
        fd = (st <= -2).sum(0) / np.maximum(n, 1)
        fb = (st >= 2).sum(0) / np.maximum(n, 1)
        ymeds = []
        for yv in np.unique(years):
            s_ = st[years == yv]
            ymeds.append(np.where(np.isfinite(s_).sum(0) >= 3, np.nanmedian(s_, 0), np.nan))
        Y = np.stack(ymeds)
        ny = np.isfinite(Y).sum(0)
        yfd = np.where(ny >= 3, np.nansum(Y <= -0.5, 0) / np.maximum(ny, 1), np.nan)
        yfb = np.where(ny >= 3, np.nansum(Y >= 0.5, 0) / np.maximum(ny, 1), np.nan)
        ymed = np.where(ny >= 3, np.nanmedian(Y, 0), np.nan)
        b8med = np.nanmedian(b8, 0)
    gate = np.abs(b8med) <= B08_GATE
    med = np.where(gate, med, np.nan)
    fill = np.nan_to_num(med)
    z["F"] = {"med": med, "absmed": np.abs(med), "zA": A, "zB": B,
              "half_agree": np.sign(A) * np.sign(B) * np.minimum(np.abs(A), np.abs(B)), "frac_dark": fd, "frac_bright": fb,
              "yearfrac_dark": yfd, "yearfrac_bright": yfb, "yearmed": ymed, "med_s15": ndi.gaussian_filter(fill, 1.5),
              "med_s30": ndi.gaussian_filter(fill, 3.0),
              "std5": np.sqrt(np.maximum(ndi.uniform_filter(fill ** 2, 5) - ndi.uniform_filter(fill, 5) ** 2, 0)),
              "dog13": ndi.gaussian_filter(fill, 1) - ndi.gaussian_filter(fill, 3), "lap2": ndi.gaussian_laplace(fill, 2)}
    z["n_r037"] = int(sel.sum())
    return z


def label_masks(name: str, z: dict) -> tuple[np.ndarray, np.ndarray]:
    if name in ("T1", "T2", "T2L", "T3_QUGA1", "T3_PEAL1"):
        return evd.truth_sets(name, z)
    if name == "T3_BAIF1":
        with rasterio.open(TRUTH / "LTHBAIF1_truth10m.tif") as src:
            cls = np.full(z["shape"], np.nan, np.float32)
            reproject(src.read(5), cls, src_transform=src.transform, src_crs=src.crs, dst_transform=z["T"], dst_crs="EPSG:32629",
                      resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        return cls == 1, cls == 0
    if name == "SONAR":
        p = np.full(z["shape"], np.nan, np.float32)
        for f in SONAR:
            if not f.exists():
                continue
            with rasterio.open(f) as src:
                tmp = np.full(z["shape"], np.nan, np.float32)
                reproject(src.read(1), tmp, src_transform=src.transform, src_crs=src.crs, dst_transform=z["T"], dst_crs="EPSG:32629",
                          resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
                p = np.where(np.isfinite(tmp), tmp, p)
        deep = z["depth"] >= 12  # além da cobertura COSMO: é aí que o sonar acrescenta
        return (p >= 0.6) & deep, (p <= 0.05) & deep
    if name == "SL3":
        d = pd.read_csv(SL3)
        x, y = Transformer.from_crs(4326, 32629, always_xy=True).transform(d.lon.values, d.lat.values)
        r = ((z["T"].f - y) // 10).astype(int)
        c = ((x - z["T"].c) // 10).astype(int)
        ok = (r >= 0) & (r < z["shape"][0]) & (c >= 0) & (c < z["shape"][1]) & np.isfinite(d.rough_m.values)
        rough = np.full(z["shape"], np.nan, np.float32)
        cnt = np.zeros(z["shape"], np.int32)
        for rr, cc, v in zip(r[ok], c[ok], d.rough_m.values[ok]):
            rough[rr, cc] = v if np.isnan(rough[rr, cc]) else max(rough[rr, cc], v)
            cnt[rr, cc] += 1
        return (rough >= 0.15) & (cnt >= 2), (rough <= 0.04) & (cnt >= 2)
    raise ValueError(name)


def samples(z: dict, name: str, seed: int, max_pos: int = 5000):
    pos, neg = label_masks(name, z)
    dep = z["depth"]
    ok = z["sea"].astype(bool) & ~z["excl"] & (dep >= 3) & (dep <= 25)
    for v in z["F"].values():
        ok &= np.isfinite(v)
    pi = np.flatnonzero((pos & ok).ravel())
    ni = np.flatnonzero((neg & ok & ~pos).ravel())
    if len(pi) < 5 or len(ni) < 25:
        return None
    d = dep.ravel()
    ip, ineg = depth_stratified_sample(d[pi], d[ni], 0.5, 5, max_pos=max_pos, seed=seed)
    if len(ip) < 5:
        return None
    idx = np.r_[pi[ip], ni[ineg]]
    y = np.r_[np.ones(len(ip), int), np.zeros(len(ineg), int)]
    X = np.column_stack([z["F"][f].ravel()[idx] for f in FEATURES])
    rr, cc = np.unravel_index(idx, z["shape"])
    return {"X": X, "y": y, "d": d[idx], "x": z["T"].c + (cc + .5) * 10, "yc": z["T"].f - (rr + .5) * 10,
            "base": -z["F"]["med"].ravel()[idx], "idx": idx}


def model(seed: int):
    return CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                                                 class_weight="balanced", random_state=seed), method="isotonic", cv=3)


def h4_frozen_lozo_scores(zone: str, test_name: str, idx: np.ndarray, seed: int):
    """Previsões da H4 congelada em leave-one-zone-out nos mesmos píxeis (critério 4 do pré-registo v2)."""
    _h = importlib.util.spec_from_file_location("h4", PROJECT_ROOT / "scripts" / "run_h4_classifier.py")
    h4 = importlib.util.module_from_spec(_h)
    _h.loader.exec_module(h4)
    zones = {zn: h4.zone_features(zn) for zn in h4.ZONE_SETS}
    tr = [zn for zn in zones if zn != zone]
    Xs, ys, ws = [], [], []
    for zn in tr:
        for s, w in h4.ZONE_SETS[zn]:
            X, y, *_ = h4.samples(zones[zn], s, seed)
            Xs.append(X); ys.append(y); ws.append(np.full(len(y), w))
    m = h4.model(seed).fit(np.concatenate(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    zt = zones[zone]
    Xt = np.column_stack([zt["F"][f].ravel()[idx] for f in h4.FEATURES])
    ok = np.isfinite(Xt).all(1)
    p = np.full(len(idx), np.nan)
    p[ok] = m.predict_proba(Xt[ok])[:, 1]
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="H4 revista: todas as fontes, leave-one-zone-out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="h4b_lozo_dev")
    ap.add_argument("--save-model", type=Path, default=None)
    a = ap.parse_args()
    zones = {zn: zone_features(zn) for zn in LABELS}
    print("passagens R037 por zona:", {zn: z["n_r037"] for zn, z in zones.items()})
    train = {}
    for zn, z in zones.items():
        parts = []
        for name, w in LABELS[zn]:
            s = samples(z, name, a.seed)
            if s is None:
                print(f"  {zn}/{name}: sem amostras suficientes")
                continue
            print(f"  {zn}/{name}: {int(s['y'].sum())} pos / {int((s['y'] == 0).sum())} neg (peso {w})")
            parts.append((s["X"], s["y"], np.full(len(s["y"]), w)))
        train[zn] = tuple(np.concatenate(v) for v in zip(*parts))
    results = {}
    for held in zones:
        tr = [zn for zn in zones if zn != held]
        m = model(a.seed).fit(np.concatenate([train[zn][0] for zn in tr]), np.concatenate([train[zn][1] for zn in tr]),
                              sample_weight=np.concatenate([train[zn][2] for zn in tr]))
        for name in TESTS[held]:
            s = samples(zones[held], name, a.seed)
            if s is None:
                continue
            pr = m.predict_proba(s["X"])[:, 1]
            r = evaluate(s["y"], pr, s["d"], s["x"], s["yc"], 500.0, baseline=s["base"], prob=pr, n_boot=1000, seed=a.seed).row()
            if name in ("T3_QUGA1", "T3_PEAL1"):
                ph4 = h4_frozen_lozo_scores(held, name, s["idx"], a.seed)
                ok = np.isfinite(ph4)
                r["gain_vs_h4_frozen"] = [round(v, 3) for v in gain_block_ci(s["y"][ok], pr[ok], ph4[ok], block_ids(s["x"][ok], s["yc"][ok], 500), 1000, a.seed)]
            results[f"{name}@{held}"] = r
            print(f"| {name} (treino {'+'.join(tr)}) | h4b | AUC {r['auc']} {r['auc_ci95']} | ganho vs mediana z R037 {r['gain_vs_baseline']} | "
                  f"vs H4 congelada {r.get('gain_vs_h4_frozen')} | PR-AUC {r['pr_auc']} | recall@FPR5 {r['recall_fpr5']} | ECE {r['ece']} | {r['by_depth']}")
    mfin = model(a.seed).fit(np.concatenate([train[zn][0] for zn in zones]), np.concatenate([train[zn][1] for zn in zones]),
                             sample_weight=np.concatenate([train[zn][2] for zn in zones]))
    if a.save_model:
        import joblib
        a.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": mfin, "features": FEATURES, "labels": LABELS, "orbit": ORBIT, "b08_gate": B08_GATE, "seed": a.seed,
                     "version": "h4b-2026-10-26"}, a.save_model)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.label}.json").write_text(json.dumps({"features": FEATURES, "labels": LABELS, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
