#!/usr/bin/env python3
"""
scripts/eval_detectors.py — Avaliação comum dos detetores contra os conjuntos de teste pré-registados.

Conjuntos (docs/HOLDOUT_PREREGISTO.md, docs/PLANO_SALTO_QUALITATIVO.md §5), na grelha de 10 m de cada zona:
  T1  VALADO  cumes CCMAR (multifeixe) — píxel do cume vs água a ≥ 150 m, profundidade emparelhada;
  T2  ANCAO   litologia de Teixeira 2025 (Fig. 20): rocha (núcleo a > 20 m do bordo) vs areia/lodo mapeados;
  T2L ANCAO   rocha vs só lodos (o confundidor escuro);
  T3  ANCAO/IF/SE  COSMO LTHQUGA1 / LTHFA1 / LTHPEAL1: rocha estável vs fundo liso (scripts/cosmo_truth_rasters.py).
Excluídos sempre: holdouts espaciais v1 e v2 (src.constants.HOLDOUT_BBOX_WGS84, HOLDOUT2_BBOX_WGS84) e recifes
artificiais (PSOEM lyr13 + tracejado da Fig. 20). Negativos estratificados por profundidade (classes de 0,5 m, 5 por
positivo); blocos de 500 m.

Opções do H1' (ledger E020): ``--compare-to`` (ganho emparelhado contra várias referências), ``--calibrate-lozo``
(Brier e ECE depois de calibração isotónica leave-one-zone-out) e ``--primary``/``--free-refs``/``--h4-ref`` (regra de
decisão declarada).

Métodos: as linhas de base pré-registadas e qualquer método novo registado em METHODS (função zona → mapa de score,
maior = mais provável rocha). Resultados em outputs/eval/<etiqueta>.json e uma linha por (método, conjunto) no
stdout para o ledger.

Uso:
    python scripts/eval_detectors.py --maps-suffix _dev --methods uv_median_z,depth --label baselines_dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.reef_eval import depth_stratified_sample, evaluate  # noqa: E402
from src.analysis.reef_eval import block_ids, gain_block_ci  # noqa: E402
from src.constants import HOLDOUT2_BBOX_WGS84, HOLDOUT_BBOX_WGS84  # noqa: E402

PDIR = PROJECT_ROOT / "data" / "processed" / "persistence"
TRUTH = PROJECT_ROOT / "data" / "processed" / "cosmo_truth"
TEIXEIRA = PROJECT_ROOT / "data" / "external" / "eia3587" / "fig20_teixeira2025_layers.geojson"
PSOEM = PROJECT_ROOT / "data" / "external" / "psoem" / "lyr13_complexos_recifais_faro_quarteira.geojson"
CUMES = PROJECT_ROOT / "data" / "vector" / "valado_cumes_ccmar.csv"
OUT = PROJECT_ROOT / "outputs" / "eval"

ZONE_OF_SET = {"T1": "valado", "T2": "ancao", "T2L": "ancao", "T3_QUGA1": "ancao", "T3_FA1": "if", "T3_PEAL1": "se"}
COSMO_OF_SET = {"T3_QUGA1": "LTHQUGA1", "T3_FA1": "LTHFA1", "T3_PEAL1": "LTHPEAL1"}


def load_zone(zone: str, suffix: str) -> dict:
    meta = json.loads((PDIR / f"{zone}{suffix}_maps.json").read_text())
    z = dict(np.load(PDIR / f"{zone}{suffix}_maps.npz"))
    ny, nx = z["depth"].shape
    z["T"] = from_origin(meta["x0"], meta["y1"], 10, 10)
    z["shape"] = (ny, nx)
    z["meta"] = meta
    return z


def exclusion_mask(z: dict) -> np.ndarray:
    T, shape = z["T"], z["shape"]
    ps = gpd.read_file(PSOEM).to_crs(32629)
    lith = gpd.read_file(TEIXEIRA).to_crs(32629)
    art = lith[lith.layer.str.startswith("recifes artificiais")]
    from shapely.geometry import box
    geoms = list(ps.geometry) + list(art.geometry)
    for W, S, E, N in (HOLDOUT_BBOX_WGS84, HOLDOUT2_BBOX_WGS84):
        hb = gpd.GeoSeries.from_xy([W, E], [S, N], crs=4326).to_crs(32629)
        geoms.append(box(hb.x.min(), hb.y.min(), hb.x.max(), hb.y.max()))
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=T).astype(bool)


def truth_sets(name: str, z: dict) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de positivos e negativos candidatos (antes do emparelhamento por profundidade)."""
    T, shape = z["T"], z["shape"]
    if name in ("T2", "T2L"):
        lith = gpd.read_file(TEIXEIRA).to_crs(32629)
        r = lambda prefix: rasterize(((g, 1) for g in lith[lith.layer.str.startswith(prefix)].geometry), out_shape=shape, transform=T).astype(bool)
        rock, pq, art, mud = r("rocha"), r("plataforma"), r("recifes artificiais"), r("lodos")
        hull = rasterize([(lith.unary_union.convex_hull.buffer(-100), 1)], out_shape=shape, transform=T).astype(bool)
        pos = ndi.binary_erosion(rock, iterations=2)
        far = ~ndi.binary_dilation(rock | pq | art, iterations=2)
        neg = (mud if name == "T2L" else hull) & far
        return pos, neg
    if name == "T1":
        c = pd.read_csv(CUMES)
        x, y = Transformer.from_crs(4326, 32629, always_xy=True).transform(c.lon.values, c.lat.values)
        pos = np.zeros(shape, bool)
        for xi, yi in zip(x, y):
            r_, c_ = int((T.f - yi) // 10), int((xi - T.c) // 10)
            if 0 <= r_ < shape[0] and 0 <= c_ < shape[1]:
                pos[r_, c_] = True
        neg = ~ndi.binary_dilation(pos, iterations=15)
        return pos, neg
    sector = COSMO_OF_SET[name]
    with rasterio.open(TRUTH / f"{sector}_truth10m.tif") as src:
        cls = np.full(shape, np.nan, np.float32)
        reproject(src.read(5), cls, src_transform=src.transform, src_crs=src.crs, dst_transform=T, dst_crs="EPSG:32629",
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    return cls == 1, cls == 0


# ---- métodos: zona -> mapa de score (maior = mais provável rocha)
def m_uv_median_z(z):
    return -z["ALL_med_B03"]


def m_depth(z):
    return z["depth"]


def m_h1_static(z):
    """H1: assinatura estática da decomposição de posto 1 (scripts/run_h1_decomposition.py)."""
    return -np.load(PDIR / f"{z['meta']['name']}_h1.npz")["static"]


def m_h3_yearfrac(z):
    return np.load(PDIR / f"{z['meta']['name']}_h3.npz")["yearfrac"]


def m_h3_yearmed(z):
    return -np.load(PDIR / f"{z['meta']['name']}_h3.npz")["yearmed"]


def _h1p(z, key):
    """Produtos H1' da pilha alargada (<zona>_dev50, mesma grelha UV): run_h1_decomposition.py --banded."""
    f = np.load(PDIR / f"{z['meta']['name'].removesuffix('_dev')}_dev50_h1p.npz")[key]
    if f.shape != z["shape"]:
        raise ValueError("grelha H1' diferente da zona")
    return f


def m_uv_abs_median_z(z):
    """|mediana z| (E017): sinal bilateral, escuro ou claro."""
    return np.abs(z["ALL_med_B03"])


def m_h1p_abs(z):
    return np.abs(_h1p(z, "static"))


def m_h1p_neg(z):
    return -_h1p(z, "static")


def m_wmed_abs(z):
    return np.abs(_h1p(z, "wmed"))


def m_wmed_neg(z):
    return -_h1p(z, "wmed")


def m_h4_lozo(z):
    """H4 (receita congelada) treinada nas OUTRAS zonas de desenvolvimento e aplicada a esta (critério 4 do v2)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("h4", PROJECT_ROOT / "scripts" / "run_h4_classifier.py")
    h4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h4)
    zone = z["meta"]["name"].removesuffix("_dev")
    Xs, ys, ws = [], [], []
    for zn, sets in h4.ZONE_SETS.items():
        if zn == zone:
            continue
        zt = h4.zone_features(zn)
        for s, w in sets:
            X, y, *_ = h4.samples(zt, s, 42)
            Xs.append(X); ys.append(y); ws.append(np.full(len(y), w))
    m = h4.model(42).fit(np.concatenate(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    zf = h4.zone_features(zone)
    X = np.column_stack([zf["F"][f].ravel() for f in h4.FEATURES])
    ok = np.isfinite(X).all(1) & zf["sea"].ravel().astype(bool)
    out = np.full(X.shape[0], np.nan, np.float32)
    out[ok] = m.predict_proba(X[ok])[:, 1]
    return out.reshape(z["shape"])


METHODS = {"uv_median_z": m_uv_median_z, "depth": m_depth, "h1_static": m_h1_static,
           "h3_yearfrac": m_h3_yearfrac, "h3_yearmed": m_h3_yearmed, "uv_abs_median_z": m_uv_abs_median_z,
           "h1p_abs": m_h1p_abs, "h1p_neg": m_h1p_neg, "wmed_abs": m_wmed_abs, "wmed_neg": m_wmed_neg, "h4_lozo": m_h4_lozo}


def register_raster_method(name: str, pattern: str) -> None:
    """Regista um método lido de um raster por zona (ex.: sdb_anomaly da composição), reamostrado à grelha da zona."""
    def f(z, pattern=pattern):
        path = Path(pattern.format(zone=z["meta"]["name"].replace("_dev", "")))
        out = np.full(z["shape"], np.nan, np.float32)
        with rasterio.open(path) as src:
            reproject(src.read(1), out, src_transform=src.transform, src_crs=src.crs, dst_transform=z["T"],
                      dst_crs="EPSG:32629", resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        return out
    METHODS[name] = f


def main() -> None:
    p = argparse.ArgumentParser(description="Avaliação comum dos detetores (T1–T3)")
    p.add_argument("--maps-suffix", default="_dev")
    p.add_argument("--methods", default="uv_median_z,depth")
    p.add_argument("--sets", default="T1,T2,T2L,T3_QUGA1,T3_FA1,T3_PEAL1")
    p.add_argument("--baseline", default="uv_median_z", help="Método de referência para o ganho emparelhado.")
    p.add_argument("--raster-method", action="append", default=[], help="nome=padrão com {zone}, ex.: sdb=data/x/{zone}_sdb.tif")
    p.add_argument("--min-depth", type=float, default=3.0)
    p.add_argument("--max-depth", type=float, default=25.0)
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-to", default="", help="Referências para ganho emparelhado, separadas por vírgulas.")
    p.add_argument("--calibrate-lozo", default="", help="Conjuntos densos para a calibração isotónica leave-one-zone-out.")
    p.add_argument("--primary", default=None, help="Método candidato da regra de decisão (E020).")
    p.add_argument("--free-refs", default="", help="Referências sem treino; conta a de maior AUC pontual em cada conjunto.")
    p.add_argument("--h4-ref", default=None, help="Referência do critério 4 do pré-registo v2.")
    p.add_argument("--decision-sets", default="T3_QUGA1,T3_PEAL1")
    a = p.parse_args()
    for spec in a.raster_method:
        n, pat = spec.split("=", 1)
        register_raster_method(n, pat)
    methods = a.methods.split(",")
    refs = [m for m in a.compare_to.split(",") if m]
    results, samples = {}, {}
    cache = {}
    for s in a.sets.split(","):
        zone = ZONE_OF_SET[s]
        if not (PDIR / f"{zone}{a.maps_suffix}_maps.npz").exists():
            print(f"{s}: sem mapas de {zone}{a.maps_suffix}"); continue
        if s.startswith("T3") and not (TRUTH / f"{COSMO_OF_SET[s]}_truth10m.tif").exists():
            print(f"{s}: sem verdade COSMO"); continue
        if zone not in cache:
            z = load_zone(zone, a.maps_suffix)
            z["excl"] = exclusion_mask(z)
            z["scores"] = {m: METHODS[m](z) for m in set(methods) | set(refs) | {a.baseline, "depth"}}
            cache[zone] = z
        z = cache[zone]
        pos, neg = truth_sets(s, z)
        dep = z["depth"]
        ok = z["sea"].astype(bool) & ~z["excl"] & (dep >= a.min_depth) & (dep <= a.max_depth)
        for m in z["scores"].values():
            ok &= np.isfinite(m)
        pi = np.flatnonzero((pos & ok).ravel())
        ni_pool = np.flatnonzero((neg & ok & ~pos).ravel())
        if len(pi) < 10 or len(ni_pool) < 50:
            print(f"{s}: positivos/negativos insuficientes ({len(pi)}/{len(ni_pool)})"); continue
        d = dep.ravel()
        # mesma distribuição de profundidade nas duas classes (classes de 0,5 m, 5 negativos por positivo)
        ip, ineg = depth_stratified_sample(d[pi], d[ni_pool], 0.5, 5, max_pos=5000, seed=a.seed)
        pi, ni = pi[ip], ni_pool[ineg]
        idx = np.r_[pi, ni]
        y = np.r_[np.ones(len(pi), int), np.zeros(len(ni), int)]
        rr, cc = np.unravel_index(idx, z["shape"])
        xs, ys = z["T"].c + (cc + .5) * 10, z["T"].f - (rr + .5) * 10
        base = z["scores"][a.baseline].ravel()[idx]
        samples[s] = {"zone": zone, "y": y, "sc": {m: z["scores"][m].ravel()[idx] for m in z["scores"]}}
        blocks = block_ids(xs, ys, 500.0)
        results[s] = {}
        for m in methods:
            sc = z["scores"][m].ravel()[idx]
            r = evaluate(y, sc, d[idx], xs, ys, 500.0, baseline=None if m == a.baseline else base, n_boot=1000, seed=a.seed).row()
            r["gain_vs"] = {ref: [round(v, 3) for v in gain_block_ci(y, sc, z["scores"][ref].ravel()[idx], blocks, 1000, a.seed)]
                            for ref in refs if ref != m}
            results[s][m] = r
            g = r.get("gain_vs_baseline")
            print(f"| {s} | {m} | AUC {r['auc']} {r['auc_ci95']} | só prof. {r['auc_depth_only']} | PR-AUC {r['pr_auc']} | "
                  f"recall@FPR5 {r['recall_fpr5']} | ganho vs {a.baseline} {g} | n {r['n_pos']}/{r['n_neg']}, {r['n_blocks']} blocos | {r['by_depth']}")
    if a.calibrate_lozo:
        from sklearn.isotonic import IsotonicRegression
        from src.analysis.reef_eval import ece
        dense = [s for s in a.calibrate_lozo.split(",") if s in samples]
        for s, smp in samples.items():
            tr = [k for k in dense if samples[k]["zone"] != smp["zone"]]
            if not tr:
                continue
            for m in methods:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(np.concatenate([samples[k]["sc"][m] for k in tr]), np.concatenate([samples[k]["y"] for k in tr]))
                pr = iso.predict(smp["sc"][m])
                results[s][m]["calibration_lozo"] = {"trained_on": tr, "brier": round(float(np.mean((pr - smp["y"]) ** 2)), 4),
                                                     "ece": round(float(ece(smp["y"], pr)), 4)}
    decision = None
    if a.primary:
        free = [m for m in a.free_refs.split(",") if m]
        decision = {"primary": a.primary, "sets": {}}
        for s in a.decision_sets.split(","):
            if s not in results:
                decision["sets"][s] = "sem resultados"
                continue
            best = max(free, key=lambda m: results[s][m]["auc"])
            g = results[s][a.primary]["gain_vs"][best]
            row = {"best_free_ref": best, "gain": g, "useful": bool(g[0] >= 0.02 and g[1] >= 0)}
            if a.h4_ref:
                g4 = results[s][a.primary]["gain_vs"][a.h4_ref]
                row["gain_vs_h4"] = g4
                row["criterion4"] = bool(g4[1] >= -0.02)
            decision["sets"][s] = row
        rows = [v for v in decision["sets"].values() if isinstance(v, dict)]
        decision["useful"] = bool(len(rows) == len(decision["sets"]) and all(r["useful"] for r in rows))
        decision["holdout_v2_candidate"] = bool(decision["useful"] and a.h4_ref and all(r["criterion4"] for r in rows))
        print("DECISÃO:", json.dumps(decision, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.label}.json").write_text(json.dumps({"args": vars(a), "results": results, "decision": decision}, indent=2, ensure_ascii=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
