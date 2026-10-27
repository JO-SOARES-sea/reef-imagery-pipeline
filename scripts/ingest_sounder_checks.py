#!/usr/bin/env python3
"""
scripts/ingest_sounder_checks.py — Ingestão das verificações com a sonda (protocolo E024 + emenda E024a).

Entradas:
  --results  a lista cega devolvida (`verificacao_sonda_cega.csv` preenchida: hora_utc, profundidade_sonda_m,
             pedra_ou_areia, notas);
  --track    um ou mais CSV de trilho com colunas lon, lat e, se houver, rough_m (ex.: saída de
             scripts/extract_sl3_soundings.py). Sem trilho nenhuma marca conta como visitada (protocolo §4).
A chave (`verificacao_sonda_chave.csv`) diz qual marca é candidata e qual é controlo; só é lida aqui, em terra.

Saída: outputs/eval/e025_sonda_<label>.json (resultados para o ledger, E025 em diante). As marcas na caixa do
holdout v2 (VVA004) vão seladas para outputs/eval/sonda_selado_v2_<label>.csv e não entram em nenhuma métrica.
Nada aqui se ajusta aos resultados: limiares, raio e regra vêm de src/analysis/sounder_checks.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.sounder_checks import cohen_kappa, decide, match_track, max3x3, observer_label, sl3_label  # noqa: E402

_s = importlib.util.spec_from_file_location("evd", PROJECT_ROOT / "scripts" / "eval_detectors.py")
evd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(evd)
PDIR = evd.PDIR
KEY = PROJECT_ROOT / "data" / "external" / "persistent_structures" / "verificacao_sonda_chave.csv"
KAPPA_MIN = 0.4


def method_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score de cada método em cada marca: máximo na vizinhança 3×3 (protocolo §5)."""
    out = {m: np.full(len(df), np.nan) for m in ("h4_frozen", "uv_abs_median_z", "uv_median_z")}
    fwd = Transformer.from_crs(4326, 32629, always_xy=True)
    for zone, g in df.groupby("zona"):
        z = evd.load_zone(zone, "_dev")
        maps = {"h4_frozen": np.load(PDIR / f"{zone}_frozen_prob.npy"), "uv_abs_median_z": np.abs(z["ALL_med_B03"]),
                "uv_median_z": -z["ALL_med_B03"]}
        x, y = fwd.transform(g.lon.values, g.lat.values)
        r = ((z["T"].f - y) // 10).astype(int)
        c = ((x - z["T"].c) // 10).astype(int)
        pos = df.index.get_indexer(g.index)
        for m, arr in maps.items():
            out[m][pos] = max3x3(arr, r, c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingestão das verificações com a sonda (E024)")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--track", type=Path, action="append", default=[])
    ap.add_argument("--label", required=True, help="Etiqueta da saída, ex.: 2026-11-15")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    key = pd.read_csv(KEY)
    res = pd.read_csv(a.results, dtype={"pedra_ou_areia": str, "notas": str})
    df = key.merge(res[["nome_gpx", "hora_utc", "profundidade_sonda_m", "pedra_ou_areia", "notas"]], on="nome_gpx", how="left")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sealed = df[df.selado_holdout_v2]
    sealed.to_csv(a.out_dir / f"sonda_selado_v2_{a.label}.csv", index=False)
    df = df[~df.selado_holdout_v2].reset_index(drop=True)

    fwd = Transformer.from_crs(4326, 32629, always_xy=True)
    mx, my = fwd.transform(df.lon.values, df.lat.values)
    tracks = [pd.read_csv(t) for t in a.track]
    T = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame(columns=["lon", "lat", "rough_m"])
    tx, ty = fwd.transform(T.lon.values.astype(float), T.lat.values.astype(float)) if len(T) else (np.array([]), np.array([]))
    vis, npts, rmax = match_track(np.column_stack([mx, my]), np.column_stack([tx, ty]) if len(T) else None,
                                  T["rough_m"].values if "rough_m" in T else None)
    df["visitada"], df["n_pontos_trilho"], df["rugosidade_max_m"] = vis, npts, rmax
    df["rotulo_observador"] = np.where(vis, observer_label(df.pedra_ou_areia.tolist()), np.nan)
    df["rotulo_sl3"] = np.where(vis, sl3_label(rmax, npts), np.nan)
    for m, s in method_scores(df).items():
        df[f"score_{m}"] = s

    kappa = cohen_kappa(df.rotulo_observador, df.rotulo_sl3)
    cand = (df.tipo == "candidata").to_numpy()
    main_rows = ~df.diagnostico_holdout_v1.to_numpy()
    out = {"protocolo": "E024 + E024a", "label": a.label, "n_marcas": int(len(key)), "n_seladas_v2": int(len(sealed)),
           "n_visitadas": int(vis.sum()), "n_com_rotulo_observador": int(np.isfinite(df.rotulo_observador).sum()),
           "n_com_rotulo_sl3": int(np.isfinite(df.rotulo_sl3).sum()), "kappa_observador_sl3": None if np.isnan(kappa) else round(kappa, 3),
           "metodos": {}, "por_dominio": {}, "diagnostico_holdout_v1": df[~main_rows].drop(columns=["lat", "lon"]).to_dict("records")}
    labels = {"observador": df.rotulo_observador.to_numpy()}
    if np.isnan(kappa) or kappa < KAPPA_MIN:
        labels["sl3"] = df.rotulo_sl3.to_numpy()  # κ baixo ou indefinido: os dois rótulos lado a lado (protocolo §4)
    for lname, y in labels.items():
        for m in ("h4_frozen", "uv_abs_median_z", "uv_median_z"):
            s = df[f"score_{m}"].to_numpy()
            out["metodos"][f"{m}|{lname}"] = decide(y[main_rows], s[main_rows], cand[main_rows], df.zona.to_numpy()[main_rows], seed=a.seed)
    for dom, g in df[main_rows].groupby("dominio_validado"):
        i = g.index.to_numpy()
        out["por_dominio"][dom] = decide(labels["observador"][i], df.score_h4_frozen.to_numpy()[i], cand[i], df.zona.to_numpy()[i], seed=a.seed)
    df.to_csv(a.out_dir / f"e025_sonda_{a.label}_marcas.csv", index=False)
    (a.out_dir / f"e025_sonda_{a.label}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "diagnostico_holdout_v1"}, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
