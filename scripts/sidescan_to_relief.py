#!/usr/bin/env python3
"""
scripts/sidescan_to_relief.py — Rocha e relevo do fundo a partir do mosaico de sonar lateral do IH, calibrados pelo COSMO.

O mosaico de sonar de varrimento lateral do Instituto Hidrográfico (AIA 3587, Aditamento, Anexo D, Desenho 2; campanhas
2000–2017) mostra a rocha (textura, brilho e sombras acústicas), mas não tem cotas. Os métodos de reconstrução 3D a partir
de sonar lateral (sombras H = L·A/(R+L), shape-from-shading, Neural SfS de Bore & Folkesson 2023) precisam da geometria de
cada ping (altitude, alcance, navegação), que se perdeu no mosaico impresso (JPEG, ~2,1 m/píxel).

Por isso usa-se uma calibração empírica com verdade independente:
  1. Na faixa onde o sonar e o multifeixe COSMO LTHQUGA1 (APA, 0,30 m, até ~13 m ZH) se sobrepõem, o relevo local do
     COSMO (DEM − passa-baixo de 25 m, máximo por célula do sonar) é o alvo.
  2. Feições do sonar: intensidade normalizada pelo ganho ao longo do alcance, textura multi-escala, contraste brilho/sombra
     e derivada na direção transversal às passagens (sombras acústicas), contraste local.
  3. Modelos de gradient boosting avaliados por validação cruzada em blocos espaciais (sem píxeis vizinhos em treino e
     teste). Comparados com linhas de base (só intensidade).
  4. O modelo final é aplicado a uma zona alvo (por omissão, Greta/B-24) e gravado como GeoTIFF EPSG:3763 com a mesma
     convenção do COSMO (cotas negativas referidas ao ZH): probabilidade de rocha, relevo estimado e pseudo-DEM.

O pseudo-DEM é uma ESTIMATIVA (EMODnet + relevo previsto), não uma medição: resolução efetiva ~4–10 m, erro reportado em
`<out>/metrics.json`. Serve para localizar e priorizar cabeços, não para navegação.

Uso:
    python scripts/sidescan_to_relief.py --target-bbox 10200 -297900 12000 -296000 --out outputs/sidescan/greta_b24
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sidescan_to_relief")

SONAR = Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/data/raw/seabed/eia3587/sonar_ih_desenho2_tm06.tif")
COSMO = Path("/Volumes/4G_WD_BLACK/BACKUP/bulk_download_2026-09-28_04-23-11/LTHQUGA1_20250612_20250616_DEM.tif")
COSMO_2018 = Path("/Volumes/4G_WD_BLACK/BACKUP/bulk_download_2026-09-28_04-23-11/LTHQUGA1_20181003_20181013_DEM.tif")
STABLE_MAX_DZ_M = 0.3  # rocha = relevo >= 0,3 m em 2018 E 2025, com a cota a variar <= 0,3 m (DL-025 do UV)
SMOOTH_MAX_RELIEF_M = 0.15  # não rocha = fundo liso nos dois anos
EMODNET = Path(__file__).resolve().parent.parent / "data" / "processed" / "EMODnet_batimetria_Quarteira_Garrao.tif"

RELIEF_MIN_SPEARMAN = 0.3  # abaixo disto o relevo previsto não acompanha o COSMO e não é gravado
ROCK_RELIEF_M = 0.3  # célula com relevo local máximo >= 0,3 m no COSMO = rocha saliente (mesmo critério do DL-025 do UV)


def track_angle(img: np.ndarray, valid: np.ndarray) -> float:
    """Ângulo (graus, convenção de ndi.rotate) que põe as faixas de nadir na horizontal."""
    fill = np.where(valid, img, np.nanmean(img[valid]))
    h, w = fill.shape
    c = fill[h // 2 - min(h, 700) // 2: h // 2 + min(h, 700) // 2, w // 2 - min(w, 700) // 2: w // 2 + min(w, 700) // 2]
    best = (-1.0, 0.0)
    for ang in np.arange(-45, 45.1, 0.5):
        r = ndi.rotate(c, ang, reshape=False, order=1)[c.shape[0] // 4: 3 * c.shape[0] // 4, c.shape[1] // 4: 3 * c.shape[1] // 4]
        prof = r.mean(axis=1)
        sc = (prof - ndi.uniform_filter1d(prof, 60)).std()
        if sc > best[0]:
            best = (sc, ang)
    return best[1]


def sonar_features(img: np.ndarray, valid: np.ndarray, res: float, angle_deg: float) -> dict[str, np.ndarray]:
    """Feições por píxel: ganho normalizado, textura, contraste brilho/sombra e derivada transversal às passagens."""
    fill = np.where(valid, img, np.nanmedian(img[valid])).astype(np.float32)
    gain = ndi.uniform_filter(fill, int(round(65 / res)))  # ganho ao longo do alcance (dente de serra entre nadires)
    I = fill / np.maximum(gain, 1.0)

    # Direção transversal às passagens (perpendicular às faixas de nadir): onde as sombras acústicas se projetam
    theta = np.radians(angle_deg)
    ux, uy = np.sin(theta), np.cos(theta)
    gx, gy = ndi.sobel(I, axis=1), ndi.sobel(I, axis=0)
    d_across = gx * ux + gy * uy
    d_along = -gx * uy + gy * ux

    f = {"I": I, "gain": gain, "d_across": d_across, "abs_d_across": np.abs(d_across), "abs_d_along": np.abs(d_along)}
    for k in (3, 7, 15):
        m = ndi.uniform_filter(I, k)
        f[f"std{k}"] = np.sqrt(np.maximum(ndi.uniform_filter(I * I, k) - m * m, 0))
        f[f"range{k}"] = ndi.maximum_filter(I, k) - ndi.minimum_filter(I, k)  # brilho vs sombra local
        f[f"mean{k}"] = m
        f[f"dacr_std{k}"] = np.sqrt(np.maximum(ndi.uniform_filter(d_across ** 2, k) - ndi.uniform_filter(d_across, k) ** 2, 0))
    f["shadow_frac7"] = ndi.uniform_filter((I < 0.7).astype(np.float32), 7)
    f["highlight_frac7"] = ndi.uniform_filter((I > 1.3).astype(np.float32), 7)
    return {k: v.astype(np.float32) for k, v in f.items()}


def cosmo_on_grid(transform, shape, crs, path: Path = COSMO) -> tuple[np.ndarray, np.ndarray]:
    """Relevo local máximo (m) e cota média do COSMO na grelha do sonar."""
    with rasterio.open(path) as src:
        dec = 3  # 0,9 m: suficiente para relevo de blocos >= 0,3 m e cabe em memória
        dem = src.read(1, out_shape=(src.height // dec, src.width // dec), resampling=Resampling.average).astype(np.float32)
        dem[dem <= -9000] = np.nan
        t = src.transform * src.transform.scale(src.width / dem.shape[1], src.height / dem.shape[0])
        ok = np.isfinite(dem)
        w = int(round(25 / abs(t.a)))
        num = ndi.uniform_filter(np.where(ok, dem, 0), w)
        den = ndi.uniform_filter(ok.astype(np.float32), w)
        low = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
        rel = np.where(ok & np.isfinite(low), dem - low, np.nan).astype(np.float32)
        rel_max = np.full(shape, np.nan, np.float32)
        z = np.full(shape, np.nan, np.float32)
        reproject(rel, rel_max, src_transform=t, src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.max, src_nodata=np.nan, dst_nodata=np.nan)
        reproject(dem, z, src_transform=t, src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return rel_max, z


def read_window(bounds):
    with rasterio.open(SONAR) as src:
        win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        img = src.read(1, window=win).astype(np.float32)
        return img, src.window_transform(win), src.crs, src.res[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rocha e relevo a partir do mosaico de sonar lateral do IH, calibrado pelo COSMO")
    parser.add_argument("--train-bbox", type=float, nargs=4, default=[1480, -293280, 7560, -288520],
                        help="Zona de calibração em EPSG:3763 (por omissão, a extensão do COSMO LTHQUGA1).")
    parser.add_argument("--target-bbox", type=float, nargs=4, required=True, help="Zona a reconstruir, EPSG:3763 (W S E N).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block-m", type=float, default=250.0)
    parser.add_argument("--max-samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-features", default="gain",
                        help="Feições a excluir, separadas por vírgulas (por omissão 'gain', que codifica a posição face ao nadir).")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- calibração
    img, T, crs, res = read_window(args.train_bbox)
    valid = img > 0
    ang = track_angle(img, valid)
    logger.info(f"Calibração: {img.shape}, rotação das passagens {ang}°")
    F = sonar_features(img, valid, res, ang)
    rel_max, z = cosmo_on_grid(T, img.shape, crs)
    rel18, z18 = cosmo_on_grid(T, img.shape, crs, COSMO_2018)
    edge = ndi.binary_erosion(valid, iterations=int(round(30 / res)))  # longe da borda do mosaico
    both = np.isfinite(rel_max) & np.isfinite(rel18) & np.isfinite(z) & np.isfinite(z18) & (z < -1.0)
    stable_rock = both & (rel_max >= ROCK_RELIEF_M) & (rel18 >= ROCK_RELIEF_M) & (np.abs(z - z18) <= STABLE_MAX_DZ_M)
    smooth = both & (rel_max < SMOOTH_MAX_RELIEF_M) & (rel18 < SMOOTH_MAX_RELIEF_M)
    use = edge & (stable_rock | smooth)  # rótulos limpos: rocha estável vs fundo liso nos dois anos
    idx = np.flatnonzero(use.ravel())
    if len(idx) > args.max_samples:
        idx = rng.choice(idx, args.max_samples, replace=False)
    drop = {f.strip() for f in args.drop_features.split(",") if f.strip()}
    names = [k for k in F if k not in drop]
    X = np.column_stack([F[k].ravel()[idx] for k in names])
    y_rel = rel_max.ravel()[idx]
    y_rock = stable_rock.ravel()[idx].astype(int)
    rows, cols = np.unravel_index(idx, img.shape)
    bpx = int(round(args.block_m / res))
    groups = (rows // bpx) * 10000 + cols // bpx
    logger.info(f"Amostras: {len(idx)} ({y_rock.mean():.1%} rocha), {len(np.unique(groups))} blocos de {args.block_m:.0f} m")

    oof_p = np.full(len(idx), np.nan); oof_r = np.full(len(idx), np.nan); oof_base = np.full(len(idx), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y_rock, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=args.seed)
        oof_p[te] = clf.fit(X[tr], y_rock[tr]).predict_proba(X[te])[:, 1]
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, loss="absolute_error", random_state=args.seed)
        oof_r[te] = reg.fit(X[tr], y_rel[tr]).predict(X[te])
        base = HistGradientBoostingClassifier(max_iter=100, random_state=args.seed)
        ib = [names.index("I")]
        oof_base[te] = base.fit(X[tr][:, ib], y_rock[tr]).predict_proba(X[te][:, ib])[:, 1]

    # métricas à resolução nativa (~2 m) e agregadas a ~10 m (resolução efetiva esperada do mosaico)
    def agg10(vals):
        key = (rows // 5) * 100000 + cols // 5
        order = np.argsort(key); k = key[order]; v = vals[order]
        uniq, start = np.unique(k, return_index=True)
        return np.maximum.reduceat(v, start)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sklearn.metrics import average_precision_score
    from src.analysis.reef_eval import auc_block_ci, gain_block_ci, recall_at_fpr
    xs = T.c + (cols + 0.5) * res
    ys = T.f - (rows + 0.5) * res
    from src.analysis.reef_eval import block_ids as _bid
    blk = _bid(xs, ys, 500.0)
    auc_r, lo_r, hi_r = auc_block_ci(y_rock, oof_p, blk, n_boot=500, seed=args.seed)
    g, glo, ghi = gain_block_ci(y_rock, oof_p, oof_base, blk, n_boot=500, seed=args.seed)
    rk = y_rock == 1
    metrics = {
        "features_used": names, "features_dropped": sorted(drop),
        "target": "rocha estável (relevo ≥0,3 m em 2018 e 2025, |Δz| ≤0,3 m) vs fundo liso (<0,15 m nos dois anos)",
        "rock_auc_blocks500_ci95": [round(auc_r, 3), round(lo_r, 3), round(hi_r, 3)],
        "rock_gain_vs_intensity_ci95": [round(g, 3), round(glo, 3), round(ghi, 3)],
        "rock_pr_auc": float(average_precision_score(y_rock, oof_p)),
        "rock_pr_auc_intensity_only": float(average_precision_score(y_rock, oof_base)),
        "rock_recall_fpr5": recall_at_fpr(y_rock, oof_p, 0.05),
        "relief_on_rock_mae_m": float(mean_absolute_error(y_rel[rk], oof_r[rk])) if rk.any() else None,
        "relief_on_rock_mae_m_median_baseline": float(mean_absolute_error(y_rel[rk], np.full(rk.sum(), np.median(y_rel[rk])))) if rk.any() else None,
        "relief_on_rock_spearman": float(spearmanr(y_rel[rk], oof_r[rk]).correlation) if rk.sum() > 10 else None,
        "calibration_area_epsg3763": args.train_bbox,
        "track_rotation_deg": float(ang),
        "n_samples": int(len(idx)), "rock_fraction": float(y_rock.mean()), "n_blocks": int(len(np.unique(groups))),
        "rock_auc_spatial_cv": float(roc_auc_score(y_rock, oof_p)),
        "rock_auc_intensity_only": float(roc_auc_score(y_rock, oof_base)),
        "relief_mae_m": float(mean_absolute_error(y_rel, oof_r)),
        "relief_mae_m_predict_median_baseline": float(mean_absolute_error(y_rel, np.full_like(y_rel, np.median(y_rel)))),
        "relief_spearman": float(spearmanr(y_rel, oof_r).correlation),
        "relief_spearman_10m_blocks": float(spearmanr(agg10(y_rel), agg10(oof_r)).correlation),
        "rock_auc_10m_blocks": float(roc_auc_score((agg10(y_rel) >= ROCK_RELIEF_M).astype(int), agg10(oof_p))),
    }
    logger.info(json.dumps(metrics, indent=2))

    # ---- modelo final e aplicação à zona alvo
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=args.seed).fit(X, y_rock)
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, loss="absolute_error", random_state=args.seed).fit(X, y_rel)

    timg, TT, _, _ = read_window(args.target_bbox)
    tvalid = timg > 0
    tang = track_angle(timg, tvalid)
    TF = sonar_features(timg, tvalid, res, tang)
    TX = np.column_stack([TF[k].ravel() for k in names])
    p = clf.predict_proba(TX)[:, 1].reshape(timg.shape).astype(np.float32)
    r = reg.predict(TX).reshape(timg.shape).astype(np.float32)
    p[~tvalid] = np.nan; r[~tvalid] = np.nan

    emod = np.full(timg.shape, np.nan, np.float32)
    with rasterio.open(EMODNET) as e:
        src = e.read(1).astype(np.float32)
        reproject(src, emod, src_transform=e.transform, src_crs=e.crs, dst_transform=TT, dst_crs=crs, resampling=Resampling.bilinear)
    pseudo_dem = np.where(tvalid, emod + np.maximum(r, 0), np.nan).astype(np.float32)  # cota (negativa) + relevo

    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "crs": crs, "transform": TT, "width": timg.shape[1],
               "height": timg.shape[0], "nodata": np.nan, "compress": "deflate"}
    tags = {"source": "Mosaico sonar lateral IH (AIA 3587 Desenho 2) calibrado com COSMO LTHQUGA1 2025",
            "status": "ESTIMATIVA — não é medição; ver metrics.json", "target_track_rotation_deg": str(tang)}
    products = [("rock_probability", p, "probabilidade de rocha saliente (>=0,3 m)")]
    # O relevo só é gravado se a validação mostrar que o mosaico o prevê (no Desenho 2 do IH não prevê: ver metrics.json)
    relief_ok = metrics["relief_spearman_10m_blocks"] >= RELIEF_MIN_SPEARMAN
    metrics["relief_products_written"] = bool(relief_ok)
    if relief_ok:
        products += [("relief_estimated_m", r, "relevo local estimado (m acima do fundo envolvente)"),
                     ("pseudo_dem_zh_m", pseudo_dem, "EMODnet + relevo estimado; cotas negativas ZH (convenção COSMO)")]
    else:
        logger.warning(f"Relevo não validado (Spearman 10 m = {metrics['relief_spearman_10m_blocks']:.2f} < {RELIEF_MIN_SPEARMAN}): "
                       "só a probabilidade de rocha é gravada.")
    for name, arr, desc in products:
        with rasterio.open(args.out / f"{name}.tif", "w", **profile) as dst:
            dst.write(arr, 1); dst.set_band_description(1, desc); dst.update_tags(**tags)
    metrics["target_bbox_epsg3763"] = args.target_bbox
    metrics["target_track_rotation_deg"] = float(tang)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"GeoTIFFs e métricas em {args.out}")


if __name__ == "__main__":
    main()
