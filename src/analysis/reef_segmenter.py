"""
src/analysis/reef_segmenter.py — Segmentação Semântica e Detetor de Recifes Marinhos.

Integra:
1. Máscara de água NDWI (Sentinel-2) + Filtro de profundidade batimétrica (2 m a 25 m EMODnet).
2. Detetor supervisionado (ReefClassifier) treinado no Ground Truth multifeixe e batimetria física.
3. Modo de limiares físicos absolutos calibrados (sem normalização percentílica distorcida por cena).
4. Validação estrita exata (Exact-Pixel ROC-AUC) contra pontos aleatórios do fundo marinho (>150 m de recifes).
5. Extração vetorial de polígonos/aglomerados de candidatos a recife natural com área mínima.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import geometry_mask, shapes
from scipy import stats
from shapely.geometry import shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.analysis.features import extract_morpho_spectral_features, resample_emodnet_screen
from src.ingestion.sentinel_reader import build_clean_water_mask, read_sentinel_bands

logger = logging.getLogger("reef_pipeline.analysis.segmenter")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL_FEATURES = [
    "sdb_vrm50",
    "sdb_bpi100",
    "sdb_slope",
    "B03_std30",
    "b3_anom",
    "gr_ratio",
    "sam_rock_score",
    "turbidity",
    "B02",
    "B03",
    "B04",
    "sdb_m",
    "sdb_anomaly_m",
]


class ReefClassifier:
    """Classificador supervisionado de recifes rochosos vs fundo arenoso."""

    def __init__(self, model: Any | None = None, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or DEFAULT_MODEL_FEATURES
        if model is not None:
            self.model = model
            # Verifica se já está treinado (possui classes_)
            self.is_fitted = hasattr(model, "classes_") or hasattr(model, "estimators_")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced",
            )
            self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> ReefClassifier:
        logger.info(
            f"Treinando ReefClassifier ({self.model.__class__.__name__}) em {len(y)} amostras "
            f"({int(np.sum(y == 1))} GT recifes, {int(np.sum(y == 0))} fundo arenoso)..."
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ReefClassifier não está treinado.")
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_map(
        self,
        features: dict[str, np.ndarray],
        analysis_mask: np.ndarray,
        batch_size: int = 50000,
    ) -> np.ndarray:
        """Gera um mapa contínuo raster de probabilidade [0.0, 1.0] sobre a área de análise."""
        logger.info("Mapeando probabilidades de recife com modelo supervisionado...")
        first_feat = next(iter(features.values()))
        h, w = first_feat.shape
        reef_proba = np.zeros((h, w), dtype=np.float32)

        valid_idx = np.nonzero(analysis_mask)
        n_valid = len(valid_idx[0])
        if n_valid == 0:
            return reef_proba

        # Extrai matriz de features para os pixels válidos
        X_valid = np.column_stack(
            [
                np.nan_to_num(features.get(k, np.zeros((h, w), dtype=np.float32))[valid_idx], nan=0.0)
                for k in self.feature_names
            ]
        )

        probs = np.zeros(n_valid, dtype=np.float32)
        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            probs[start:end] = self.model.predict_proba(X_valid[start:end])[:, 1].astype(np.float32)

        reef_proba[valid_idx] = probs
        return reef_proba


def _new_random_forest(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=seed, class_weight="balanced"
    )


def collect_zone_samples(
    zone_configs: dict[str, dict[str, Any]],
    feature_names: list[str] | None = None,
    exclusion_px: int = 15,
    sample_bg_ratio: int = 20,
    seed: int = 42,
    depth_matched: bool = True,
    depth_tol_m: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Recolhe, por zona, as amostras GT (y=1) e de fundo (y=0, a mais de ``exclusion_px`` pixels de
    qualquer cume) com as respetivas features e a profundidade EMODnet de cada amostra.

    Com ``depth_matched=True`` (padrão), cada cume recebe ``sample_bg_ratio`` pontos de fundo à
    mesma profundidade EMODnet (± ``depth_tol_m``). Sem isto, o fundo aleatório está quase sempre a
    outra profundidade e o modelo aprende "raso vs fundo" (o sinal das bandas inverte-se entre zonas
    com cumes costeiros e cumes ao largo) em vez de "rocha vs areia".

    Devolve ``{zona: {"X": (n, f), "y": (n,), "depth": (n,), "group": (n,)}}`` (``group`` = índice do
    cume GT mais próximo, unidade de reamostragem do bootstrap) para permitir treino e
    validação leave-one-zone-out sem voltar a ler as cenas.
    """
    features_to_use = feature_names or DEFAULT_MODEL_FEATURES
    samples: dict[str, dict[str, np.ndarray]] = {}

    logger.info("Coletando amostras de treino multizona (GT vs Fundo arenoso)...")

    for z_key, cfg in zone_configs.items():
        X_list, y_list, depth_list, group_list = [], [], [], []
        scene = read_sentinel_bands(cfg["scene_path"])
        water_mask = build_clean_water_mask(scene, ndwi_threshold=0.0, apply_glint_filter=True)
        depth_m, depth_mask = resample_emodnet_screen(
            cfg["emodnet_path"], scene.shape, scene.crs, scene.transform, min_depth=2.0, max_depth=25.0
        )
        analysis_mask = water_mask & depth_mask
        features = extract_morpho_spectral_features(
            scene.blue, scene.green, scene.red, scene.nir, water_mask=water_mask, depth_emodnet=depth_m
        )

        df_gt = pd.read_csv(cfg["ground_truth_path"])
        gdf_gt = gpd.GeoDataFrame(
            df_gt, geometry=gpd.points_from_xy(df_gt.lon, df_gt.lat), crs="EPSG:4326"
        ).to_crs(scene.crs)

        # Sítios de mergulho não entram na verdade (ledger E027); só os pontos/píxeis do ground_truth_path.
        gt_pxs: list[tuple[int, int]] = []
        gt_cume: list[int] = []  # cume de cada píxel GT: unidade de reamostragem do bootstrap
        with rasterio.open(cfg["scene_path"]) as src:
            for i, (_, row) in enumerate(gdf_gt.iterrows()):
                px, py = ~src.transform * (row.geometry.x, row.geometry.y)
                px, py = int(round(px)), int(round(py))
                if 0 <= py < src.height and 0 <= px < src.width and analysis_mask[py, px] and (py, px) not in gt_pxs:
                    gt_pxs.append((py, px))
                    gt_cume.append(int(row["cume_id"]) if "cume_id" in gdf_gt.columns else i)

        bg_candidates = analysis_mask.copy()
        for py, px in gt_pxs:
            bg_candidates[max(0, py - exclusion_px) : py + exclusion_px + 1, max(0, px - exclusion_px) : px + exclusion_px + 1] = False

        bg_r, bg_c = np.nonzero(bg_candidates)
        rng = np.random.default_rng(seed)
        if depth_matched:
            bg_pxs = _depth_matched_background(
                gt_pxs, bg_r, bg_c, depth_m, sample_bg_ratio, depth_tol_m, rng
            )
        else:
            n_bg = min(len(bg_r), max(100, len(gt_pxs) * sample_bg_ratio))
            pick = rng.choice(len(bg_r), size=n_bg, replace=False)
            bg_pxs = list(zip(bg_r[pick], bg_c[pick], strict=True))

        # Cada amostra pertence ao cume do píxel GT mais próximo: o bootstrap reamostra cumes, não pixels
        gt_arr = np.asarray(gt_pxs, dtype=np.float64).reshape(-1, 2)
        for label, pxs in ((1, gt_pxs), (0, bg_pxs)):
            for r, c in pxs:
                feats = [float(features[k][r, c]) for k in features_to_use]
                if np.all(np.isfinite(feats)):
                    X_list.append(feats)
                    y_list.append(label)
                    depth_list.append(float(depth_m[r, c]))
                    group_list.append(gt_cume[int(np.argmin(np.hypot(gt_arr[:, 0] - r, gt_arr[:, 1] - c)))] if len(gt_arr) else -1)

        samples[z_key] = {
            "X": np.array(X_list, dtype=np.float32).reshape(-1, len(features_to_use)),
            "y": np.array(y_list, dtype=int),
            "depth": np.array(depth_list, dtype=np.float32),
            "group": np.array(group_list, dtype=int),
        }

    return samples


def _depth_matched_background(
    gt_pxs: list[tuple[int, int]],
    bg_r: np.ndarray,
    bg_c: np.ndarray,
    depth_m: np.ndarray,
    per_gt: int,
    tol_m: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Para cada cume, sorteia ``per_gt`` pixels de fundo com profundidade a ± ``tol_m`` da do cume."""
    bg_depth = depth_m[bg_r, bg_c]
    chosen: set[tuple[int, int]] = set()
    for r, c in gt_pxs:
        d = depth_m[r, c]
        if not np.isfinite(d):
            continue
        idx = np.flatnonzero(np.abs(bg_depth - d) <= tol_m)
        if len(idx) == 0:
            continue
        for i in rng.choice(idx, size=min(per_gt, len(idx)), replace=False):
            chosen.add((int(bg_r[i]), int(bg_c[i])))
    return sorted(chosen)


def _stack_samples(samples: dict[str, dict[str, np.ndarray]], zones: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.concatenate([samples[z]["X"] for z in zones])
    y = np.concatenate([samples[z]["y"] for z in zones])
    depth = np.concatenate([samples[z]["depth"] for z in zones])
    return X, y, depth


def leave_one_zone_out(
    samples: dict[str, dict[str, np.ndarray]],
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, dict[str, Any]]:
    """
    Validação honesta por zona: cada zona é avaliada por um modelo que nunca viu os seus pontos.

    Para cada zona devolve ``auc`` (modelo completo) e ``auc_depth_only`` (o mesmo Random Forest
    treinado só com a profundidade EMODnet). Se ``auc`` não for claramente superior a
    ``auc_depth_only``, o detetor está a reconhecer a profundidade típica dos cumes, não o recife.
    ``gain_ci95`` é o IC 95% bootstrap de ``auc − auc_depth_only``: com poucos cumes é largo, e um
    limite inferior <= 0 significa que o ganho não se distingue do ruído.
    """
    results: dict[str, dict[str, Any]] = {}
    for held_out in samples:
        train_zones = [z for z in samples if z != held_out]
        X_te, y_te, d_te = samples[held_out]["X"], samples[held_out]["y"], samples[held_out]["depth"]
        n_gt = int(np.sum(y_te == 1))
        if not train_zones or n_gt == 0 or n_gt == len(y_te):
            results[held_out] = {
                "auc": None,
                "auc_depth_only": None,
                "gain": None,
                "gain_ci95": None,
                "n_gt": n_gt,
                "n_bg": int(np.sum(y_te == 0)),
            }
            continue

        X_tr, y_tr, d_tr = _stack_samples(samples, train_zones)
        p_full = _new_random_forest(seed).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        p_depth = _new_random_forest(seed).fit(d_tr.reshape(-1, 1), y_tr).predict_proba(d_te.reshape(-1, 1))[:, 1]

        auc = float(roc_auc_score(y_te, p_full))
        auc_depth = float(roc_auc_score(y_te, p_depth))
        gain_lo, gain_hi = _bootstrap_gain_ci(
            y_te, p_full, p_depth, n_boot=n_boot, seed=seed, groups=samples[held_out].get("group")
        )

        results[held_out] = {
            "auc": auc,
            "auc_depth_only": auc_depth,
            "gain": auc - auc_depth,
            "gain_ci95": (gain_lo, gain_hi),
            "n_gt": n_gt,
            "n_bg": int(np.sum(y_te == 0)),
        }
    return results


def _bootstrap_gain_ci(
    y: np.ndarray,
    p_model: np.ndarray,
    p_baseline: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    IC 95% (bootstrap) da diferença de AUC modelo − linha de base nos mesmos pontos.

    Com ``groups`` (cume de origem de cada amostra) reamostra cumes inteiros: os pixels de um cume e
    o seu fundo são fortemente autocorrelacionados, e reamostrar pixels dá um IC demasiado estreito.
    Sem ``groups`` reamostra pixels (estratificado). Devolve ``(nan, nan)`` se não houver réplicas válidas.
    """
    rng = np.random.default_rng(seed)
    gains: list[float] = []
    if groups is None:
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            gains.append(roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_baseline[idx]))
    else:
        members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
        keys = list(members)
        for _ in range(n_boot):
            idx = np.concatenate([members[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            if len(np.unique(y[idx])) < 2:
                continue
            gains.append(roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_baseline[idx]))
    if len(gains) < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(gains, (2.5, 97.5))
    return float(lo), float(hi)


def train_multi_zone_classifier(
    zone_configs: dict[str, dict[str, Any]],
    feature_names: list[str] | None = None,
    exclusion_px: int = 15,
    sample_bg_ratio: int = 20,
    seed: int = 42,
    exclude_zones: list[str] | None = None,
    samples: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[ReefClassifier, dict[str, Any]]:
    """
    Treina o modelo supervisionado nas zonas do Algarve com Ground Truth disponível.

    ``exclude_zones`` deixa zonas fora do treino — use-o para avaliar uma zona com um modelo
    que nunca viu os seus pontos (as métricas dessa zona com o modelo completo são in-sample).
    ``samples`` (de :func:`collect_zone_samples`) evita voltar a ler as cenas.
    """
    features_to_use = feature_names or DEFAULT_MODEL_FEATURES
    if samples is None:
        samples = collect_zone_samples(zone_configs, features_to_use, exclusion_px, sample_bg_ratio, seed)

    train_zones = [z for z in samples if z not in (exclude_zones or [])]
    X, y, _ = _stack_samples(samples, train_zones)
    groups = np.concatenate([np.full(len(samples[z]["y"]), z) for z in train_zones])

    # CV agrupada por zona (leave-one-zone-out): folds aleatórios misturam cumes vizinhos
    # entre treino e teste e inflacionam a AUC por autocorrelação espacial
    cv_oof = np.full(len(y), np.nan, dtype=np.float32)
    for z in train_zones:
        te = groups == z
        if te.all() or not np.any(y[te] == 1):
            continue
        m_cv = _new_random_forest(seed).fit(X[~te], y[~te])
        cv_oof[te] = m_cv.predict_proba(X[te])[:, 1]

    scored = np.isfinite(cv_oof)
    cv_auc = float(roc_auc_score(y[scored], cv_oof[scored])) if scored.any() and len(set(y[scored])) == 2 else None
    logger.info(f"Validação Cruzada Leave-One-Zone-Out ROC-AUC ({len(train_zones)} zonas): {cv_auc}")

    # Treina o classificador final sobre as zonas de treino
    clf = ReefClassifier(feature_names=features_to_use)
    clf.fit(X, y)

    metrics = {
        "n_samples": len(y),
        "n_gt": int(np.sum(y == 1)),
        "n_bg": int(np.sum(y == 0)),
        "train_zones": train_zones,
        "cv_auc_oof": cv_auc,
        "feature_importances": dict(zip(features_to_use, clf.model.feature_importances_, strict=True)),
    }
    return clf, metrics


def compute_reef_probability_absolute(
    features: dict[str, np.ndarray],
    water_mask: np.ndarray,
    depth_mask: np.ndarray,
) -> np.ndarray:
    """
    Calcula a probabilidade de recife com limiares físicos calibrados e absolutos,
    eliminando a normalização percentílica por cena que gera falsos positivos em fundos homogéneos.
    """
    logger.info("Calculando probabilidade de recife com limiares físicos absolutos...")
    analysis_mask = water_mask & depth_mask

    vrm = features.get("sdb_vrm50", features.get("psdb_vrm50", np.zeros_like(water_mask, dtype=np.float32)))
    bpi = features.get("sdb_bpi100", features.get("psdb_bpi100", np.zeros_like(water_mask, dtype=np.float32)))
    slope = features.get("sdb_slope", features.get("psdb_slope", np.zeros_like(water_mask, dtype=np.float32)))
    tex = features.get("B03_std30", np.zeros_like(water_mask, dtype=np.float32))

    # Funções de transferência físicas calibradas (sem distorção percentílica de cena)
    vrm_score = np.clip((vrm - 0.0020) / 0.0050, 0.0, 1.0)
    bpi_score = np.clip(bpi / 1.5, 0.0, 1.0)
    slope_score = np.clip((slope - 0.05) / 0.12, 0.0, 1.0)
    tex_score = np.clip((tex - 0.0008) / 0.0018, 0.0, 1.0)

    score = (
        0.35 * vrm_score +
        0.25 * bpi_score +
        0.20 * slope_score +
        0.20 * tex_score
    )

    reef_proba = np.where(analysis_mask, score, 0.0).astype(np.float32)
    return reef_proba


PHYSICAL_FEATURE = "sdb_anomaly_m"


def compute_reef_probability_physical(
    features: dict[str, np.ndarray],
    water_mask: np.ndarray,
    depth_mask: np.ndarray,
    flagged_fraction: float = 0.05,
) -> np.ndarray:
    """
    Regra física sem treino: rocha com macroalgas é mais escura que areia à mesma profundidade,
    pelo que a anomalia ``sdb_m − EMODnet`` (metros, > 0 = fundo mais escuro) é positiva nos recifes.

    A anomalia passa por uma logística centrada no percentil ``1 − flagged_fraction`` da própria
    zona (p = 0.5 nos ~5% de água mais escura para a sua profundidade). Não usa Ground Truth, pelo
    que a sua avaliação contra o GT não tem fuga de dados.
    """
    analysis_mask = water_mask & depth_mask
    anom = features.get(PHYSICAL_FEATURE)
    if anom is None:
        raise KeyError(f"Feature '{PHYSICAL_FEATURE}' em falta: extraia as features com depth_emodnet.")

    valid = analysis_mask & np.isfinite(anom)
    if not valid.any():
        return np.zeros(anom.shape, dtype=np.float32)

    p50, p_hi = np.percentile(anom[valid], (50, 100 * (1 - flagged_fraction)))
    scale = max((p_hi - p50) / 3.0, 1e-3)
    with np.errstate(over="ignore", invalid="ignore"):
        proba = 1.0 / (1.0 + np.exp(-(anom - p_hi) / scale))
    return np.where(valid, proba, 0.0).astype(np.float32)


def evaluate_physical_rule(
    zone_samples: dict[str, np.ndarray],
    feature_names: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Avalia a regra física numa zona sobre amostras com fundo à mesma profundidade (de
    :func:`collect_zone_samples`), no mesmo formato de :func:`leave_one_zone_out`.

    ``auc`` é a da anomalia de escurecimento (sem treino); ``auc_depth_only`` a da profundidade
    EMODnet nas mesmas amostras (~0.5 por construção). ``gain_ci95`` é o IC 95% bootstrap da diferença.
    """
    i = (feature_names or DEFAULT_MODEL_FEATURES).index(PHYSICAL_FEATURE)
    y, x, d = zone_samples["y"], zone_samples["X"][:, i].astype(np.float64), zone_samples["depth"].astype(np.float64)
    n_gt = int(np.sum(y == 1))
    if n_gt == 0 or n_gt == len(y):
        return {"auc": None, "auc_depth_only": None, "gain": None, "gain_ci95": None, "n_gt": n_gt, "n_bg": int(np.sum(y == 0))}

    auc = float(roc_auc_score(y, x))
    auc_depth = float(roc_auc_score(y, d))
    return {
        "auc": auc,
        "auc_depth_only": auc_depth,
        "gain": auc - auc_depth,
        "gain_ci95": _bootstrap_gain_ci(y, x, d, n_boot=n_boot, seed=seed, groups=zone_samples.get("group")),
        "n_gt": n_gt,
        "n_bg": int(np.sum(y == 0)),
    }


def compute_reef_probability_map(
    features: dict[str, np.ndarray],
    water_mask: np.ndarray,
    depth_mask: np.ndarray,
    model: ReefClassifier | None = None,
    mode: str = "supervised",
) -> np.ndarray:
    """
    Gera o mapa de probabilidade de recife rochoso natural:
    - mode="supervised": Utiliza o modelo preditivo calibrado com o Ground Truth multifeixe.
    - mode="absolute": Utiliza limiares físicos absolutos independentes da cena.
    - mode="physical": Anomalia de escurecimento à mesma profundidade (sem treino).
    """
    analysis_mask = water_mask & depth_mask

    if mode == "physical":
        return compute_reef_probability_physical(features, water_mask, depth_mask)

    if mode == "supervised" and model is not None and model.is_fitted:
        return model.predict_proba_map(features, analysis_mask)

    return compute_reef_probability_absolute(features, water_mask, depth_mask)


def _local_max_3x3(proba: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = proba.shape
    return np.array(
        [
            float(np.nanmax(proba[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]))
            for r, c in zip(rows, cols, strict=True)
        ],
        dtype=np.float64,
    )


def evaluate_against_background(
    reef_proba: np.ndarray,
    analysis_mask: np.ndarray,
    gt_rows: np.ndarray,
    gt_cols: np.ndarray,
    n_background: int = 2000,
    exclusion_px: int = 5,
    seed: int = 0,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Validação estatística rigorosa:
    - Avalia valores de probabilidade EXATOS no pixel do Ground Truth (sem inflação 3x3 max).
    - Amostra pontos aleatórios de fundo marinho dentro da mesma cena afastados do GT (exclusion_px).
    - Calcula o ROC-AUC exato, teste de hipótese Mann-Whitney U, taxas de deteção e separação de contraste.
    """
    gt_rows = np.asarray(gt_rows, dtype=int)
    gt_cols = np.asarray(gt_cols, dtype=int)

    valid = np.isfinite(reef_proba)
    proba_clean = np.where(valid, reef_proba, 0.0)

    # Identifica pixels de fundo arenoso com exclusão estrita em redor do GT
    candidates = analysis_mask & valid
    for r, c in zip(gt_rows, gt_cols, strict=True):
        candidates[max(0, r - exclusion_px) : r + exclusion_px + 1, max(0, c - exclusion_px) : c + exclusion_px + 1] = False
    bg_r, bg_c = np.nonzero(candidates)

    if len(gt_rows) == 0 or len(bg_r) == 0:
        logger.warning("Validação contra o fundo impossível: sem pontos GT ou sem pixels de fundo.")
        return {
            "auc": None,
            "auc_exact": None,
            "auc_3x3_max": None,
            "gt_detection_rate": None,
            "bg_detection_rate": None,
            "p_value": None,
            "gt_median": None,
            "bg_median": None,
            "delta_contrast": None,
            "fpr5_thresh": None,
            "recall_fpr5": None,
            "recall_fpr10": None,
            "n_gt": len(gt_rows),
            "n_background": 0,
        }

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(bg_r), size=min(n_background, len(bg_r)), replace=False)

    # 1. Avaliação EXATA no pixel (métrica científica primária)
    gt_scores_exact = np.array([proba_clean[r, c] for r, c in zip(gt_rows, gt_cols, strict=True)], dtype=np.float64)
    bg_scores_exact = np.array([proba_clean[r, c] for r, c in zip(bg_r[pick], bg_c[pick], strict=True)], dtype=np.float64)

    labels = np.concatenate([np.ones(len(gt_scores_exact)), np.zeros(len(bg_scores_exact))])
    all_scores = np.concatenate([gt_scores_exact, bg_scores_exact])
    auc_exact = float(roc_auc_score(labels, all_scores))

    # Teste de Mann-Whitney U (H1: GT > BG)
    _, p_val = stats.mannwhitneyu(gt_scores_exact, bg_scores_exact, alternative="greater")

    gt_med = float(np.median(gt_scores_exact))
    bg_med = float(np.median(bg_scores_exact))
    delta = gt_med - bg_med

    # Taxas de deteção no limiar fixo especificado
    gt_det_rate = float(np.mean(gt_scores_exact >= threshold))
    bg_det_rate = float(np.mean(bg_scores_exact >= threshold))

    # Taxa de deteção em pontos de operação da ROC (FPR 5% e FPR 10%)
    fpr5_thresh = float(np.percentile(bg_scores_exact, 95))
    fpr10_thresh = float(np.percentile(bg_scores_exact, 90))
    rec_fpr5 = float(np.mean(gt_scores_exact >= fpr5_thresh))
    rec_fpr10 = float(np.mean(gt_scores_exact >= fpr10_thresh))

    # 2. Avaliação auxiliar 3x3 max (para registo histórico comparativo)
    gt_scores_3x3 = _local_max_3x3(proba_clean, gt_rows, gt_cols)
    bg_scores_3x3 = _local_max_3x3(proba_clean, bg_r[pick], bg_c[pick])
    all_scores_3x3 = np.concatenate([gt_scores_3x3, bg_scores_3x3])
    auc_3x3_max = float(roc_auc_score(labels, all_scores_3x3))

    return {
        "auc": auc_exact,
        "auc_exact": auc_exact,
        "auc_3x3_max": auc_3x3_max,
        "gt_detection_rate": gt_det_rate,
        "bg_detection_rate": bg_det_rate,
        "p_value": float(p_val),
        "gt_median": gt_med,
        "bg_median": bg_med,
        "delta_contrast": delta,
        "fpr5_thresh": fpr5_thresh,
        "recall_fpr5": rec_fpr5,
        "recall_fpr10": rec_fpr10,
        "n_gt": len(gt_scores_exact),
        "n_background": len(bg_scores_exact),
        "gt_scores_exact": gt_scores_exact,
        "bg_scores_exact": bg_scores_exact,
    }


def extract_candidate_reefs(
    reef_proba: np.ndarray,
    profile: dict[str, Any],
    threshold: float = 0.50,
    min_area_m2: float = 500.0,
) -> gpd.GeoDataFrame:
    """Extrai aglomerados e polígonos contíguos de recifes prováveis com prob_max e prob_mean exatos."""
    logger.info(f"Vetorizando candidatos com probabilidade >= {threshold}...")
    binary_mask = (reef_proba >= threshold).astype(np.uint8)

    transform = profile["transform"]
    crs = profile["crs"]

    records = []
    for geom_dict, val in shapes(binary_mask, mask=(binary_mask == 1), transform=transform):
        geom = shape(geom_dict)
        area_m2 = geom.area
        if area_m2 >= min_area_m2:
            # Máscara só na janela do polígono: O(área do polígono), não O(cena) por candidato
            minx, miny, maxx, maxy = geom.bounds
            (c0, r0), (c1, r1) = ~transform @ (minx, maxy), ~transform @ (maxx, miny)
            r_lo, r_hi = max(0, int(np.floor(min(r0, r1)))), min(reef_proba.shape[0], int(np.ceil(max(r0, r1))) + 1)
            c_lo, c_hi = max(0, int(np.floor(min(c0, c1)))), min(reef_proba.shape[1], int(np.ceil(max(c0, c1))) + 1)
            win_proba = reef_proba[r_lo:r_hi, c_lo:c_hi]
            poly_px_mask = geometry_mask(
                [geom_dict],
                out_shape=win_proba.shape,
                transform=transform @ Affine.translation(c_lo, r_lo),
                invert=True,
            )
            valid_px = poly_px_mask & (binary_mask[r_lo:r_hi, c_lo:c_hi] == 1) & np.isfinite(win_proba)
            if np.any(valid_px):
                px_vals = win_proba[valid_px]
                prob_max = round(float(np.max(px_vals)), 4)
                prob_mean = round(float(np.mean(px_vals)), 4)
            else:
                prob_max = 0.0
                prob_mean = 0.0
            records.append({
                "geometry": geom,
                "area_m2": round(area_m2, 1),
                "prob_max": prob_max,
                "prob_mean": prob_mean,
            })

    if not records:
        logger.warning("Nenhum candidato a recife encontrado com o limiar especificado.")
        return gpd.GeoDataFrame(
            columns=["geometry", "area_m2", "prob_max", "prob_mean"],
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(f"Detetados {len(gdf)} aglomerados de recife natural (área mínima: {min_area_m2} m²).")
    return gdf
