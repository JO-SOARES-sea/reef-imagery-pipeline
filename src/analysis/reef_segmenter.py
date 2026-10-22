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
from rasterio.features import shapes
from scipy import stats
from shapely.geometry import shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.analysis.features import extract_morpho_spectral_features, resample_emodnet_screen
from src.ingestion.sentinel_reader import build_clean_water_mask, read_sentinel_bands, save_geotiff

logger = logging.getLogger("reef_pipeline.analysis.segmenter")

DEFAULT_MODEL_FEATURES = [
    "sdb_vrm50",
    "sdb_bpi100",
    "sdb_slope",
    "B03_std30",
    "b3_anom",
    "gr_ratio",
    "B02",
    "B03",
    "B04",
    "sdb_m",
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


def train_multi_zone_classifier(
    zone_configs: dict[str, dict[str, Any]],
    feature_names: list[str] | None = None,
    exclusion_px: int = 15,
    sample_bg_ratio: int = 20,
    seed: int = 42,
) -> tuple[ReefClassifier, dict[str, Any]]:
    """
    Treina o modelo supervisionado em todas as zonas do Algarve com Ground Truth disponível.
    Amostra pontos de fundo arenoso a mais de 150 m (exclusion_px=15) de qualquer cume.
    """
    features_to_use = feature_names or DEFAULT_MODEL_FEATURES
    X_list, y_list, group_list = [], [], []

    logger.info("Coletando amostras de treino multizona (GT vs Fundo arenoso)...")

    for z_key, cfg in zone_configs.items():
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

        gt_pxs = []
        with rasterio.open(cfg["scene_path"]) as src:
            for idx, row in gdf_gt.iterrows():
                px, py = ~src.transform * (row.geometry.x, row.geometry.y)
                px, py = int(round(px)), int(round(py))
                if 0 <= py < src.height and 0 <= px < src.width and analysis_mask[py, px]:
                    gt_pxs.append((py, px))

        bg_candidates = analysis_mask.copy()
        for py, px in gt_pxs:
            bg_candidates[max(0, py - exclusion_px) : py + exclusion_px + 1, max(0, px - exclusion_px) : px + exclusion_px + 1] = False

        bg_r, bg_c = np.nonzero(bg_candidates)
        rng = np.random.default_rng(seed)
        n_bg = min(len(bg_r), max(100, len(gt_pxs) * sample_bg_ratio))
        pick = rng.choice(len(bg_r), size=n_bg, replace=False)
        bg_pxs = list(zip(bg_r[pick], bg_c[pick], strict=True))

        for r, c in gt_pxs:
            feats = [float(features[k][r, c]) for k in features_to_use]
            if np.all(np.isfinite(feats)):
                X_list.append(feats)
                y_list.append(1)
                group_list.append(z_key)

        for r, c in bg_pxs:
            feats = [float(features[k][r, c]) for k in features_to_use]
            if np.all(np.isfinite(feats)):
                X_list.append(feats)
                y_list.append(0)
                group_list.append(z_key)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    groups = np.array(group_list)

    # Validação cruzada estratificada (5-Fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in skf.split(X, y):
        m_cv = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=seed, class_weight="balanced"
        )
        m_cv.fit(X[tr], y[tr])
        cv_oof[te] = m_cv.predict_proba(X[te])[:, 1]

    cv_auc = float(roc_auc_score(y, cv_oof))
    logger.info(f"Validação Cruzada 5-Fold Stratified CV ROC-AUC: {cv_auc:.3f}")

    # Treina o classificador final sobre o dataset completo
    clf = ReefClassifier(feature_names=features_to_use)
    clf.fit(X, y)

    metrics = {
        "n_samples": len(y),
        "n_gt": int(np.sum(y == 1)),
        "n_bg": int(np.sum(y == 0)),
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
    """
    analysis_mask = water_mask & depth_mask

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
    exclusion_px: int = 15,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Validação estatística rigorosa:
    - Avalia valores de probabilidade EXATOS no pixel do Ground Truth (sem inflação 3x3 max).
    - Amostra 1000 a 2000 pontos aleatórios de fundo marinho dentro da mesma cena a mais de 150 m (exclusion_px=15) de qualquer cume.
    - Calcula o ROC-AUC exato, teste de hipótese Mann-Whitney U e separação de contraste.
    """
    gt_rows = np.asarray(gt_rows, dtype=int)
    gt_cols = np.asarray(gt_cols, dtype=int)

    valid = np.isfinite(reef_proba)
    proba_clean = np.where(valid, reef_proba, 0.0)

    # Identifica pixels de fundo arenoso com exclusão estrita de 150m em redor do GT
    candidates = analysis_mask & valid
    for r, c in zip(gt_rows, gt_cols, strict=True):
        candidates[max(0, r - exclusion_px) : r + exclusion_px + 1, max(0, c - exclusion_px) : c + exclusion_px + 1] = False
    bg_r, bg_c = np.nonzero(candidates)

    if len(gt_rows) == 0 or len(bg_r) == 0:
        logger.warning("Validação contra o fundo impossível: sem pontos GT ou sem pixels de fundo.")
        return {
            "auc_exact": None,
            "auc_3x3_max": None,
            "p_value": None,
            "gt_median": None,
            "bg_median": None,
            "delta_contrast": None,
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
        "auc_exact": auc_exact,
        "auc_3x3_max": auc_3x3_max,
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
    """Extrai aglomerados e polígonos contíguos de recifes prováveis."""
    logger.info(f"Vetorizando candidatos com probabilidade >= {threshold}...")
    binary_mask = (reef_proba >= threshold).astype(np.uint8)

    transform = profile["transform"]
    crs = profile["crs"]

    records = []
    for geom_dict, val in shapes(binary_mask, mask=(binary_mask == 1), transform=transform):
        geom = shape(geom_dict)
        area_m2 = geom.area
        if area_m2 >= min_area_m2:
            records.append({
                "geometry": geom,
                "area_m2": round(area_m2, 1),
            })

    if not records:
        logger.warning("Nenhum candidato a recife encontrado com o limiar especificado.")
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs=crs)

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(f"Detetados {len(gdf)} aglomerados de recife natural (área mínima: {min_area_m2} m²).")
    return gdf
