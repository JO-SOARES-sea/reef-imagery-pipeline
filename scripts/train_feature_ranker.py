#!/usr/bin/env python3
"""
train_feature_ranker.py — Treino do Modelo de Ranking Tabular (Learning-to-Rank)
================================================================================

Treina um modelo (Random Forest) utilizando as features extraídas pelo motor físico.
O objetivo é classificar/pontuar as imagens com base em preferências pairwise
(Qual é a melhor imagem de um par).

Features esperadas (B02-only, consistent with BVI model): benthic_contrast, snr, fft_clean, edge_entropy, dyn_range, signal

Gera:
- artifacts/training_features.csv
- artifacts/pairwise_labels.csv
- models/feature_ranker_model.pkl
- models/feature_ranker_metadata.json
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)

# Diretórios
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, 'artifacts')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# B02-only features used by the synthetic fallback (8 expert-scored dates).
SYNTHETIC_FEATURE_COLS = [
    "benthic_contrast",    # Edge strength (Sobel+Laplacian on B02)
    "snr",                 # Signal-to-noise ratio of B02
    "fft_clean",           # FFT cleanliness of B02
    "edge_entropy",        # Structural complexity of B02
    "dyn_range",           # Direct BVI weight lookup
    "signal",              # Raw B02 signal level
]

# Physical/bathymetric features available in data/real_pairwise_features.csv
# (15 Algarve sites × 13–14 pairs per site = 103 pairs total). When real data
# is detected, we substitute these for SYNTHETIC_FEATURE_COLS at runtime.
# `nearest_isobath_proximity` is derived (1 / (1 + d/100m)) from the raw
# nearest_isobath_distance_m column to avoid the inf-sentinel artefact.
REAL_NUMERIC_COLS = [
    "kd_b02",
    "water_trans",
    "image_contrast",          # renamed from CSV's 'contrast' to avoid clash
                               # with DISABLED_FEATURES["contrast"] (B02-only)
    "signal_strength",
    "cleanliness",
    "bathy_slope_proxy",
    "contour_density_proxy",
    "n_isobaths_aoi",
    "nearest_isobath_proximity",
]
BATHY_ZONE_VALUES = ["very_shallow", "nearshore_mid", "shallow_reef", "offshore", "unknown"]
REAL_ZONE_COLS = [f"zone_{v}" for v in BATHY_ZONE_VALUES[:-1]]  # 'unknown' is reference
REAL_FEATURE_COLS = REAL_NUMERIC_COLS + REAL_ZONE_COLS

# Bound at module import to the synthetic schema; main() may swap to real.
FEATURE_COLS = SYNTHETIC_FEATURE_COLS

DISABLED_FEATURES = {"contrast": "non-discriminative in B02-only pipeline"}


def _augment_real_features(df):
    """Add derived columns (proximity, one-hot zones, renamed contrast)."""
    import numpy as np
    if "nearest_isobath_distance_m" in df.columns:
        d = df["nearest_isobath_distance_m"].replace([np.inf, -np.inf], np.nan)
        df["nearest_isobath_proximity"] = (1.0 / (1.0 + d / 100.0)).fillna(0.0)
    if "bathy_zone_class" in df.columns:
        for v in BATHY_ZONE_VALUES[:-1]:
            df[f"zone_{v}"] = (df["bathy_zone_class"] == v).astype(float)
    if "contrast" in df.columns and "image_contrast" not in df.columns:
        df["image_contrast"] = df["contrast"]
    return df

def load_real_pairwise_data():
    """
    Tenta carregar pares semi-reais de data/real_pairwise_*.csv.
    Retorna (df_features, df_labels) ou (None, None) se não disponível.
    """
    real_features_path = os.path.join(PROJECT_DIR, 'data', 'real_pairwise_features.csv')
    real_labels_path = os.path.join(PROJECT_DIR, 'data', 'real_pairwise_labels.csv')
    
    if not os.path.exists(real_features_path) or not os.path.exists(real_labels_path):
        log.info("Real pairwise data not found, will use synthetic.")
        return None, None
    
    try:
        df_features = pd.read_csv(real_features_path)
        df_labels = pd.read_csv(real_labels_path)

        # The real CSV ships physical/bathymetric features; derive the
        # ones the model expects (proximity + zone one-hot) before checking.
        df_features = _augment_real_features(df_features)

        required_cols = ['image_id'] + REAL_FEATURE_COLS
        missing = [c for c in required_cols if c not in df_features.columns]
        if missing:
            log.warning(f"Real data missing columns: {missing}, falling back to synthetic.")
            return None, None

        log.info(f"Loaded real pairwise data: {len(df_labels)} pairs from {len(df_features)} records "
                 f"({df_features['site'].nunique() if 'site' in df_features.columns else '?'} sites)")
        return df_features, df_labels
    except Exception as e:
        log.warning(f"Failed to load real data: {e}, falling back to synthetic.")
        return None, None


def generate_dummy_pairwise_data(**kwargs):
    """Real expert-labeled data from 8 Sentinel-2 images of Pedra de Santa Eulalia."""
    import itertools

    images = [
        {"id": "2023-03-15", "snr": 98.5, "fft_clean": 8664, "benthic_contrast": 0.2, "edge_entropy": 7.26, "dyn_range": 0.013, "signal": 0.141},
        {"id": "2025-03-29", "snr": 127.9, "fft_clean": 13478, "benthic_contrast": 0.2, "edge_entropy": 6.00, "dyn_range": 0.008, "signal": 0.136},
        {"id": "2023-09-01", "snr": 137.0, "fft_clean": 17044, "benthic_contrast": 0.1, "edge_entropy": 6.51, "dyn_range": 0.005, "signal": 0.122},
        {"id": "2025-09-15", "snr": 138.4, "fft_clean": 17221, "benthic_contrast": 0.1, "edge_entropy": 7.29, "dyn_range": 0.005, "signal": 0.118},
        {"id": "2025-09-25", "snr": 129.6, "fft_clean": 15034, "benthic_contrast": 0.1, "edge_entropy": 6.63, "dyn_range": 0.008, "signal": 0.127},
        {"id": "2026-02-22", "snr": 86.0, "fft_clean": 6139, "benthic_contrast": 0.2, "edge_entropy": 6.68, "dyn_range": 0.014, "signal": 0.125},
        {"id": "2025-10-05", "snr": 108.1, "fft_clean": 10764, "benthic_contrast": 0.1, "edge_entropy": 7.23, "dyn_range": 0.007, "signal": 0.120},
        {"id": "2024-09-30", "snr": 129.1, "fft_clean": 704, "benthic_contrast": 1.1, "edge_entropy": 1.98, "dyn_range": 0.006, "signal": 0.120},
    ]

    expert_scores = {
        "2025-09-25": 1.0,
        "2024-09-30": 0.7,
    }

    records = []
    labels = []
    pair_id = 0
    for a, b in itertools.combinations(images, 2):
        score_a = expert_scores.get(a["id"], 0.1)
        score_b = expert_scores.get(b["id"], 0.1)
        if score_a != score_b:
            winner = a if score_a > score_b else b
            loser = b if score_a > score_b else a
            records.append({"image_id": winner["id"], **{k: winner[k] for k in FEATURE_COLS}})
            records.append({"image_id": loser["id"], **{k: loser[k] for k in FEATURE_COLS}})
            labels.append({"pair_id": pair_id, "image_a": a["id"], "image_b": b["id"], "winner": winner["id"]})
            pair_id += 1

    df_features = pd.DataFrame(records).drop_duplicates(subset="image_id")
    df_labels = pd.DataFrame(labels)
    return df_features, df_labels

def prepare_pointwise_dataset(df_features, df_labels):
    """
    Converte as preferências pairwise num dataset pointwise para um Regressor.
    Winner recebe score 1.0, Loser recebe score 0.0.
    Isto permite inferir um score absoluto para qualquer imagem individualmente na Fase 3.
    """
    X = []
    y = []
    
    for _, row in df_labels.iterrows():
        feat_a = df_features[df_features['image_id'] == row['image_a']].iloc[0]
        feat_b = df_features[df_features['image_id'] == row['image_b']].iloc[0]
        
        if row['winner'] == row['image_a']:
            X.append(feat_a[FEATURE_COLS].values)
            y.append(1.0)
            X.append(feat_b[FEATURE_COLS].values)
            y.append(0.0)
        else:
            X.append(feat_b[FEATURE_COLS].values)
            y.append(1.0)
            X.append(feat_a[FEATURE_COLS].values)
            y.append(0.0)
            
    # Criar um DataFrame limpo
    df_X = pd.DataFrame(X, columns=FEATURE_COLS)
    return df_X, np.array(y)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("=== Treino do Ranker de Features Físicas ===")
    
    # 1. Carregar/Gerar Dados
    log.info("Loading dataset...")
    
    # Try real data first. The real CSV has physical/bathymetric features;
    # when that schema is in use, swap FEATURE_COLS to REAL_FEATURE_COLS so
    # the rest of the pipeline (pointwise prep, training, importance) sees
    # the right feature set.
    global FEATURE_COLS
    df_features, df_labels = load_real_pairwise_data()
    data_source = "real"
    if df_features is not None:
        FEATURE_COLS = REAL_FEATURE_COLS
    else:
        log.info("Generating synthetic pairwise data...")
        df_features, df_labels = generate_dummy_pairwise_data()
        data_source = "synthetic"
        FEATURE_COLS = SYNTHETIC_FEATURE_COLS

    log.info(f"Using {data_source} data: {len(df_labels)} pairs, {len(FEATURE_COLS)} features")
    
    # Exportar datasets para auditabilidade
    features_csv = os.path.join(ARTIFACTS_DIR, 'training_features.csv')
    labels_csv = os.path.join(ARTIFACTS_DIR, 'pairwise_labels.csv')
    df_features.to_csv(features_csv, index=False)
    df_labels.to_csv(labels_csv, index=False)
    log.info(f"Dataset exportado: {features_csv} e {labels_csv}")
    
    # 2. Preparar Dataset Pointwise e Split
    X, y = prepare_pointwise_dataset(df_features, df_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log.info(f"Split de dados: {len(X_train)} amostras de treino, {len(X_test)} amostras de validação/teste.")
    
    # 3. Treinar Modelo
    log.info("A treinar RandomForestRegressor (Pointwise Ranker)...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Avaliar Modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Transformar em classificação binária (>= 0.5 é "melhor")
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_test_bin = (y_test >= 0.5).astype(int)
    acc = accuracy_score(y_test_bin, y_pred_bin)
    
    log.info(f"Resultados Validação -> MSE: {mse:.4f} | Precisão (Ranking Binário): {acc:.2%}")
    
    # 5. Importância das Features e Permutation Importance
    importance = model.feature_importances_
    feat_importance = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
    log.info("Importância Gini das Features:")
    for feat, imp in feat_importance:
        log.info(f"  - {feat}: {imp:.4f}")
        
    log.info("A calcular Permutation Importance no conjunto de teste...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance_dict = {}
    log.info("Permutation Importance:")
    for i, col in enumerate(FEATURE_COLS):
        perm_importance_dict[col] = {
            "mean": float(perm_importance.importances_mean[i]),
            "std": float(perm_importance.importances_std[i])
        }
        log.info(f"  - {col}: {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")
        
    # Export Permutation Importance to artifacts
    perm_path = os.path.join(ARTIFACTS_DIR, 'permutation_importance.json')
    with open(perm_path, 'w') as f:
        json.dump(perm_importance_dict, f, indent=2)
    log.info(f"Permutation Importance guardada em: {perm_path}")
        
    # 6. Guardar Modelo e Metadados
    model_path = os.path.join(MODELS_DIR, 'feature_ranker_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    import datetime
    metadata = {
        "model_version": "2.0",
        "schema_version": "3.0",
        "canonical": True,
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "algorithm": "RandomForestRegressor",
        "features": FEATURE_COLS,
        "data_source": data_source,
        "metrics": {
            "validation_mse": mse,
            "validation_accuracy": acc
        },
        "feature_importance": {feat: imp for feat, imp in feat_importance},
        "dataset_size_pairs": len(df_labels),
        "disabled_features": {
            "contrast": "Constant proxy (0.8) — does not discriminate images"
        },
        "deprecated_models": [
            "visibility_rf_bathy.pkl"
        ],
    }
    
    meta_path = os.path.join(MODELS_DIR, 'feature_ranker_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    log.info(f"Modelo guardado em: {model_path}")
    log.info(f"Metadados guardados em: {meta_path}")
    log.info("=== Concluído ===")

if __name__ == "__main__":
    main()
