#!/usr/bin/env python3
"""
train_bathy_ml.py — Train ML model with bathymetry features
===========================================================

Trains a Random Forest regressor to predict underwater visibility score
using bathymetry-derived features + Sentinel-2 physical variables.

Features used:
    - Sentinel-2: snr_mean_16m, kd_b02_estimated, water_transmittance_twoway
    - Bathymetry: nearest_isobath_distance_m, nearest_isobath_depth_m,
                  dist_to_isobath_10/20/30/50/100m, bathymetry_zone_class,
                  bathy_slope_proxy, contour_density_proxy

Target: visibility_score (0-1, higher = better visibility)

Usage:
    python scripts/train_bathy_ml.py
    python scripts/train_bathy_ml.py --output-model models/visibility_rf.pkl

Author: 3ruiruirui-sketch
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ih_bathy_features import BathyFeatureEngine

log = logging.getLogger(__name__)

# ── Known reef observation points in Algarve ─────────────────────────────────
# Each point has a known visibility score from historical pipeline runs
TRAINING_SITES = [
    # (lon, lat, name, approximate visibility score from past runs)
    (-8.2193, 37.0636, "pedra_do_alto", 0.72),
    (-8.210492, 37.069071, "albufeira_reef", 0.68),
    (-8.2296, 37.0562, "gale_spot", 0.65),
    (-7.6603, 37.0468, "east_reef", 0.58),
    (-8.20982, 37.05815, "target_main", 0.70),
    (-8.20, 37.07, "albufeira_north", 0.60),
    (-8.25, 37.05, "albufeira_west", 0.55),
    (-8.15, 37.10, "olhos_de_agua", 0.62),
    (-8.0, 37.1, "faro_east", 0.50),
    (-7.8, 37.0, "tavira", 0.45),
    (-8.5, 37.0, "sagres", 0.75),
    (-8.3, 37.08, "vilamoura", 0.58),
    (-8.1, 37.15, "alvor", 0.63),
    (-8.4, 36.95, "sagres_south", 0.70),
    (-7.5, 37.1, "vila_real_sto_antonio", 0.40),
]

# Synthetic Sentinel-2 features (in production, these come from summary.csv)
# Each point gets a slightly different S2 signature based on its location
np.random.seed(42)


def build_training_df() -> pd.DataFrame:
    """Build training dataframe with bathymetry + S2 features for all sites."""
    engine = BathyFeatureEngine(cache_dir="data/cache")

    records = []
    for lon, lat, name, vis_score in TRAINING_SITES:
        log.info("Computing features for %s (%.4f, %.4f)...", name, lon, lat)
        bathy = engine.compute_features_for_point(lon, lat, buffer_m=5_000)

        # Synthetic Sentinel-2 features based on zone and depth
        zone = bathy["bathymetry_zone_class"]
        depth_m = bathy["nearest_isobath_depth_m"] or 20.0

        # Very shallow = better SNR, clearer water
        if zone == "very_shallow":
            base_snr = 45.0
            base_kd = 0.045
            base_trans = 0.85
        elif zone == "shallow_reef":
            base_snr = 35.0
            base_kd = 0.055
            base_trans = 0.75
        elif zone == "nearshore_mid":
            base_snr = 25.0
            base_kd = 0.065
            base_trans = 0.60
        else:
            base_snr = 15.0
            base_kd = 0.080
            base_trans = 0.45

        # Add some noise
        snr = base_snr + np.random.normal(0, 5)
        kd = base_kd + np.random.normal(0, 0.01)
        trans = base_trans + np.random.normal(0, 0.05)

        record = {
            # Metadata
            "site": name,
            "lon": lon,
            "lat": lat,
            # Sentinel-2 features
            "snr_mean_16m": round(snr, 2),
            "kd_b02_estimated": round(kd, 4),
            "water_transmittance_twoway": round(trans, 4),
            # Bathymetry features
            "nearest_isobath_distance_m": bathy["nearest_isobath_distance_m"],
            "nearest_isobath_depth_m": bathy["nearest_isobath_depth_m"],
            "dist_to_isobath_10m": bathy["dist_to_isobath_10m"],
            "dist_to_isobath_20m": bathy["dist_to_isobath_20m"],
            "dist_to_isobath_30m": bathy["dist_to_isobath_30m"],
            "dist_to_isobath_50m": bathy["dist_to_isobath_50m"],
            "dist_to_isobath_100m": bathy["dist_to_isobath_100m"],
            "bathy_zone_class": bathy["bathymetry_zone_class"],
            "bathy_slope_proxy": bathy["bathymetry_slope_proxy"],
            "contour_density_proxy": bathy["contour_density_proxy"],
            "n_isobaths_aoi": bathy["n_isobaths_in_aoi"],
            # Target
            "visibility_score": vis_score,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def train_model(df: pd.DataFrame):
    """Train a Random Forest regressor."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    # Prepare features
    feature_cols = [
        "snr_mean_16m",
        "kd_b02_estimated",
        "water_transmittance_twoway",
        "nearest_isobath_distance_m",
        "nearest_isobath_depth_m",
        "dist_to_isobath_10m",
        "dist_to_isobath_20m",
        "dist_to_isobath_30m",
        "dist_to_isobath_50m",
        "dist_to_isobath_100m",
        "bathy_slope_proxy",
        "contour_density_proxy",
        "n_isobaths_aoi",
    ]

    # Encode categorical
    le = LabelEncoder()
    df["bathy_zone_class_enc"] = le.fit_transform(df["bathy_zone_class"].fillna("unknown"))
    feature_cols.append("bathy_zone_class_enc")

    X = df[feature_cols].copy()
    # Handle inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df["visibility_score"].values

    # Train/test split (small dataset — use LOOCV-like approach)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        preds[test_idx] = rf.predict(X_test)[0]

    mae = np.mean(np.abs(preds - y))
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    r2 = 1 - np.sum((preds - y) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Final model on all data
    rf_final = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    rf_final.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_final.feature_importances_,
    }).sort_values("importance", ascending=False)

    return rf_final, le, importance, {"mae": mae, "rmse": rmse, "r2": r2}, preds


def main():
    parser = argparse.ArgumentParser(description="Train visibility model with bathymetry features")
    parser.add_argument("--output-model", default="models/visibility_rf_bathy.pkl",
                        help="Path to save trained model")
    parser.add_argument("--output-csv", default="data/training_features.csv",
                        help="Path to save training features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("=" * 60)
    log.info("Building training dataset with bathymetry features...")
    log.info("=" * 60)

    df = build_training_df()

    # Save training data
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Training data saved: %s (%d rows)", out_csv, len(df))

    log.info("\nTraining features preview:")
    print(df[["site", "visibility_score", "bathy_zone_class",
              "nearest_isobath_distance_m", "nearest_isobath_depth_m"]].to_string())

    log.info("\n" + "=" * 60)
    log.info("Training Random Forest model...")
    log.info("=" * 60)

    model, le, importance, metrics, preds = train_model(df)

    log.info("\nCross-validation results (Leave-One-Out):")
    log.info("  MAE:  %.4f", metrics["mae"])
    log.info("  RMSE: %.4f", metrics["rmse"])
    log.info("  R²:   %.4f", metrics["r2"])

    log.info("\nPredictions vs Actual:")
    for i, row in df.iterrows():
        log.info("  %-25s | actual=%.3f | pred=%.3f | err=%+.3f",
                 row["site"], row["visibility_score"], preds[i],
                 preds[i] - row["visibility_score"])

    log.info("\nFeature importance:")
    for _, row in importance.iterrows():
        log.info("  %-30s %.4f", row["feature"], row["importance"])

    # Save model
    import pickle
    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with open(out_model, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "features": importance["feature"].tolist()}, f)
    log.info("\nModel saved: %s", out_model)

    # Save a JSON summary
    summary = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 200,
        "max_depth": 5,
        "n_training_sites": len(df),
        "metrics": metrics,
        "feature_importance": importance.to_dict("records"),
        "zone_label_mapping": dict(zip(le.classes_, range(len(le.classes_)))),
    }
    summary_path = out_model.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Summary saved: %s", summary_path)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
