#!/usr/bin/env python3
"""
predict_bathy_ml.py — Predict bathymetry-based visibility score for any location
=================================================================================

Uses the trained Random Forest model (models/visibility_rf_bathy.pkl) to predict
underwater visibility scores for new Algarve locations based on IH/DGRM bathymetry
features + synthetic Sentinel-2 signatures.

Usage:
    # Single location prediction
    python scripts/predict_bathy_ml.py --lon -8.21 --lat 37.07
    
    # Batch prediction from CSV
    python scripts/predict_bathy_ml.py --batch locations.csv --output predictions.csv
    
    # With custom S2 parameters
    python scripts/predict_bathy_ml.py --lon -8.21 --lat 37.07 --snr 40 --kd 0.05

Output:
    - Predicted visibility score (0-1, higher = better)
    - Bathymetry zone classification
    - Distance to key isobaths (10m, 20m, 30m)
    - Feature vector used for prediction

Author: 3ruiruirui-sketch
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ih_bathy_features import BathyFeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_model(model_path: str = "models/visibility_rf_bathy.pkl"):
    """Load trained model and label encoder."""
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        return model_data["model"], model_data["label_encoder"], model_data.get("features", [])
    except FileNotFoundError:
        log.error(f"Model not found: {model_path}")
        log.error("Run: python scripts/train_bathy_ml.py")
        sys.exit(1)
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        sys.exit(1)


def prepare_features(bathy_feats: dict, snr_mean: float, kd_b02: float, 
                     transmittance: float, le) -> np.ndarray:
    """Prepare feature vector in exact order expected by model."""
    
    # Zone encoding
    zone = bathy_feats.get("bathymetry_zone_class", "unknown")
    if zone in le.classes_:
        zone_enc = le.transform([zone])[0]
    else:
        zone_enc = le.transform(["unknown"])[0] if "unknown" in le.classes_ else 0
    
    # Handle infinity values
    def safe(val):
        if val is None:
            return 0.0
        return val if val != float("inf") else 99999.0
    
    # Feature order must match training
    X = np.array([[
        safe(bathy_feats.get("dist_to_isobath_30m")),
        safe(bathy_feats.get("dist_to_isobath_10m")),
        safe(bathy_feats.get("nearest_isobath_distance_m")),
        bathy_feats.get("n_isobaths_in_aoi", 0),
        snr_mean,
        kd_b02,
        transmittance,
        bathy_feats.get("contour_density_proxy", 0),
        safe(bathy_feats.get("dist_to_isobath_20m")),
        safe(bathy_feats.get("dist_to_isobath_50m")),
        bathy_feats.get("nearest_isobath_depth_m") or 0,
        zone_enc,
        bathy_feats.get("bathy_slope_proxy", 0),
        safe(bathy_feats.get("dist_to_isobath_100m")),
    ]]).reshape(1, -1)
    
    return X


def predict_for_location(lon: float, lat: float, model, le, features_list,
                         snr_mean: float = None, kd_b02: float = None,
                         transmittance: float = None, cache_dir: str = "data/cache") -> dict:
    """Predict visibility score for a single location."""
    
    # Get bathymetry features
    engine = BathyFeatureEngine(cache_dir=cache_dir)
    bathy = engine.compute_features_for_point(lon=lon, lat=lat, buffer_m=5000)
    
    # Infer S2 parameters from bathymetry zone if not provided
    zone = bathy["bathymetry_zone_class"]
    if snr_mean is None:
        zone_snr_map = {
            "very_shallow": 45.0,
            "shallow_reef": 35.0,
            "nearshore_mid": 25.0,
            "mid_depth": 20.0,
            "offshore": 15.0,
            "unknown": 20.0,
        }
        snr_mean = zone_snr_map.get(zone, 25.0)
    
    if kd_b02 is None:
        zone_kd_map = {
            "very_shallow": 0.045,
            "shallow_reef": 0.055,
            "nearshore_mid": 0.065,
            "mid_depth": 0.075,
            "offshore": 0.085,
            "unknown": 0.065,
        }
        kd_b02 = zone_kd_map.get(zone, 0.065)
    
    if transmittance is None:
        # Beer-Lambert approximation
        import math
        depth_m = bathy.get("nearest_isobath_depth_m") or 16.0
        transmittance = math.exp(-2 * kd_b02 * depth_m)
    
    # Prepare features and predict
    X = prepare_features(bathy, snr_mean, kd_b02, transmittance, le)
    score = model.predict(X)[0]
    
    return {
        "lon": lon,
        "lat": lat,
        "predicted_visibility_score": round(score, 4),
        "bathymetry_zone": zone,
        "nearest_isobath_depth_m": bathy.get("nearest_isobath_depth_m"),
        "nearest_isobath_distance_m": round(bathy.get("nearest_isobath_distance_m", 0), 1),
        "dist_to_10m_isobath_m": round(bathy.get("dist_to_isobath_10m", 0), 1) if bathy.get("dist_to_isobath_10m") != float("inf") else None,
        "dist_to_20m_isobath_m": round(bathy.get("dist_to_isobath_20m", 0), 1) if bathy.get("dist_to_isobath_20m") != float("inf") else None,
        "dist_to_30m_isobath_m": round(bathy.get("dist_to_isobath_30m", 0), 1) if bathy.get("dist_to_isobath_30m") != float("inf") else None,
        "n_isobaths_in_aoi": bathy.get("n_isobaths_in_aoi", 0),
        "contour_density_proxy": round(bathy.get("contour_density_proxy", 0), 2),
        "synthetic_snr_mean_16m": round(snr_mean, 2),
        "synthetic_kd_b02": round(kd_b02, 4),
        "synthetic_transmittance": round(transmittance, 4),
    }


def batch_predict(input_csv: str, output_csv: str, model, le, features_list):
    """Batch prediction from CSV file."""
    df = pd.read_csv(input_csv)
    required = ["lon", "lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Input CSV missing columns: {missing}")
        sys.exit(1)
    
    results = []
    for _, row in df.iterrows():
        result = predict_for_location(
            lon=row["lon"],
            lat=row["lat"],
            model=model,
            le=le,
            features_list=features_list,
            snr_mean=row.get("snr_mean_16m"),
            kd_b02=row.get("kd_b02"),
            transmittance=row.get("transmittance"),
        )
        results.append(result)
    
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    log.info(f"Batch predictions saved: {output_csv} ({len(out_df)} locations)")


def main():
    parser = argparse.ArgumentParser(description="Predict visibility score using bathymetry ML model")
    parser.add_argument("--lon", type=float, help="Longitude (WGS84)")
    parser.add_argument("--lat", type=float, help="Latitude (WGS84)")
    parser.add_argument("--snr", type=float, default=None, help="Sentinel-2 SNR mean (optional)")
    parser.add_argument("--kd", type=float, default=None, help="Kd(490) value (optional)")
    parser.add_argument("--transmittance", type=float, default=None, help="Water transmittance (optional)")
    parser.add_argument("--batch", type=str, help="Input CSV with columns: lon,lat[,snr_mean_16m,kd_b02,transmittance]")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV for batch mode")
    parser.add_argument("--model", type=str, default="models/visibility_rf_bathy.pkl", help="Path to trained model")
    parser.add_argument("--json", action="store_true", help="Output as JSON (single prediction only)")
    args = parser.parse_args()
    
    # Load model
    log.info("Loading model...")
    model, le, features_list = load_model(args.model)
    log.info(f"✓ Model loaded: {len(features_list)} features")
    
    if args.batch:
        # Batch mode
        batch_predict(args.batch, args.output, model, le, features_list)
    else:
        # Single location mode
        if args.lon is None or args.lat is None:
            parser.error("--lon and --lat required (or use --batch)")
        
        result = predict_for_location(
            lon=args.lon,
            lat=args.lat,
            model=model,
            le=le,
            features_list=features_list,
            snr_mean=args.snr,
            kd_b02=args.kd,
            transmittance=args.transmittance,
        )
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n" + "="*60)
            print(f"LOCATION: ({result['lon']:.4f}°W, {result['lat']:.4f}°N)")
            print("="*60)
            print(f"\n📊 PREDICTED VISIBILITY SCORE: {result['predicted_visibility_score']:.3f}")
            print("   (0=poor, 1=excellent)")
            print(f"\n🌊 BATHYMETRY ZONE: {result['bathymetry_zone']}")
            print(f"   Nearest isobath: {result['nearest_isobath_depth_m']}m at {result['nearest_isobath_distance_m']}m")
            print("\n📏 DISTANCE TO KEY ISOBATHS:")
            if result['dist_to_10m_isobath_m']:
                print(f"   10m isobath: {result['dist_to_10m_isobath_m']:.0f}m")
            if result['dist_to_20m_isobath_m']:
                print(f"   20m isobath: {result['dist_to_20m_isobath_m']:.0f}m")
            if result['dist_to_30m_isobath_m']:
                print(f"   30m isobath: {result['dist_to_30m_isobath_m']:.0f}m")
            print("\n📡 SYNTHETIC S2 PARAMETERS:")
            print(f"   SNR mean: {result['synthetic_snr_mean_16m']}")
            print(f"   Kd(490): {result['synthetic_kd_b02']}")
            print(f"   Transmittance: {result['synthetic_transmittance']}")
            print("="*60)


if __name__ == "__main__":
    main()
