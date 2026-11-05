#!/usr/bin/env python3
"""
extract_real_pairs.py — Extract semi-real labeled pairs from historical data
================================================================================

Generates pairwise training data from:
1. Real bathymetry features (IH/DGRM service) for 15 Algarve sites
2. Visibility scores from previous model predictions (as proxy for ground truth ranking)
3. Semi-synthetic S2 features (SNR, Kd, transmittance) varied realistically by bathy zone

Output format compatible with train_feature_ranker.py

Usage:
    python scripts/extract_real_pairs.py
    # Generates: data/real_pairwise_features.csv, data/real_pairwise_labels.csv
"""

import sys
import json
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FEATURES = DATA_DIR / "real_pairwise_features.csv"
OUTPUT_LABELS = DATA_DIR / "real_pairwise_labels.csv"

# Zone-based S2 parameter ranges (from production analysis)
# These are semi-synthetic but realistically grounded
ZONE_S2_PROFILES = {
    "very_shallow": {"snr_range": (50, 90), "kd_range": (0.040, 0.055), "base_quality": 0.85},
    "shallow_reef": {"snr_range": (40, 70), "kd_range": (0.045, 0.065), "base_quality": 0.75},
    "nearshore_mid": {"snr_range": (25, 55), "kd_range": (0.055, 0.075), "base_quality": 0.65},
    "mid_depth": {"snr_range": (15, 40), "kd_range": (0.065, 0.090), "base_quality": 0.50},
    "offshore": {"snr_range": (8, 25), "kd_range": (0.075, 0.120), "base_quality": 0.35},
    "unknown": {"snr_range": (20, 50), "kd_range": (0.050, 0.080), "base_quality": 0.55},
}


def compute_transmittance(kd: float, depth_m: float = 16.0) -> float:
    """Beer-Lambert two-way transmittance."""
    import math
    return math.exp(-2 * kd * depth_m)


def generate_s2_features_for_zone(zone: str, seed_offset: int = 0) -> dict:
    """Generate semi-synthetic S2 features based on bathymetry zone."""
    profile = ZONE_S2_PROFILES.get(zone, ZONE_S2_PROFILES["unknown"])
    
    np.random.seed(42 + seed_offset)  # Reproducible per call
    
    snr = np.random.uniform(*profile["snr_range"])
    kd = np.random.uniform(*profile["kd_range"])
    trans = compute_transmittance(kd)
    
    # Cleanliness correlates with SNR and zone quality
    base_clean = profile["base_quality"] * 15000
    noise_factor = np.random.uniform(0.7, 1.3)
    cleanliness = min(15000, max(1000, base_clean * noise_factor * (snr / 50)))
    
    # Contrast - semi-synthetic based on transmittance (higher trans = better contrast visibility)
    # In reality this should come from actual image analysis, but we use proxy
    contrast = min(0.95, 0.5 + 0.4 * trans + np.random.normal(0, 0.05))
    contrast = max(0.1, contrast)  # Floor at 0.1
    
    return {
        "kd_b02": round(kd, 5),
        "water_trans": round(trans, 5),
        "contrast": round(contrast, 4),
        "signal_strength": round(snr, 2),
        "cleanliness": round(cleanliness, 1),
    }


def load_site_data() -> pd.DataFrame:
    """Load training_features.csv with real bathy + visibility scores."""
    training_path = DATA_DIR / "training_features.csv"
    
    if not training_path.exists():
        log.error(f"Training features file not found: {training_path}")
        sys.exit(1)
    
    df = pd.read_csv(training_path)
    log.info(f"Loaded {len(df)} sites from {training_path}")
    return df


def generate_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate all pairwise combinations from sites.
    Winner = site with higher visibility_score.
    """
    records = []
    labels = []
    pair_id = 0
    
    sites = df.to_dict('records')
    
    for site_a, site_b in itertools.combinations(sites, 2):
        site_a_name = site_a['site']
        site_b_name = site_b['site']
        
        score_a = site_a['visibility_score']
        score_b = site_b['visibility_score']
        
        # Generate semi-synthetic S2 features for each site (different realizations)
        s2_a = generate_s2_features_for_zone(site_a['bathy_zone_class'], seed_offset=pair_id * 2)
        s2_b = generate_s2_features_for_zone(site_b['bathy_zone_class'], seed_offset=pair_id * 2 + 1)
        
        # Determine winner based on visibility_score
        if score_a > score_b:
            winner = site_a_name
            label = "A"
        elif score_b > score_a:
            winner = site_b_name
            label = "B"
        else:
            # Tie - skip or random
            continue
        
        # Record A features
        records.append({
            "image_id": f"{site_a_name}_pair{pair_id}",
            "site": site_a_name,
            "pair_id": pair_id,
            "side": "A",
            **s2_a,  # Semi-synthetic S2
            # Real bathy features from site A
            "nearest_isobath_distance_m": site_a['nearest_isobath_distance_m'],
            "nearest_isobath_depth_m": site_a['nearest_isobath_depth_m'],
            "dist_to_isobath_10m": site_a['dist_to_isobath_10m'],
            "dist_to_isobath_20m": site_a['dist_to_isobath_20m'],
            "dist_to_isobath_30m": site_a['dist_to_isobath_30m'],
            "dist_to_isobath_50m": site_a['dist_to_isobath_50m'],
            "dist_to_isobath_100m": site_a['dist_to_isobath_100m'],
            "bathy_zone_class": site_a['bathy_zone_class'],
            "bathy_slope_proxy": site_a['bathy_slope_proxy'],
            "contour_density_proxy": site_a['contour_density_proxy'],
            "n_isobaths_aoi": site_a['n_isobaths_aoi'],
        })
        
        # Record B features
        records.append({
            "image_id": f"{site_b_name}_pair{pair_id}",
            "site": site_b_name,
            "pair_id": pair_id,
            "side": "B",
            **s2_b,  # Semi-synthetic S2
            # Real bathy features from site B
            "nearest_isobath_distance_m": site_b['nearest_isobath_distance_m'],
            "nearest_isobath_depth_m": site_b['nearest_isobath_depth_m'],
            "dist_to_isobath_10m": site_b['dist_to_isobath_10m'],
            "dist_to_isobath_20m": site_b['dist_to_isobath_20m'],
            "dist_to_isobath_30m": site_b['dist_to_isobath_30m'],
            "dist_to_isobath_50m": site_b['dist_to_isobath_50m'],
            "dist_to_isobath_100m": site_b['dist_to_isobath_100m'],
            "bathy_zone_class": site_b['bathy_zone_class'],
            "bathy_slope_proxy": site_b['bathy_slope_proxy'],
            "contour_density_proxy": site_b['contour_density_proxy'],
            "n_isobaths_aoi": site_b['n_isobaths_aoi'],
        })
        
        # Label
        labels.append({
            "pair_id": pair_id,
            "image_a": f"{site_a_name}_pair{pair_id}",
            "image_b": f"{site_b_name}_pair{pair_id}",
            "winner": f"{winner}_pair{pair_id}",
            "winner_side": label,
            "score_a": score_a,
            "score_b": score_b,
            "score_delta": abs(score_a - score_b),
        })
        
        pair_id += 1
    
    df_features = pd.DataFrame(records)
    df_labels = pd.DataFrame(labels)
    
    return df_features, df_labels


def main():
    log.info("=== Extracting Semi-Real Labeled Pairs ===")
    
    # Load site data
    df_sites = load_site_data()
    
    # Generate pairs
    log.info("Generating pairwise combinations...")
    df_features, df_labels = generate_pairs(df_sites)
    
    log.info(f"Generated {len(df_labels)} pairs from {len(df_sites)} sites")
    log.info(f"  - Features: {len(df_features)} records")
    log.info(f"  - Labels: {len(df_labels)} comparisons")
    
    # Stats
    a_wins = sum(df_labels['winner_side'] == 'A')
    b_wins = sum(df_labels['winner_side'] == 'B')
    log.info(f"  - A wins: {a_wins}, B wins: {b_wins}")
    
    # Save
    df_features.to_csv(OUTPUT_FEATURES, index=False)
    df_labels.to_csv(OUTPUT_LABELS, index=False)
    
    log.info(f"Saved: {OUTPUT_FEATURES}")
    log.info(f"Saved: {OUTPUT_LABELS}")
    
    # Also save metadata
    meta = {
        "source": "training_features.csv",
        "n_sites": len(df_sites),
        "n_pairs": len(df_labels),
        "sites": df_sites['site'].tolist(),
        "generation_method": "all_combinations_with_visibility_score_labels",
        "s2_features": "semi_synthetic_zone_based",
        "bathy_features": "real_ih_dgrm",
    }
    meta_path = DATA_DIR / "real_pairs_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved metadata: {meta_path}")
    
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
