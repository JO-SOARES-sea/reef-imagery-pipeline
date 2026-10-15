#!/usr/bin/env python3
"""
Reef Visibility ML Predictor v2.0 (Hybrid Physical-Heuristic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Atualizado com modelação ótica da coluna de água (Kd490, Secchi).
Calcula o contraste real entre Areia e Rocha a uma profundidade alvo,
considerando a refração da luz solar (Lei de Snell) e atenuação sazonal.
Calibrado com os inputs reais do confronto entre 2025-09-25 e 2023-10-01.
"""

import argparse
import math
from datetime import datetime, timezone

import pandas as pd
import planetary_computer as pc
from pystac_client import Client

from src.constants import GLINT_PENALTY, GLINT_PENALTY_DEFAULT, KD490_TABLE
from src.ranking_model import predict_score
from src.utils import get_kd490

try:
    from src.cmems_kd490 import KD490_TABLE_LIVE as KD490_TABLE  # shadows the static import above
except Exception:
    pass  # keep static KD490_TABLE on any failure

# Constantes Físicas (Banda B02 - Azul 490nm / Banda B03 - Verde 560nm)
SAND_R_REF = 0.25  # Refletância da areia branca
ROCK_R_REF = 0.05  # Refletância do recife escuro
N_WATER = 1.333  # Índice de refração da água do mar


def extract_features_from_stac(row, depth):
    cc = row["cloud_cover"]
    sun_el = row["sun_elevation"]
    month = row["datetime"].month

    # 1. Filtro Crítico de Nuvens
    if cc > 80:
        cloud_transmittance = 0.0
    else:
        cloud_transmittance = max(0.0, 1.0 - (cc / 100.0))

    # 2. Ótica Geométrica (Lei de Snell e Ângulo Zenital)
    sza_air = 90.0 - sun_el  # Solar Zenith Angle no ar
    if sza_air >= 90:
        sza_air = 89.9

    glint_penalty = GLINT_PENALTY.get(month, GLINT_PENALTY_DEFAULT)

    # Refração na água (SZA_underwater)
    sin_sza_water = math.sin(math.radians(sza_air)) / N_WATER
    sza_water = math.degrees(math.asin(sin_sza_water))

    # Distância real que a luz percorre na água
    optical_path_length = depth / math.cos(math.radians(sza_water))

    # 3. Atenuação da Água (Kd_B02 = Kd490 sazonal para banda azul)
    kd_b02 = get_kd490(month, KD490_TABLE)

    # Transmitância da coluna de água (ida e volta)
    water_trans = math.exp(-2 * kd_b02 * optical_path_length)

    # 4. Cálculo do Sinal e Contraste Bentónico
    sand_sig = SAND_R_REF * water_trans * cloud_transmittance * glint_penalty
    rock_sig = ROCK_R_REF * water_trans * cloud_transmittance * glint_penalty

    if sand_sig <= 0:
        contrast = 0.0
    else:
        contrast = ((sand_sig - rock_sig) / sand_sig) * 100.0

    signal_strength = sand_sig / SAND_R_REF

    # Return features dict instead of hardcoded score
    return {
        "cloud_cover": cc,
        "sza_air": sza_air,
        "glint_penalty": glint_penalty,
        "kd_b02": kd_b02,
        "water_trans": water_trans,
        "contrast": contrast,
        "signal_strength": signal_strength,
        "cleanliness": 5000,  # Default for STAC metadata phase (before image download)
    }


def predict_top_5_days(lat, lon, depth, years_back=4):
    # Drift monitoring: reset at batch start (shadow mode, best-effort)
    _has_drift = False
    try:
        from src.drift_export import export_to_file as drift_export_file
        from src.drift_monitor import log_summary as drift_log_summary
        from src.drift_monitor import reset as drift_reset

        drift_reset()
        _has_drift = True
    except ImportError:
        _has_drift = False
    except Exception:
        # drift_reset() threw something other than ImportError (e.g. RuntimeError).
        # Keep _has_drift = False — shadow mode must never block inference.
        _has_drift = False

    print(f"🔍 A pesquisar histórico STAC para [{lat:.4f}, {lon:.4f}] com Modelação Físico-Ótica a {depth:.1f} metros...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    end_date = datetime.now(timezone.utc)
    start_date = datetime(end_date.year - years_back, 1, 1, tzinfo=timezone.utc)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": 5}},
    )

    items = list(search.items())
    if not items:
        print("Nenhuma passagem limpa encontrada.")
        return

    data = []
    for item in items:
        props = item.properties
        if props.get("s2:nodata_pixel_percentage", 100) > 20:
            continue

        data.append(
            {
                "date_str": item.datetime.strftime("%Y-%m-%d"),
                "datetime": item.datetime,
                "cloud_cover": props.get("eo:cloud_cover", 100),
                "sun_elevation": props.get("view:sun_elevation", 45),  # assume 45 if missing
                "id": item.id,
            }
        )

    df = pd.DataFrame(data)

    # Aplicar Motor Físico + Modelo de Ranking ML
    def apply_ranker(r):
        features = extract_features_from_stac(r, depth)
        return predict_score(features)["score"]

    df["physics_score"] = df.apply(apply_ranker, axis=1)

    # Drift monitoring: batch-end summary (shadow mode, best-effort)
    if _has_drift:
        try:
            batch_id = f"stac_{lat:.4f}_{lon:.4f}_{depth:.0f}m"
            drift_log_summary()
            drift_export_file(batch_id=batch_id)
        except Exception:
            pass

    df = df.sort_values("physics_score", ascending=False).drop_duplicates("date_str")
    top5 = df.head(5).reset_index(drop=True)

    print(f"\n🏆 OS 5 MELHORES DIAS (Modelação de Contraste B02/B03 a {depth:.1f} metros)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for idx, row in top5.iterrows():
        print(f"{idx+1}. Data: {row['date_str']} | Contraste Ótico Efetivo: {row['physics_score']:.1f}/100")
        print(f"    ↳ Nuvens: {row['cloud_cover']:.2f}% | Elev. Solar: {row['sun_elevation']:.1f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--depth", type=float, default=22.0, help="Profundidade alvo em metros")
    parser.add_argument("--years", type=int, default=4, help="Número de anos para trás a pesquisar")
    args = parser.parse_args()
    predict_top_5_days(args.lat, args.lon, args.depth, years_back=args.years)
