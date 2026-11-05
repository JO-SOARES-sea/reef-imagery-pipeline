#!/usr/bin/env python3
import logging
from datetime import datetime, timedelta
import pandas as pd

import argparse
# Importa o motor do nosso projeto
from argenta_engine import (
    get_ground_truth_dataset,
    FeatureEngineer,
    ArgentaSafetyModel,
    MultiSourceTideFetcher,
    calcular_altura_barra,
    get_moon_phase_strength,
    get_moon_details,
    visualize_training_performance,
    visualize_feature_importance
)

def main_interactive_loop(argenta_ml, fetcher):
    """
    Função principal que gere o loop interativo com o utilizador.
    """
    while True:
        try:
            user_input = input("\nInsira a Data e Hora da passagem (Formato: YYYY-MM-DD HH:MM) ou 'sair': ")
            if user_input.lower() == 'sair':
                break
            
            target_dt = datetime.strptime(user_input, "%Y-%m-%d %H:%M")
            
            dia_antes = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            dia_depois = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            df_tides_periodo = fetcher.fetch_period(dia_antes, dia_depois)

            # Encontrar os próximos picos de maré e a fase da lua
            df_tides_periodo['Datetime'] = pd.to_datetime(df_tides_periodo['Data'] + ' ' + df_tides_periodo['Hora'], format='%Y-%m-%d %H:%M')
            proximos_picos = df_tides_periodo[df_tides_periodo['Datetime'] > target_dt].copy().sort_values('Datetime')
            
            info_pico_1 = "Não disponível"
            info_pico_2 = "Não disponível"
            target_date_str = target_dt.strftime('%Y-%m-%d')

            if len(proximos_picos) >= 1:
                pico1 = proximos_picos.iloc[0]
                date_info = "" if pico1['Data'] == target_date_str else f" ({pd.to_datetime(pico1['Data']).strftime('%d/%m')})"
                info_pico_1 = f"{pico1['Tipo']} às {pico1['Hora']}{date_info} ({pico1['App_Barra_m']:.2f}m)"
            if len(proximos_picos) >= 2:
                pico2 = proximos_picos.iloc[1]
                date_info = "" if pico2['Data'] == target_date_str else f" ({pd.to_datetime(pico2['Data']).strftime('%d/%m')})"
                info_pico_2 = f"{pico2['Tipo']} às {pico2['Hora']}{date_info} ({pico2['App_Barra_m']:.2f}m)"

            moon_details = get_moon_details(target_dt)

            dados_barra = calcular_altura_barra(target_dt, df_tides_periodo.copy())

            df_tides_dia = df_tides_periodo[df_tides_periodo['Data'] == target_dt.strftime('%Y-%m-%d')]
            amplitude_mare = df_tides_dia['App_Barra_m'].max() - df_tides_dia['App_Barra_m'].min()

            features_para_previsao = {
                'App_Barra_m': dados_barra['App_Barra_m_interpolada'],
                'Fase_Lua_Pct': get_moon_phase_strength(target_dt),
                'Ciclo_Encoded': 1 if dados_barra['Ciclo'] == 'A encher' else -1,
                'Hydro_Pressure_Index': dados_barra['App_Barra_m_interpolada'] * get_moon_phase_strength(target_dt),
                'Mes': target_dt.month,
                'amplitude_mare': amplitude_mare,
                'tempo_ate_pico_min': dados_barra['tempo_ate_pico_min']
            }

            previsao = argenta_ml.prever_offset_doca(features_para_previsao)

            print("\n" + "="*50)
            print(f"  RELATÓRIO DE SEGURANÇA PARA {user_input}")
            print("="*50)
            print(f"  [1] Maré na Barra (Teórica):      {previsao['app_barra_m']:.2f} m")
            print(f"  [2] Offset Doca (ML Preditivo):   {previsao['offset_ml_previsto']:+.2f} m")
            print(f"  [3] Altura Doca (Prevista):       {previsao['altura_doca_prevista_pura']:.2f} m")
            print(f"  [4] Altura Doca (+ Segurança):    {previsao['altura_doca_com_viés_segurança']:.2f} m")
            print("-" * 50)
            print("  🌊 Próximos Picos (Barra):")
            print(f"     - {info_pico_1}")
            print(f"     - {info_pico_2}")
            print(f"  🌙 Fase da Lua: {moon_details}")
            print("-" * 50)
            
            if "LIVRE" in previsao['DECISAO_SEGURANCA']:
                print(f"  🟢 VEREDICTO: PASSE LIVRE - {previsao['altura_doca_com_viés_segurança']:.2f}m (Altura <= {argenta_ml.BRIDGE_LIMIT_M:.2f}m)")
            else:
                print("  🔴 VEREDICTO: PONTE BLOQUEADA - RISCO DE COLISÃO")
            print("="*50)

        except ValueError as e:
            logging.error(f"Erro: {e}. Por favor, use o formato YYYY-MM-DD HH:MM.")
        except Exception as e:
            logging.error(f"Ocorreu um erro inesperado: {e}")

def run_single_prediction(argenta_ml, fetcher, user_input):
    """Executa uma única previsão para uma data e hora específicas."""
    try:
        target_dt = datetime.strptime(user_input, "%Y-%m-%d %H:%M")
        
        dia_antes = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        dia_depois = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        df_tides_periodo = fetcher.fetch_period(dia_antes, dia_depois)

        # Encontrar os próximos picos de maré e a fase da lua
        df_tides_periodo['Datetime'] = pd.to_datetime(df_tides_periodo['Data'] + ' ' + df_tides_periodo['Hora'], format='%Y-%m-%d %H:%M')
        proximos_picos = df_tides_periodo[df_tides_periodo['Datetime'] > target_dt].copy().sort_values('Datetime')

        info_pico_1 = "Não disponível"
        info_pico_2 = "Não disponível"
        target_date_str = target_dt.strftime('%Y-%m-%d')

        if len(proximos_picos) >= 1:
            pico1 = proximos_picos.iloc[0]
            date_info = "" if pico1['Data'] == target_date_str else f" ({pd.to_datetime(pico1['Data']).strftime('%d/%m')})"
            info_pico_1 = f"{pico1['Tipo']} às {pico1['Hora']}{date_info} ({pico1['App_Barra_m']:.2f}m)"
        if len(proximos_picos) >= 2:
            pico2 = proximos_picos.iloc[1]
            date_info = "" if pico2['Data'] == target_date_str else f" ({pd.to_datetime(pico2['Data']).strftime('%d/%m')})"
            info_pico_2 = f"{pico2['Tipo']} às {pico2['Hora']}{date_info} ({pico2['App_Barra_m']:.2f}m)"

        moon_details = get_moon_details(target_dt)

        dados_barra = calcular_altura_barra(target_dt, df_tides_periodo.copy())

        df_tides_dia = df_tides_periodo[df_tides_periodo['Data'] == target_dt.strftime('%Y-%m-%d')]
        amplitude_mare = df_tides_dia['App_Barra_m'].max() - df_tides_dia['App_Barra_m'].min()

        features_para_previsao = {
            'App_Barra_m': dados_barra['App_Barra_m_interpolada'],
            'Fase_Lua_Pct': get_moon_phase_strength(target_dt),
            'Ciclo_Encoded': 1 if dados_barra['Ciclo'] == 'A encher' else -1,
            'Hydro_Pressure_Index': dados_barra['App_Barra_m_interpolada'] * get_moon_phase_strength(target_dt),
            'Mes': target_dt.month,
            'amplitude_mare': amplitude_mare,
            'tempo_ate_pico_min': dados_barra['tempo_ate_pico_min']
        }

        previsao = argenta_ml.prever_offset_doca(features_para_previsao)

        print("\n" + "="*50)
        print(f"  RELATÓRIO DE SEGURANÇA PARA {user_input}")
        print("="*50)
        print(f"  [1] Maré na Barra (Teórica):      {previsao['app_barra_m']:.2f} m")
        print(f"  [2] Offset Doca (ML Preditivo):   {previsao['offset_ml_previsto']:+.2f} m")
        print(f"  [3] Altura Doca (Prevista):       {previsao['altura_doca_prevista_pura']:.2f} m")
        print(f"  [4] Altura Doca (+ Segurança):    {previsao['altura_doca_com_viés_segurança']:.2f} m")
        print("-" * 50)
        print("  🌊 Próximos Picos (Barra):")
        print(f"     - {info_pico_1}")
        print(f"     - {info_pico_2}")
        print(f"  🌙 Fase da Lua: {moon_details}")
        print("-" * 50)
        
        if "LIVRE" in previsao['DECISAO_SEGURANCA']:
            print(f"  🟢 VEREDICTO: PASSE LIVRE - {previsao['altura_doca_com_viés_segurança']:.2f}m (Altura <= {argenta_ml.BRIDGE_LIMIT_M:.2f}m)")
        else:
            print("  🔴 VEREDICTO: PONTE BLOQUEADA - RISCO DE COLISÃO")
        print("="*50)

    except ValueError as e:
        logging.error(f"Erro: {e}. Por favor, use o formato YYYY-MM-DD HH:MM.")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    logging.info("=== SISTEMA PREDITIVO DE NAVEGAÇÃO 'ARGENTA' - DOCA DE FARO ===")
    
    df_raw = get_ground_truth_dataset()
    df_features = FeatureEngineer.build_features(df_raw)
    
    argenta_ml = ArgentaSafetyModel()
    df_features['amplitude_mare'] = 2.1
    df_features['tempo_ate_pico_min'] = 90.0
    argenta_ml.train(df_features)
    
    if not df_features.empty:
        visualize_training_performance(argenta_ml.model, df_features[argenta_ml.features], df_features[argenta_ml.target])
        visualize_feature_importance(argenta_ml.model, argenta_ml.features)

    fetcher = MultiSourceTideFetcher()

    parser = argparse.ArgumentParser(description="Argenta Navigation Safety Predictor")
    parser.add_argument("--date", help="Run a single prediction for a specific date and time (YYYY-MM-DD HH:MM)")
    args = parser.parse_args()

    if args.date:
        run_single_prediction(argenta_ml, fetcher, args.date)
    else:
        main_interactive_loop(argenta_ml, fetcher)