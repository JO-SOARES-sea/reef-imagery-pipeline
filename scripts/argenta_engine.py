import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import re
import math
import time
from datetime import datetime, timedelta

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_FIGURES_DIR = _PROJECT_ROOT / "docs" / "figures"
from sklearn.ensemble import RandomForestRegressor
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ArgentaEngine")
logger.setLevel(logging.DEBUG) # Definir nível de logging para DEBUG para ver URLs

def get_moon_phase_strength(date_obj: datetime) -> float:
    """
    Calcula a força do coeficiente de maré via aproximação da fase lunar (0.0 a 1.0).
    1.0 = Maré Viva (Lua Nova/Cheia), 0.0 = Maré Morta (Quartos).
    """
    lunar_days = 29.53058770576
    ref_new_moon = datetime(2000, 1, 6, 18, 14)
    days_since = (date_obj - ref_new_moon).total_seconds() / 86400.0
    phase = (days_since / lunar_days) % 1
    # A força atinge pico na lua nova (0, 1) e lua cheia (0.5)
    return abs(math.cos(phase * 2 * math.pi))

def get_moon_details(date_obj: datetime) -> str:
    """
    Calcula a fase da lua (percentagem e estado) para uma data específica.
    Exemplo: "89% waning"
    """
    lunar_days = 29.53058770576
    ref_new_moon = datetime(2000, 1, 6, 18, 14)
    days_since = (date_obj - ref_new_moon).total_seconds() / 86400.0
    phase_index = (days_since / lunar_days) % 1.0

    # Illumination percentage
    illumination_pct = (1 - math.cos(phase_index * 2 * math.pi)) / 2 * 100

    # State (waxing/waning)
    if 0.01 < phase_index < 0.49:
        state = "waxing"
    elif 0.51 < phase_index < 0.99:
        state = "waning"
    else:
        state = "" # Full or New moon

    return f"{illumination_pct:.0f}% {state}".strip()

def calcular_altura_barra(target_dt: datetime, df_tides_dia: pd.DataFrame) -> dict:
    """
    Calcula a altura da maré na Barra para um minuto exato usando interpolação por cosseno.
    """
    if df_tides_dia is None or df_tides_dia.empty:
        raise ValueError(f"Dados insuficientes (DataFrame vazio) para interpolar a data {target_dt}.")
        
    if 'Datetime' not in df_tides_dia.columns:
        df_tides_dia['Datetime'] = pd.to_datetime(df_tides_dia['Data'] + ' ' + df_tides_dia['Hora'], format='%Y-%m-%d %H:%M')
        
    df_tides_dia = df_tides_dia.sort_values('Datetime').reset_index(drop=True)
    
    # Encontrar maré anterior e seguinte
    antes = df_tides_dia[df_tides_dia['Datetime'] <= target_dt]
    depois = df_tides_dia[df_tides_dia['Datetime'] > target_dt]
    
    if antes.empty or depois.empty:
        raise ValueError(f"Dados de maré insuficientes para interpolar a data {target_dt}. É necessário extrair a maré anterior e seguinte.")
        
    p1, p2 = antes.iloc[-1], depois.iloc[0]
    
    t0, t1 = p1['Datetime'], p2['Datetime']
    h0, h1 = p1['App_Barra_m'], p2['App_Barra_m']
    
    total_seconds = (t1 - t0).total_seconds()
    if total_seconds == 0: # Evita divisão por zero se os timestamps forem iguais
        return {
            'App_Barra_m_interpolada': round(h0, 3),
            'Ciclo': 'Estacionária',
            'tempo_ate_pico_min': 0.0
        }

    elapsed_seconds = (target_dt - t0).total_seconds()
    
    # Interpolação matemática da curva da maré (Regra dos Duodécimos contínua)
    fase = math.pi * (elapsed_seconds / total_seconds)
    h_target = h0 + (h1 - h0) * ((1 - math.cos(fase)) / 2)
    
    # Determinar tempo até ao pico mais próximo
    tempo_ate_pico = min(abs((target_dt - t0).total_seconds()), abs((target_dt - t1).total_seconds())) / 60
    
    return {
        'App_Barra_m_interpolada': round(h_target, 3),
        'Ciclo': 'A encher' if h1 > h0 else 'A vazar',
        'tempo_ate_pico_min': round(tempo_ate_pico, 1)
    }

class MultiSourceTideFetcher:
    """
    Módulo de integração para consumir dados de marés de múltiplas fontes.
    1. Instituto Hidrográfico (Scraping Primário)
    2. Open-Meteo Marine API (Fallback Científico e Global sem chaves)
    Inclui CACHE em CSV para evitar bloqueios em iterações massivas.
    """
    def __init__(self, cache_file=None, master_schedule_file=None):
        self.scrape_url_ih = "https://www.hidrografico.pt/m.mare"
        self.cache_file = str(cache_file) if cache_file is not None else str(_DATA_DIR / "tides_cache.csv")
        self.master_schedule_file = (
            str(master_schedule_file)
            if master_schedule_file is not None
            else str(_DATA_DIR / "tide_schedule_master.csv")
        )
        
        if os.path.exists(self.cache_file):
            self.cache = pd.read_csv(self.cache_file)
        else:
            self.cache = pd.DataFrame(columns=['Data', 'Hora', 'App_Barra_m', 'Fase_Lua_Pct', 'Ciclo', 'Tipo', 'source'])

        if os.path.exists(self.master_schedule_file):
            self.master_schedule = pd.read_csv(self.master_schedule_file)
        else:
            self.master_schedule = None

    def fetch_historical_tides(self, target_date: str) -> pd.DataFrame:
        # 0. Check Master Schedule (highest priority)
        if self.master_schedule is not None:
            df_master = self._read_from_master(target_date)
            if not df_master.empty:
                logger.info(f"Dados para {target_date} carregados do ficheiro mestre local.")
                return df_master

        # 1. Check Cache
        if not self.cache.empty and target_date in self.cache['Data'].values:
            cached_data = self.cache[self.cache['Data'] == target_date].copy()
            source = cached_data['source'].iloc[0] if 'source' in cached_data.columns and not cached_data['source'].empty else 'desconhecida'
            logger.info(f"Dados para {target_date} carregados da cache (fonte original: {source}).")
            return cached_data
            
        # 2. Try web sources in order
        sources = [
            ("Instituto Hidrográfico", self._scrape_ih),
            ("Open-Meteo API", self._scrape_alternativa),
            ("Tide-Forecast.com", self._scrape_tide_forecast),
        ]

        for name, func in sources:
            df_tides = func(target_date)
            if not df_tides.empty:
                logger.info(f"Dados para {target_date} obtidos com sucesso de: {name}")
                df_tides['source'] = name
                # Update cache and return
                self.cache = pd.concat([self.cache, df_tides], ignore_index=True)
                self.cache.to_csv(self.cache_file, index=False)
                return df_tides
            
        logger.error(f"Não foi possível obter dados de maré para {target_date} de nenhuma fonte.")
        return pd.DataFrame()

    def _read_from_master(self, target_date: str) -> pd.DataFrame:
        if self.master_schedule is None or self.master_schedule.empty:
            return pd.DataFrame()
        
        day_data = self.master_schedule[self.master_schedule['date'] == target_date].copy()
        if day_data.empty:
            return pd.DataFrame()

        tides_data = []
        for _, row in day_data.iterrows():
            tipo = "Preia-mar" if row['tide_type'] == 'high' else "Baixa-mar"
            ciclo = 'A encher' if row['tide_type'] == 'high' else 'A vazar'
            dt_obs = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M")
            
            tides_data.append({
                'Data': row['date'],
                'Hora': row['time'],
                'App_Barra_m': row['height_m'],
                'Fase_Lua_Pct': round(get_moon_phase_strength(dt_obs), 3),
                'Ciclo': ciclo,
                'Tipo': tipo,
                'source': 'master_schedule.csv'
            })
        return pd.DataFrame(tides_data)

    def _scrape_ih(self, target_date: str) -> pd.DataFrame:
        logger.info(f"A descarregar marés (IH) para a data {target_date}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(f"{self.scrape_url_ih}?data={target_date}", headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            tides_data = []
            for table in soup.find_all('table'):
                if 'Preia-mar' in table.text or 'Baixa-mar' in table.text:
                    for row in table.find_all('tr'):
                        cols = [ele.text.strip() for ele in row.find_all(['td', 'th'])]
                        if len(cols) >= 3 and ('Preia-mar' in cols[0] or 'Baixa-mar' in cols[0]):
                            tipo, hora = cols[0], cols[1]
                            try:
                                altura = float(cols[2].replace(',', '.'))
                            except ValueError:
                                continue
                            ciclo = 'A encher' if 'Preia-mar' in tipo else 'A vazar'
                            dt_obs = datetime.strptime(f"{target_date} {hora}", "%Y-%m-%d %H:%M")
                            
                            tides_data.append({
                                'Data': target_date, 'Hora': hora, 'App_Barra_m': altura,
                                'Fase_Lua_Pct': round(get_moon_phase_strength(dt_obs), 3),
                                'Ciclo': ciclo, 'Tipo': tipo
                            })
            return pd.DataFrame(tides_data)
        except Exception as e:
            logger.warning(f"O IH falhou/bloqueou para {target_date}: {e}")
            return pd.DataFrame()

    def _scrape_alternativa(self, target_date: str) -> pd.DataFrame:
        logger.info(f"A usar fonte alternativa (Open-Meteo Marine API) para {target_date}...")
        try:
            lat, lon = 36.98, -7.88 # Coordenadas da Barra de Faro/Olhão
            url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=sea_level&start_date={target_date}&end_date={target_date}"
            logger.debug(f"Open-Meteo URL: {url}")
            
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            df_h = pd.DataFrame({
                'Datetime': pd.to_datetime(data['hourly']['time']),
                'Height': data['hourly']['sea_level']
            })
            df_h.dropna(subset=['Height'], inplace=True)
            df_h.reset_index(drop=True, inplace=True)
            
            df_h['Height'] += 2.0 # Ajuste do Nível Médio do Mar para Zero Hidrográfico local (Faro ~ +2.0m)
            
            tides_data = []
            heights = df_h['Height'].values
            for i in range(1, len(heights)-1):
                h_prev, h_curr, h_next = heights[i-1], heights[i], heights[i+1]
                dt_obs = df_h['Datetime'].iloc[i]
                fase_lua = round(get_moon_phase_strength(dt_obs), 2)
                hora_str = dt_obs.strftime('%H:%M')
                
                if h_curr > h_prev and h_curr > h_next: # Pico de Preia-mar
                    tides_data.append({'Data': target_date, 'Hora': hora_str, 'App_Barra_m': round(h_curr, 2), 'Fase_Lua_Pct': fase_lua, 'Ciclo': 'A encher', 'Tipo': 'Preia-mar'})
                elif h_curr < h_prev and h_curr < h_next: # Pico de Baixa-mar
                    tides_data.append({'Data': target_date, 'Hora': hora_str, 'App_Barra_m': round(h_curr, 2), 'Fase_Lua_Pct': fase_lua, 'Ciclo': 'A vazar', 'Tipo': 'Baixa-mar'})
            return pd.DataFrame(tides_data)
        except Exception as e:
            logger.error(f"A API de fallback Open-Meteo falhou: {e}")
            return pd.DataFrame()

    def _scrape_tide_forecast(self, target_date: str) -> pd.DataFrame:
        """Scrapes Tide-Forecast.com as a second fallback."""
        logger.info(f"A usar segunda fonte alternativa (Tide-Forecast.com) para {target_date}...")
        try:
            # The URL is for the location; the page contains multiple days.
            url = "https://www.tide-forecast.com/locations/Faro-Portugal/tides/latest"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            tides_data = []
            
            # Format for matching: e.g., "Sunday 09 June"
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            date_str_match = date_obj.strftime("%A %d %B")

            day_sections = soup.find_all('div', class_='tide-day')

            for section in day_sections:
                header = section.find('h4', class_='tide-day__date')
                if header and date_str_match in header.text:
                    table = section.find('table', class_='tide-table')
                    if not table:
                        continue
                    
                    rows = table.find('tbody').find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) < 3:
                            continue
                        
                        tide_type_text = cols[0].text.strip()
                        time_text = cols[1].text.strip() # e.g., "5:57 AM"
                        height_text = cols[2].text.strip() # e.g., "2.97 m"

                        if "High Tide" in tide_type_text:
                            tipo, ciclo = "Preia-mar", "A encher"
                        elif "Low Tide" in tide_type_text:
                            tipo, ciclo = "Baixa-mar", "A vazar"
                        else:
                            continue

                        hora = datetime.strptime(time_text, "%I:%M %p").strftime("%H:%M")
                        altura = float(height_text.split()[0])
                        dt_obs = datetime.strptime(f"{target_date} {hora}", "%Y-%m-%d %H:%M")
                        
                        tides_data.append({
                            'Data': target_date, 'Hora': hora, 'App_Barra_m': altura,
                            'Fase_Lua_Pct': round(get_moon_phase_strength(dt_obs), 3),
                            'Ciclo': ciclo, 'Tipo': tipo
                        })
            
            return pd.DataFrame(tides_data)
        except Exception as e:
            logger.error(f"O fallback Tide-Forecast.com falhou: {e}")
            return pd.DataFrame()

    def fetch_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end - start
        all_dfs = []
        for i in range(delta.days + 1):
            current_date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            df_day = self.fetch_historical_tides(current_date)
            if not df_day.empty:
                all_dfs.append(df_day)
            time.sleep(0.2)
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

class FeatureEngineer:
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("A executar feature engineering...")
        df = df.copy()
        
        if 'Hora_Observacao' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora_Observacao'], format='%d/%m/%Y %H:%M', errors='coerce')
        elif 'Hora' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], format='%Y-%m-%d %H:%M', errors='coerce')
            
        df['Mes'] = df['Datetime'].dt.month

        # 1. Parse Fase Lunar (e.g., "94% Crescente" -> 0.94)
        if 'Fase_Lua' in df.columns and df['Fase_Lua'].dtype == 'object':
            df['Fase_Lua_Pct'] = df['Fase_Lua'].str.extract(r'(\d+\.?\d*)%').astype(float) / 100.0
            df['Fase_Lua_Pct'].fillna(0.5, inplace=True)
        else:
            df['Fase_Lua_Pct'] = df.get('Fase_Lua', 0.5)

        # 2. Calculate Tidal Amplitude from string columns
        def parse_tide_height(tide_str):
            if pd.isna(tide_str):
                return np.nan
            match = re.search(r'\((\d+\.?\d*)\s*m\)', str(tide_str))
            return float(match.group(1)) if match else np.nan

        if 'Baixa_mar_str' in df.columns and 'Preia_mar_str' in df.columns:
            low_tides = df['Baixa_mar_str'].apply(parse_tide_height)
            high_tides = df['Preia_mar_str'].apply(parse_tide_height)
            df['amplitude_mare'] = high_tides - low_tides
        else:
            df['amplitude_mare'] = np.nan

        if df['amplitude_mare'].isnull().any():
            mean_amplitude = df['amplitude_mare'].mean() if not df['amplitude_mare'].isnull().all() else 2.0
            df['amplitude_mare'].fillna(mean_amplitude, inplace=True)
            logger.warning(f"Missing tidal amplitude values were filled with the mean: {mean_amplitude:.2f}m")
            
        df['Ciclo_Encoded'] = df['Ciclo'].apply(lambda x: 1 if str(x).lower().strip() == 'a encher' else -1)
        df['Hydro_Pressure_Index'] = df['App_Barra_m'] * df['Fase_Lua_Pct']
        
        logger.info(f"Features de treino geradas: {len(df)} registos.")
        return df

class ArgentaSafetyModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.features = ['App_Barra_m', 'Fase_Lua_Pct', 'Ciclo_Encoded', 'Hydro_Pressure_Index', 'Mes', 'amplitude_mare', 'tempo_ate_pico_min']
        self.target = 'Offset_Medido'
        self.BRIDGE_LIMIT_M = 2.90
        self.safety_margin = 0.15 
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        logger.info("A preparar treino do modelo Preditivo...")
        if df.empty:
            logger.error("Dataset vazio. Treino cancelado.")
            return
            
        X = df[self.features]
        y = df[self.target]
        
        self.model.fit(X, y)
        
        train_preds = self.model.predict(X)
        mse = mean_squared_error(y, train_preds)
        logger.info(f"Modelo treinado com sucesso! (Train MSE: {mse:.4f})")
        
        max_underprediction = np.max(y - train_preds)
        if max_underprediction > 0:
            self.safety_margin = max(self.safety_margin, max_underprediction + 0.05)
            logger.warning(f"Sub-predição detectada. Margem de segurança elevada para {self.safety_margin:.2f}m")
            
        self.is_trained = True

    def prever_offset_doca(self, features_input: dict) -> dict:
        if not self.is_trained:
            raise ValueError("O modelo deve ser treinado antes da inferência.")
        
        X_input = pd.DataFrame([features_input])
        X_input = X_input[self.features]

        predicted_offset = self.model.predict(X_input)[0]
        
        altura_barra = features_input['App_Barra_m']
        predicted_doca_height = altura_barra + predicted_offset
        safe_predicted_doca_height = predicted_doca_height + self.safety_margin
        
        is_safe = safe_predicted_doca_height <= self.BRIDGE_LIMIT_M
        
        return {
            "app_barra_m": altura_barra,
            "offset_ml_previsto": round(predicted_offset, 3),
            "altura_doca_prevista_pura": round(predicted_doca_height, 3),
            "altura_doca_com_viés_segurança": round(safe_predicted_doca_height, 3),
            "limite_ponte_m": self.BRIDGE_LIMIT_M,
            "DECISAO_SEGURANCA": "✅ LIVRE - SEGURO PASSAR" if is_safe else "❌ PERIGO - NÃO AVANÇAR"
        }

def get_ground_truth_dataset(csv_path=None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = str(_DATA_DIR / "ground_truth.csv")
    if os.path.exists(csv_path):
        logger.info(f"A carregar dados de treino do ficheiro: {csv_path}")
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Erro ao ler o ficheiro CSV '{csv_path}': {e}")
            logger.warning("A usar o dataset base como fallback.")

    logger.warning(f"Ficheiro de treino '{csv_path}' não encontrado. A usar o dataset base.")
    data = {
        "Data": ["17/08/2024", "21/06/2025", "15/08/2023", "02/07/2023"],
        "Hora_Observacao": ["12:15", "10:55", "13:05", "12:51"],
        "App_Barra_m": [2.40, 2.43, 2.21, 2.40],
        "Regua_Doca_m": [2.80, 3.00, 2.50, 2.30],
        "Offset_Medido": [0.40, 0.57, 0.29, -0.10],
        "Fase_Lua": ["94%", "17%", "0%", "99%"],
        "Ciclo": ["A encher", "A encher", "A encher", "A encher"]
    }
    return pd.DataFrame(data)

def visualize_training_performance(model, X_train, y_train):
    logger.info("A gerar visualização da performance no conjunto de treino...")
    y_train_vals = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train_vals, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Análise de Performance do Modelo (Ajuste aos Dados de Treino)", fontsize=16)

    ax1.scatter(y_train_vals, y_pred, alpha=0.8, edgecolors='k', s=80, label="Pontos de Dados")
    lims = [min(np.min(y_train_vals), np.min(y_pred)) - 0.1, max(np.max(y_train_vals), np.max(y_pred)) + 0.1]
    ax1.plot(lims, lims, 'r--', lw=2, label='Previsão Perfeita')
    ax1.set_xlabel("Offset Real Medido (m)")
    ax1.set_ylabel("Offset Previsto pelo Modelo (m)")
    ax1.set_title(f"Predito vs. Real (R²: {r2:.3f})")
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')

    residuals = y_train_vals - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.8, edgecolors='k', s=80)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel("Offset Previsto (m)")
    ax2.set_ylabel("Resíduo (Erro: Real - Previsto)")
    ax2.set_title("Análise de Resíduos")
    ax2.grid(True)

    for i in range(len(y_pred)):
        ax1.annotate(f"  Ponto {i+1}", (y_train_vals[i], y_pred[i]))
        ax2.annotate(f"  Ponto {i+1}", (y_pred[i], residuals[i]))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = str(_FIGURES_DIR / "model_training_fit.png")
    plt.savefig(plot_path)
    logger.info(f"Gráfico de análise de performance guardado em: {plot_path}")
    logger.warning("NOTA: Esta visualização mostra o ajuste aos dados de TREINO. Com mais dados, um conjunto de TESTE separado é crucial para uma avaliação real.")
    plt.close(fig)

def visualize_feature_importance(model, features, plot_path=None):
    if plot_path is None:
        plot_path = str(_FIGURES_DIR / "feature_importance.png")
    logger.info("A gerar visualização da importância das features...")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind='barh', ax=ax, color='skyblue')
    
    ax.set_title("Importância das Features para o Modelo de Offset da Doca")
    ax.set_xlabel("Importância (Gini)")
    ax.set_ylabel("Feature")
    
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Gráfico de importância das features guardado em: {plot_path}")
    plt.close(fig)