import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
import planetary_computer as pc
from pystac_client import Client
import warnings

# Suppress rasterio warnings about block sizes
warnings.filterwarnings("ignore")

# Re-use your physics logic
from src.reef_ml_predictor import calculate_physics_score
from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio

lat, lon = 37.05811, -8.20978
years = 10

catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
end_date = datetime.now()
start_date = datetime(end_date.year - years, 1, 1)

search = catalog.search(
    collections=['sentinel-2-l2a'],
    intersects={'type': 'Point', 'coordinates': [lon, lat]},
    datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
    query={"eo:cloud_cover": {"lt": 10}}
)

items = list(search.items())
data = []
for item in items:
    props = item.properties
    if props.get('s2:nodata_pixel_percentage', 100) > 20: continue
    rec = {
        'date_str': item.datetime.strftime('%Y-%m-%d'),
        'datetime': item.datetime,
        'cloud_cover': props.get('eo:cloud_cover', 100),
        'sun_elevation': props.get('view:sun_elevation', 45),
        'item': item
    }
    data.append(rec)
    
df = pd.DataFrame(data)
df['heuristic_score'] = df.apply(lambda r: calculate_physics_score(r, 16.0), axis=1)
df = df.sort_values('heuristic_score', ascending=False).drop_duplicates('date_str')
shortlist = df.head(30) # top 30 candidates

results = []
for idx, row in shortlist.iterrows():
    item = row['item']
    date_str = row['date_str']
    
    b02_href = item.assets["B02"].href
    b03_href = item.assets["B03"].href
    b08_href = item.assets["B08"].href
    
    BUFFER_M = 500.0
    env = rasterio.Env(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR')
    
    with env:
        try:
            with rasterio.open(b02_href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                window = from_bounds(x - BUFFER_M, y - BUFFER_M, x + BUFFER_M, y + BUFFER_M, src.transform)
                b02_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(b03_href) as src:
                b03_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(b08_href) as src:
                b08_arr = src.read(1, window=window).astype(np.float32)
        except Exception as e:
            continue
            
    b02_ref = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03_ref = np.clip(b03_arr / 10000.0, 0, 1.5)
    b08_ref = np.clip(b08_arr / 10000.0, 0, 1.5)
    
    if b02_ref.max() == 0 or np.all(np.isnan(b02_ref)):
        continue
        
    # Water-safe local cloud detection
    from src.enhancer import water_safe_cloud_mask
    cloud_pct = water_safe_cloud_mask(b02_ref, b08_ref) / 100.0
    if cloud_pct > 0.5:
        continue
        
    # Sunglint correction
    p95_b02 = np.percentile(b02_ref[b02_ref > 0], 95) if np.any(b02_ref > 0) else 0.0
    p95_b03 = np.percentile(b03_ref[b03_ref > 0], 95) if np.any(b03_ref > 0) else 0.0
    
    b02_corr = np.clip(b02_ref - 0.8 * p95_b02 * 0.05, 0, 1.0)
    b03_corr = np.clip(b03_ref - 0.8 * p95_b03 * 0.05, 0, 1.0)
    
    # Calculate low/high frequency ratio (FFT cleanliness)
    f_transform = np.fft.fft2(b02_corr)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    h, w = b02_corr.shape
    cy, cx = h // 2, w // 2
    
    # Low frequency mask (macro structures)
    r_low = 5
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx)**2 + (y - cy)**2 <= r_low**2
    # High frequency mask (noise, ripples, glint)
    r_high = 15
    mask_high = (x - cx)**2 + (y - cy)**2 >= r_high**2
    
    low_power = np.sum(power[mask_low])
    high_power = np.sum(power[mask_high])
    cleanliness = low_power / (high_power + 1e-12)
    
    # Edge complexity of geological contours (Entropy Gate)
    macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0
        
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, 0.045)
    snr_map = make_snr_map(b02_corr, window=5)
    snr_mean = float(np.nanmean(snr_map))
    
    # Measured local contrast (p99 / p1)
    p1 = np.percentile(b02_corr[b02_corr > 0], 1) if np.any(b02_corr > 0) else 0.0
    p99 = np.percentile(b02_corr[b02_corr > 0], 99) if np.any(b02_corr > 0) else 0.0
    contrast_ratio = p99 / p1 if p1 > 0 else 1.0
    
    results.append({
        'date': date_str,
        'cloud': row['cloud_cover'],
        'kd': kd_est,
        'snr': snr_mean,
        'contrast_ratio': contrast_ratio,
        'cleanliness': cleanliness,
        'edge_entropy': edge_entropy,
        'raw_mean': float(np.mean(b02_ref))
    })

df_res = pd.DataFrame(results)
# Rank by a composite formula:
# We want:
# 1. High Cleanliness (calm water): log10(cleanliness)
# 2. High Edge Entropy (structural information): edge_entropy
# 3. Clear water: 1 / kd
# 4. Correct water signal range: penalize if raw_mean is extremely high (sun glint/fog) or low.
# Composite score = log10(cleanliness) * edge_entropy * (0.045 / kd)

df_res['score'] = np.log10(df_res['cleanliness']) * df_res['edge_entropy'] * (0.045 / df_res['kd'])
# Penalize if cleanliness is less than 5000 (very wavy)
df_res.loc[df_res['cleanliness'] < 5000, 'score'] *= 0.1
# Penalize if raw_mean is not in [0.06, 0.15]
df_res.loc[(df_res['raw_mean'] < 0.06) | (df_res['raw_mean'] > 0.15), 'score'] *= 0.1

df_res = df_res.sort_values('score', ascending=False).reset_index(drop=True)

print("🏆 RANKING GEOFÍSICO COMPATÍVEL COM INSIGHTS DE ANALISTA SÉNIOR 🏆")
print("-" * 110)
for i, r in df_res.iterrows():
    print(f"{i+1:2}. {r['date']} | Score: {r['score']:.3f} | Cleanliness: {r['cleanliness']:8.1f} | EdgeEntropy: {r['edge_entropy']:.3f} | Kd: {r['kd']:.4f} | SNR: {r['snr']:.2f} | ContrastRatio: {r['contrast_ratio']:.3f} | RawMean: {r['raw_mean']:.3f}")
