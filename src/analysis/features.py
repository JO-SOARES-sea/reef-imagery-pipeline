"""
src/analysis/features.py — Extração de Features Morfométricas e Espectrais Submarinas.

Implementa:
  • pSDB (Pseudo-Satellite Derived Bathymetry): Índice espectral relativo Stumpf et al. (2003)
  • BPI (Bathymetric Position Index): Diferença topográfica entre o cume e o anel vizinho
  • VRM (Vector Ruggedness Measure - Sappington et al., 2007): Rugosidade tridimensional do relevo
  • Textura focal: Desvio-padrão espacial das bandas B02 e B03
  • Resamostragem e Depth-Screen do EMODnet (2 m a 25 m)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage as ndi

logger = logging.getLogger("reef_pipeline.analysis.features")

PX_M = 10.0  # Resolução espacial padrão do Sentinel-2 (bandas 10m)


# ── Pegadas Espaciais (Footprints) ─────────────────────────────────────────────
def square(h: int) -> np.ndarray:
    return np.ones((2 * h + 1, 2 * h + 1), bool)


def annulus(r_in: int, r_out: int) -> np.ndarray:
    y, x = np.ogrid[-r_out : r_out + 1, -r_out : r_out + 1]
    d2 = x * x + y * y
    return (d2 > r_in * r_in) & (d2 <= r_out * r_out)


# ── Estatísticas Focais Tolerantes a NaN ──────────────────────────────────────
def _focal_sum(a: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(a)
    w = fp.astype(float)
    s = ndi.convolve(np.where(valid, a, 0.0), w, mode="constant", cval=0.0)
    n = ndi.convolve(valid.astype(float), w, mode="constant", cval=0.0)
    return s, n


def focal_mean(a: np.ndarray, fp: np.ndarray, min_frac: float = 0.5) -> np.ndarray:
    s, n = _focal_sum(a, fp)
    out = np.where(n >= min_frac * fp.sum(), s / np.maximum(n, 1), np.nan)
    return out.astype(np.float32)


def focal_std(a: np.ndarray, fp: np.ndarray, min_frac: float = 0.5) -> np.ndarray:
    s, n = _focal_sum(a, fp)
    s2, _ = _focal_sum(a * a, fp)
    m = s / np.maximum(n, 1)
    v = np.maximum(s2 / np.maximum(n, 1) - m * m, 0)
    out = np.where(n >= min_frac * fp.sum(), np.sqrt(v), np.nan)
    return out.astype(np.float32)


def _sobel(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = ndi.sobel(np.nan_to_num(a, nan=0.0), axis=1, mode="nearest") / (8 * PX_M)
    gy = ndi.sobel(np.nan_to_num(a, nan=0.0), axis=0, mode="nearest") / (8 * PX_M)
    return gx, gy


# ── Funções de Extração de Features ──────────────────────────────────────────
def compute_psdb(blue: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    Calcula a batimetria espectral relativa não calibrada (Stumpf et al., 2003):
        pSDB = ln(1000 * B02) / ln(1000 * B03)
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.log(np.maximum(1000.0 * blue, 1e-4))
        den = np.log(np.maximum(1000.0 * green, 1e-4))
        out = num / den

    invalid = ~np.isfinite(out) | (blue <= 0.001) | (green <= 0.001) | ~np.isfinite(blue) | ~np.isfinite(green)
    return np.where(invalid, np.nan, out).astype(np.float32)


def compute_slope(depth_proxy: np.ndarray) -> np.ndarray:
    """Calcula a magnitude do declive do relevo submarino a partir do gradiente de Sobel."""
    gx, gy = _sobel(depth_proxy)
    slope = np.hypot(gx, gy).astype(np.float32)
    return np.where(np.isfinite(depth_proxy), slope, np.nan)


def compute_bpi(depth_proxy: np.ndarray, r_in: int = 5, r_out: int = 10) -> np.ndarray:
    """
    Calcula o Bathymetric Position Index (BPI).
    r_in=5 (50 m) e r_out=10 (100 m) identifica cumes rochosos sobrelevados.
    """
    ann = annulus(r_in, r_out)
    bpi = focal_mean(depth_proxy, ann) - depth_proxy
    return np.where(np.isfinite(depth_proxy), bpi, np.nan).astype(np.float32)


def compute_vrm(depth_proxy: np.ndarray, h: int = 2) -> np.ndarray:
    """
    Calcula o Vector Ruggedness Measure (VRM) de Sappington et al. (2007) numa janela (2h+1)x(2h+1).
    Utiliza float64 para evitar cancelamento catastrófico em pequenos gradientes batimétricos.
    h=2 -> Janela 5x5 pixels (50 m).
    """
    gx, gy = _sobel(depth_proxy)
    gx64, gy64 = gx.astype(np.float64), gy.astype(np.float64)
    slope = np.hypot(gx64, gy64)
    aspect = np.arctan2(gy64, gx64)

    sin_s = np.sin(np.arctan(slope))
    cos_s = np.cos(np.arctan(slope))
    nx = sin_s * np.cos(aspect)
    ny = sin_s * np.sin(aspect)
    nz = cos_s

    fp = square(h)
    mx = focal_mean(nx, fp).astype(np.float64)
    my = focal_mean(ny, fp).astype(np.float64)
    mz = focal_mean(nz, fp).astype(np.float64)

    r = np.sqrt(mx * mx + my * my + mz * mz)
    vrm = np.maximum(0.0, 1.0 - r)
    return np.where(np.isfinite(depth_proxy), vrm, np.nan).astype(np.float32)


def calibrate_sdb_depth(
    psdb: np.ndarray,
    depth_emodnet: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calibra o índice relativo pSDB para profundidade física estimada em metros (Stumpf et al., 2003).
    Se depth_emodnet estiver disponível, ajusta linearmente z = m1 * psdb + m0 na faixa de interesse.
    """
    if depth_emodnet is not None and water_mask is not None:
        valid = water_mask & np.isfinite(psdb) & np.isfinite(depth_emodnet) & (depth_emodnet > 3.0) & (depth_emodnet < 22.0)
        if np.sum(valid) > 500:
            from scipy import stats
            slope_fit, intercept_fit, _, _, _ = stats.linregress(psdb[valid], depth_emodnet[valid])
            if slope_fit > 0:
                sdb_m = slope_fit * psdb + intercept_fit
                return np.clip(sdb_m, 1.0, 35.0).astype(np.float32)
    # Calibração empírica padrão da costa do Algarve (águas límpidas)
    sdb_m = 22.0 * psdb - 17.0
    return np.clip(sdb_m, 1.0, 35.0).astype(np.float32)


def compute_curv(depth_proxy: np.ndarray) -> np.ndarray:
    """
    Calcula a curvatura do relevo (Laplaciano) sobre o proxy batimétrico psdb
    suavizado numa janela focal 11x11 (110 m), identificando cumes convexos de recifes.
    """
    sm = focal_mean(depth_proxy, square(5))
    cross_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    curv = ndi.convolve(np.nan_to_num(sm, nan=0.0), cross_kernel, mode="nearest") / (PX_M ** 2)
    return np.where(np.isfinite(depth_proxy), curv, np.nan).astype(np.float32)


def compute_focal_std(band: np.ndarray, h: int = 1) -> np.ndarray:
    """Calcula a rugosidade espectral local (desvio padrão focal). h=1 (30m), h=2 (50m)."""
    return focal_std(band, square(h))


def resample_emodnet_screen(
    emodnet_path: str | Path,
    target_shape: tuple[int, int],
    target_crs: Any,
    target_transform: Any,
    min_depth: float = 2.0,
    max_depth: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resamostra o DTM batimétrico do EMODnet para a grelha 10 m da cena Sentinel-2
    e gera a máscara de profundidade de interesse (2 m a 25 m ZH).

    IMPORTANTE (Regra de Arquitetura): O DTM nunca é utilizado como feature preditiva,
    mas estritamente como máscara de corte espacial para excluir zonas entre-marés e abissais.

    Returns:
        tuple[depth_10m, depth_mask]:
            depth_10m: Matriz de profundidade (positiva em metros abaixo do nível do mar).
            depth_mask: Booleano True onde a profundidade está entre min_depth e max_depth.
    """
    logger.info(f"Resamostrando batimetria EMODnet de: {emodnet_path}")
    destination = np.full(target_shape, np.nan, dtype=np.float32)

    with rasterio.open(emodnet_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    # No EMODnet, valores negativos representam profundidade submarina (ex: -14m = 14m abaixo do mar)
    depth_m = np.abs(np.where(destination < 0, destination, np.nan))
    depth_mask = (depth_m >= min_depth) & (depth_m <= max_depth) & np.isfinite(depth_m)

    total_valid = int(np.sum(depth_mask))
    logger.info(
        f"Máscara de profundidade EMODnet [{min_depth}m, {max_depth}m] criada: "
        f"{total_valid} pixels de interesse."
    )
    return depth_m, depth_mask


def extract_morpho_spectral_features(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray | None = None,
    water_mask: np.ndarray | None = None,
    depth_emodnet: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Extrai o conjunto estendido de features morfo-espectrais submarinas,
    incorporando a arquitetura M5_relative, calibração física SDB e VRM 3D em float64.
    """
    logger.info("Extraindo features morfo-espectrais avançadas do Sentinel-2...")

    # Aplica máscara de água antes do cálculo para anular terra firme
    b2 = np.where(water_mask, blue, np.nan) if water_mask is not None else blue
    b3 = np.where(water_mask, green, np.nan) if water_mask is not None else green
    b4 = np.where(water_mask, red, np.nan) if water_mask is not None else red
    b8 = np.where(water_mask, nir, np.nan) if (water_mask is not None and nir is not None) else nir

    # 1. Morfometria baseada em pSDB (Stumpf et al., 2003) e SDB calibrada em metros
    psdb = compute_psdb(b2, b3)
    sdb_m = calibrate_sdb_depth(psdb, depth_emodnet=depth_emodnet, water_mask=water_mask)

    # Métricas derivadas em metros (evitam desvanecimento por gradientes infinitesimais)
    sdb_slope = compute_slope(sdb_m)
    sdb_bpi100 = compute_bpi(sdb_m, r_in=5, r_out=10)   # 50 m a 100 m
    sdb_bpi200 = compute_bpi(sdb_m, r_in=10, r_out=20) # 100 m a 200 m
    sdb_vrm50 = compute_vrm(sdb_m, h=2)                # Rugosidade 3D (50 m) em float64

    # Morfometria direta sobre psdb
    slope = compute_slope(psdb)
    bpi100 = compute_bpi(psdb, r_in=5, r_out=10)
    bpi200 = compute_bpi(psdb, r_in=10, r_out=20)
    vrm50 = compute_vrm(psdb, h=2)
    curv100 = compute_curv(psdb)

    # 2. Texturas focais (rugosidade espectral do fundo rochoso)
    b02_std30 = compute_focal_std(b2, h=1)
    b03_std30 = compute_focal_std(b3, h=1)
    b02_std50 = compute_focal_std(b2, h=2)
    b03_std50 = compute_focal_std(b3, h=2)

    # Contraste local de alta frequência (anomalia em relação ao fundo suave vizinho)
    mean_val = float(np.nanmean(b3[np.isfinite(b3)])) if np.any(np.isfinite(b3)) else 0.03
    b3_sm = ndi.gaussian_filter(np.nan_to_num(b3, nan=mean_val), sigma=5)
    b3_anom = np.where(np.isfinite(b3), b3 - b3_sm, np.nan).astype(np.float32)

    # 3. Rácios espectrais fundamentais
    with np.errstate(invalid="ignore", divide="ignore"):
        r_b03_b02 = np.where((b2 > 0.001) & np.isfinite(b2), b3 / b2, np.nan)
        r_b04_b03 = np.where((b3 > 0.001) & np.isfinite(b3), b4 / b3, np.nan)
        gr_ratio = np.where((b4 > 0.005) & np.isfinite(b4), b3 / b4, np.nan)
        r_b08_b04 = np.where((b4 > 0.001) & np.isfinite(b4) & (b8 is not None), (b8 if b8 is not None else 0.0) / b4, np.nan)

    # 4. Z-Score Relative Features (M5_relative arm de UnderwaterVisibility)
    def _zscore_water(arr: np.ndarray) -> np.ndarray:
        if water_mask is None:
            valid = arr[np.isfinite(arr)]
        else:
            valid = arr[water_mask & np.isfinite(arr)]
        if len(valid) == 0:
            return np.zeros_like(arr, dtype=np.float32)
        mu = float(np.nanmean(valid))
        sigma = float(np.nanstd(valid))
        if sigma < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        z = (arr - mu) / sigma
        return np.where(np.isfinite(arr), z, np.nan).astype(np.float32)

    z_b02_b03 = _zscore_water(psdb)
    z_b03_b04 = _zscore_water(r_b04_b03)

    features = {
        "B02": b2,
        "B03": b3,
        "B04": b4,
        "psdb": psdb,
        "sdb_m": sdb_m,
        "sdb_slope": sdb_slope,
        "sdb_bpi100": sdb_bpi100,
        "sdb_bpi200": sdb_bpi200,
        "sdb_vrm50": sdb_vrm50,
        "psdb_slope": slope,
        "psdb_bpi100": bpi100,
        "psdb_bpi200": bpi200,
        "psdb_vrm50": vrm50,
        "psdb_curv100": curv100,
        "B02_std30": b02_std30,
        "B03_std30": b03_std30,
        "B02_std50": b02_std50,
        "B03_std50": b03_std50,
        "b3_anom": b3_anom,
        "r_b03_b02": r_b03_b02,
        "r_b04_b03": r_b04_b03,
        "gr_ratio": gr_ratio,
        "z_b02_b03": z_b02_b03,
        "z_b03_b04": z_b03_b04,
    }
    if b8 is not None:
        features["B08"] = b8
        features["r_b08_b04"] = r_b08_b04

    logger.info(f"{len(features)} features morfo-espectrais extraídas com sucesso.")
    return features
