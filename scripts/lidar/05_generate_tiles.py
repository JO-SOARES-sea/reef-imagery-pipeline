#!/usr/bin/env python3
"""
05_generate_tiles.py — Tiling do mosaico MDT em patches 256×256 com overlap.

Input:  {output_dir}/mosaico_algarve.tif  (EPSG:3763, produzido por 04_merge_mdt.py)
Output: {output_dir}/tiles/tile_{row:04d}_{col:04d}.tif  (GeoTIFFs individuais)
        {output_dir}/tiles_index.json                     (manifesto para reassembly)

Configuração:
    tile_size    = 256 px  (input canónico do ReefUNet)
    tile_overlap = 32  px  (evita artefactos nas bordas em inferência)
    stride       = tile_size - tile_overlap = 224 px

Os tiles com >80% de nodata são descartados (não há MDT válido nessa área).
Os tiles válidos são normalizados para [0, 1] antes de serem guardados.

Corre no DGT JupyterHub (96 cores, 503 GB RAM, CPU-only).
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('reef_pipeline.log', mode='a')],
)
logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.error("rasterio não instalado — pip install rasterio")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = lambda x, **kw: x  # type: ignore[assignment]
    HAS_TQDM = False


# ── Configuração ──────────────────────────────────────────────────────────────

@dataclass
class TileConfig:
    output_dir:   Path = Path('/home/jovyan/reef_output')
    mosaic_name:  str  = 'mosaico_algarve.tif'
    tile_size:    int  = 256
    tile_overlap: int  = 32
    nodata_frac:  float = 0.80   # descartar se >80% de pixels sem dados
    crs_work:     str  = 'EPSG:3763'

    @property
    def mosaic_path(self) -> Path:
        return self.output_dir / self.mosaic_name

    @property
    def tiles_dir(self) -> Path:
        return self.output_dir / 'tiles'

    @property
    def index_path(self) -> Path:
        return self.output_dir / 'tiles_index.json'

    @property
    def stride(self) -> int:
        return self.tile_size - self.tile_overlap


# ── Funções auxiliares ────────────────────────────────────────────────────────

# DGT MDT nodata sentinel value (rasterio nodata=None mas o fill é -9999)
_NODATA_SENTINEL = -9990.0
# Limite de profundidade — deve coincidir com SDB_OPTICAL_LIMIT_M em src/constants.py
# Mapeia [-DEPTH_LIMIT, 0] → [0, 1]; terra positiva → clipa a 1; nodata → 0
_DEPTH_LIMIT_M = 40.0


def _normalize(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """
    Normaliza MDT costeiro para [0, 1] usando a mesma janela de profundidade
    do training dataset (src/constants.py SDB_OPTICAL_LIMIT_M).

    Trata -9999.0 (sentinel DGT) como nodata → 0.
    Fórmula: clip((depth + DEPTH_LIMIT) / DEPTH_LIMIT, 0, 1)
      -40m → 0.0 | -20m → 0.5 | 0m (superfície) → 1.0 | terra > 0 → 1.0
    """
    arr = arr.astype(np.float32)
    # Máscara de nodata: sentinel DGT (-9999), nodata explícito, NaN
    mask = arr < _NODATA_SENTINEL
    if nodata is not None:
        mask |= (arr == nodata)
    mask |= np.isnan(arr)
    arr[mask] = np.nan
    norm = np.clip((arr + _DEPTH_LIMIT_M) / _DEPTH_LIMIT_M, 0.0, 1.0)
    return np.nan_to_num(norm, nan=0.0).astype(np.float32)


def generate_tiles(cfg: TileConfig) -> int:
    """
    Tila o mosaico MDT em patches 256×256 com overlap de 32 px.

    Guarda cada tile como GeoTIFF individual e um índice JSON para reassembly.

    Returns:
        Número de tiles válidos gerados.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é obrigatório para esta etapa")

    mosaic_path = cfg.mosaic_path
    if not mosaic_path.exists():
        raise FileNotFoundError(
            f"Mosaico não encontrado: {mosaic_path}\n"
            f"Corra primeiro: python scripts/lidar/04_merge_mdt.py"
        )

    cfg.tiles_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(mosaic_path)) as src:
        width    = src.width
        height   = src.height
        profile  = src.profile.copy()
        nodata   = src.nodata
        transform = src.transform
        crs       = src.crs
        logger.info("Mosaico: %d×%d px  CRS=%s  nodata=%s",
                    width, height, crs, nodata)

    tile_index: list[dict] = []
    skipped = 0
    valid   = 0
    total   = 0

    # Calcular grid de tiles
    rows = list(range(0, height, cfg.stride))
    cols = list(range(0, width,  cfg.stride))
    total_tiles = len(rows) * len(cols)
    logger.info("Grid: %d linhas × %d colunas = %d tiles potenciais  (stride=%d)",
                len(rows), len(cols), total_tiles, cfg.stride)

    with rasterio.open(str(mosaic_path)) as src:
        pbar = tqdm(total=total_tiles, desc="Tiles", unit="tile")
        for row_idx, row_off in enumerate(rows):
            for col_idx, col_off in enumerate(cols):
                total += 1

                # Janela — trunca nas bordas (sem padding artificial)
                win_h = min(cfg.tile_size, height - row_off)
                win_w = min(cfg.tile_size, width  - col_off)
                window = Window(col_off, row_off, win_w, win_h)
                data   = src.read(1, window=window)

                # Descartar tiles maioritariamente sem dados
                # Detecta sentinel DGT (-9999), nodata explícito e NaN
                nodata_mask = data < _NODATA_SENTINEL
                if nodata is not None:
                    nodata_mask |= (data == nodata)
                nodata_mask |= np.isnan(data)
                nodata_px = nodata_mask.sum()
                frac = nodata_px / data.size
                if frac > cfg.nodata_frac:
                    skipped += 1
                    pbar.update(1)
                    continue

                # Pad para tile_size × tile_size se necessário (borda do raster)
                if win_h < cfg.tile_size or win_w < cfg.tile_size:
                    padded = np.zeros((cfg.tile_size, cfg.tile_size), dtype=np.float32)
                    padded[:win_h, :win_w] = data
                    data = padded

                norm = _normalize(data, nodata)

                # Transform do canto superior esquerdo deste tile
                tile_transform = rasterio.transform.from_bounds(
                    *rasterio.transform.array_bounds(win_h, win_w, src.window_transform(window)),
                    cfg.tile_size, cfg.tile_size,
                )

                tile_name = f"tile_{row_idx:04d}_{col_idx:04d}.tif"
                tile_path = cfg.tiles_dir / tile_name
                tile_profile = {
                    'driver':    'GTiff',
                    'dtype':     'float32',
                    'width':     cfg.tile_size,
                    'height':    cfg.tile_size,
                    'count':     1,
                    'crs':       crs,
                    'transform': tile_transform,
                    'compress':  'deflate',
                    'nodata':    0.0,
                }
                with rasterio.open(str(tile_path), 'w', **tile_profile) as dst:
                    dst.write(norm, 1)

                tile_index.append({
                    'filename':  tile_name,
                    'row_idx':   row_idx,
                    'col_idx':   col_idx,
                    'row_off':   row_off,
                    'col_off':   col_off,
                    'win_h':     win_h,
                    'win_w':     win_w,
                    'nodata_frac': round(float(frac), 4),
                })
                valid += 1
                pbar.update(1)

        pbar.close()

    with open(str(cfg.index_path), 'w') as f:
        json.dump({
            'mosaic':      str(mosaic_path),
            'crs':         str(crs),
            'tile_size':   cfg.tile_size,
            'tile_overlap': cfg.tile_overlap,
            'stride':      cfg.stride,
            'raster_width':  width,
            'raster_height': height,
            'n_tiles':     valid,
            'tiles':       tile_index,
        }, f, indent=2)

    logger.info("Tiles gerados: %d válidos / %d total  (%d descartados por nodata)",
                valid, total, skipped)
    logger.info("Índice guardado em: %s", cfg.index_path)
    return valid


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gera tiles 256×256 do mosaico MDT')
    parser.add_argument('--output-dir', default='/home/jovyan/reef_output',
                        help='Directório de outputs (default: /home/jovyan/reef_output)')
    parser.add_argument('--mosaic',    default='mosaico_algarve.tif',
                        help='Nome do ficheiro mosaico dentro de output_dir')
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overlap',   type=int, default=32)
    parser.add_argument('--nodata-frac', type=float, default=0.80,
                        help='Fracção máxima de nodata antes de descartar tile (0-1)')
    args = parser.parse_args()

    cfg = TileConfig(
        output_dir   = Path(args.output_dir),
        mosaic_name  = args.mosaic,
        tile_size    = args.tile_size,
        tile_overlap = args.overlap,
        nodata_frac  = args.nodata_frac,
    )
    logger.info("Config: tile=%d  overlap=%d  stride=%d  nodata_frac=%.0f%%",
                cfg.tile_size, cfg.tile_overlap, cfg.stride, cfg.nodata_frac * 100)
    n = generate_tiles(cfg)
    logger.info("Próximo passo: python scripts/dgt_inference_unet.py --output-dir %s",
                args.output_dir)
    sys.exit(0 if n > 0 else 1)
