#!/usr/bin/env python3
"""
04_merge_mdt.py — Mosaico contínuo dos DEM rasters em EPSG:3763.

Lê todos os DEMs gerados por 03_process_laz.py e funde-os num único
GeoTIFF mosaico usando rasterio.merge com estratégia "first" (sem overlap).

Reprojecta para EPSG:3763 (PT-TM06/ETRS89) se necessário.
Output com compressão DEFLATE + tiles 256×256 para eficiência de leitura.

Input:  {output_dir}/dem/*.tif    (DEMs individuais — saída de 03_process_laz.py)
Output: {output_dir}/mosaico_algarve.tif  (EPSG:3763, float32, comprimido)

Corre no DGT JupyterHub (96 cores, 503 GB RAM, CPU-only).
Um mosaico Algarve típico ≈ 10 000 × 20 000 px ≈ 800 MB.
Com 485 GB de RAM livres, pode ser carregado integralmente.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('reef_pipeline.log', mode='a')],
)
logger = logging.getLogger(__name__)

# Repo root: auto-detect from this file's location so the script runs wherever
# the repo is cloned. Override with REEF_PIPELINE_ROOT if needed.
_env_root = os.environ.get("REEF_PIPELINE_ROOT")
WORK = Path(_env_root) if _env_root else Path(__file__).resolve().parents[2]
if WORK.exists():
    sys.path.insert(0, str(WORK))

from src.pipeline_config import PipelineConfig

try:
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.error("rasterio não instalado — pip install rasterio")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # type: ignore[assignment]


def merge_dem_rasters(cfg: PipelineConfig, force: bool = False) -> Path:
    """
    Funde todos os DEM .tif em dem/ num único mosaico em EPSG:3763.

    Estratégia de merge: "first" (sem média — evita artefactos nas bordas).
    A reprojecção para EPSG:3763 preserva a escala métrica para o PDAL.

    Args:
        cfg:   PipelineConfig
        force: re-gera mesmo que o mosaico já exista

    Returns:
        Path para o mosaico gerado.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio é obrigatório")

    mosaic_path = cfg.mosaic_path
    if mosaic_path.exists() and not force:
        logger.info("Mosaico já existe: %s  (usar --force para re-gerar)", mosaic_path)
        with rasterio.open(str(mosaic_path)) as src:
            logger.info("  Dimensões: %d×%d px  CRS: %s", src.width, src.height, src.crs)
        return mosaic_path

    dem_dir   = cfg.output_dir / 'dem'
    dem_files = sorted(dem_dir.glob('*.tif'))
    if not dem_files:
        raise FileNotFoundError(
            f"Nenhum DEM em {dem_dir}\n"
            f"Corra primeiro: python scripts/lidar/03_process_laz.py"
        )

    logger.info("Merge: %d DEMs  CRS destino: %s", len(dem_files), cfg.crs_work)
    t0 = time.time()

    # Abrir todos os DEMs (lazy — sem carregar dados ainda)
    src_files: List[rasterio.DatasetReader] = []
    try:
        for p in tqdm(dem_files, desc='Abrir DEMs', unit='file'):
            try:
                src_files.append(rasterio.open(str(p)))
            except Exception as exc:
                logger.warning("Ignorar %s: %s", p.name, exc)

        if not src_files:
            raise RuntimeError("Nenhum DEM pôde ser aberto")

        logger.info("Merge em memória (%d ficheiros válidos)...", len(src_files))
        mosaic, out_transform = merge(src_files, method='first', nodata=None)
        out_crs = src_files[0].crs

        logger.info("Mosaico bruto: %d×%d px  CRS: %s", mosaic.shape[2], mosaic.shape[1], out_crs)

        # Reprojectar para EPSG:3763 se necessário
        if str(out_crs) != cfg.crs_work:
            logger.info("Reprojectando %s → %s...", out_crs, cfg.crs_work)
            dst_crs = rasterio.crs.CRS.from_string(cfg.crs_work)
            tr, w, h = calculate_default_transform(
                out_crs, dst_crs,
                mosaic.shape[2], mosaic.shape[1],
                *rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], out_transform),
            )
            reprojected = np.empty((1, h, w), dtype=np.float32)
            reproject(
                source=mosaic,
                destination=reprojected,
                src_transform=out_transform,
                src_crs=out_crs,
                dst_transform=tr,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            mosaic       = reprojected
            out_transform = tr
            out_crs      = dst_crs

        # Guardar mosaico final
        profile = {
            'driver':    'GTiff',
            'dtype':     'float32',
            'width':     mosaic.shape[2],
            'height':    mosaic.shape[1],
            'count':     1,
            'crs':       out_crs,
            'transform': out_transform,
            'compress':  'deflate',
            'predictor': 2,
            'tiled':     True,
            'blockxsize': 256,
            'blockysize': 256,
        }

        mosaic_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(mosaic_path), 'w', **profile) as dst:
            dst.write(mosaic[0], 1)
            dst.update_tags(
                source='DGT MDT-50cm — ReefUNet Pipeline',
                classes='2 (Terreno Nu) + 9 (Água)',
                n_tiles=str(len(src_files)),
                pipeline_step='04_merge_mdt',
            )

        size_mb = mosaic_path.stat().st_size / 1e6
        logger.info("Mosaico guardado: %s  (%.0f MB, %dx%d px, %.1f s)",
                    mosaic_path, size_mb, mosaic.shape[2], mosaic.shape[1],
                    time.time() - t0)
    finally:
        for src in src_files:
            src.close()

    logger.info("Próximo passo: python scripts/lidar/05_generate_tiles.py --output-dir %s",
                cfg.output_dir)
    return mosaic_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mosaico MDT — rasterio.merge → EPSG:3763')
    parser.add_argument('--output-dir', default='/home/jovyan/reef_output')
    parser.add_argument('--force', action='store_true',
                        help='Re-gera mosaico mesmo que já exista')
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=Path(args.output_dir))
    result = merge_dem_rasters(cfg, force=args.force)
    logger.info("Mosaico: %s", result)
    sys.exit(0)
