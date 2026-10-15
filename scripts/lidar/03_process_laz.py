#!/usr/bin/env python3
"""
03_process_laz.py — Filtragem PDAL de nuvens de pontos LAZ → DEM rasters.

Para cada ficheiro LAZ em raw/:
  1. Filtra Classes 2 (Terreno Nu) e 9 (Água) via PDAL pipeline JSON
  2. Gera DEM GeoTIFF 50cm com interpolação IDW (método padrão DGT)
  3. Guarda em dem/ com o mesmo basename + .tif

Classes LiDAR DGT:
  2 = Terreno Nu (bare earth)        ← manter
  9 = Água                           ← manter (fundo do mar)
  6 = Edifícios                      ← remover
  3,4,5 = Vegetação (baixa/média/alta) ← remover

Paralelismo: multiprocessing.Pool (64 workers) — um processo por ficheiro LAZ.
PDAL é invocado via subprocess (evita problemas de concorrência com a lib C++).

Input:  {output_dir}/raw/*.laz
Output: {output_dir}/dem/*.tif  (GeoTIFF float32, resolução 0.5m, EPSG:3763)

Corre no DGT JupyterHub (96 cores, 503 GB RAM, CPU-only).
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

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
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # type: ignore[assignment]

# Classes LiDAR a manter: 2 (Terreno Nu) + 9 (Água)
KEEP_CLASSES = [2, 9]
DEM_RESOLUTION = 0.5   # metros (resolução MDT-50cm DGT)


def _build_pdal_pipeline(laz_path: Path, dem_path: Path) -> dict:
    """Constrói pipeline PDAL JSON para filtrar classes e gerar DEM."""
    return {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(laz_path),
            },
            {
                "type": "filters.range",
                # Manter apenas Classes 2 (Terreno) e 9 (Água)
                "limits": "Classification[2:2],Classification[9:9]",
            },
            {
                "type": "writers.gdal",
                "filename": str(dem_path),
                "gdaldriver": "GTiff",
                "output_type": "idw",       # Inverse Distance Weighting
                "resolution": DEM_RESOLUTION,
                "radius": 1.0,
                "data_type": "float32",
                "gdalopts": "COMPRESS=DEFLATE,PREDICTOR=2,TILED=YES,BLOCKXSIZE=256,BLOCKYSIZE=256",
            },
        ]
    }


def _process_one_laz(args: tuple) -> Dict:
    """Processa um ficheiro LAZ → DEM. Corre num processo separado."""
    laz_path, dem_path = args

    if dem_path.exists():
        return {'file': laz_path.name, 'status': 'skipped'}

    pipeline = _build_pdal_pipeline(laz_path, dem_path)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(pipeline, tmp)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ['pdal', 'pipeline', tmp_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return {
                'file':   laz_path.name,
                'status': 'error',
                'stderr': result.stderr[-500:],  # ultimos 500 chars
            }
        size = dem_path.stat().st_size if dem_path.exists() else 0
        return {'file': laz_path.name, 'status': 'ok', 'size': size}
    except subprocess.TimeoutExpired:
        return {'file': laz_path.name, 'status': 'error', 'stderr': 'timeout (300s)'}
    except FileNotFoundError:
        return {
            'file': laz_path.name, 'status': 'error',
            'stderr': 'pdal não encontrado — pip install pdal ou conda install -c conda-forge pdal',
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def process_laz_files(cfg: PipelineConfig) -> List[Dict]:
    """
    Filtra todos os LAZ em raw/ e gera DEMs em dem/ com PDAL.

    Paralelismo: multiprocessing.Pool com n_workers processos.
    Cada processo corre uma invocação de pdal separada.

    Returns:
        Lista de resultados por ficheiro.
    """
    raw_dir = cfg.output_dir / 'raw'
    dem_dir = cfg.output_dir / 'dem'
    dem_dir.mkdir(parents=True, exist_ok=True)

    laz_files = sorted(raw_dir.glob('*.laz'))
    if not laz_files:
        logger.warning("Nenhum ficheiro LAZ em %s — correr 02_download_s3.py primeiro", raw_dir)
        return []

    logger.info("PDAL: %d ficheiros LAZ  workers=%d  resolução=%.1fm",
                len(laz_files), cfg.n_workers, DEM_RESOLUTION)
    logger.info("Classes mantidas: %s (Terreno Nu + Água)", KEEP_CLASSES)

    args = [(laz, dem_dir / (laz.stem + '.tif')) for laz in laz_files]

    t0 = time.time()
    with multiprocessing.Pool(processes=cfg.n_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_one_laz, args),
            total=len(args),
            desc='PDAL LAZ→DEM',
            unit='file',
        ))

    ok      = sum(1 for r in results if r['status'] == 'ok')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors  = sum(1 for r in results if r['status'] == 'error')

    logger.info("PDAL concluído em %.1f s: %d ok / %d já existiam / %d erros",
                time.time() - t0, ok, skipped, errors)

    for r in results:
        if r['status'] == 'error':
            logger.error("  ERRO %s: %s", r['file'], r.get('stderr', '')[:200])

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Filtragem PDAL LAZ → DEM — Classes 2+9, remove urbano')
    parser.add_argument('--output-dir', default='/home/jovyan/reef_output')
    parser.add_argument('--workers',    type=int, default=None,
                        help='Nº de processos paralelos (default: min(64, ncores))')
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=Path(args.output_dir))
    if args.workers:
        cfg.n_workers = args.workers

    results = process_laz_files(cfg)
    errors  = sum(1 for r in results if r['status'] == 'error')
    logger.info("Próximo passo: python scripts/lidar/04_merge_mdt.py --output-dir %s",
                args.output_dir)
    sys.exit(0 if errors == 0 else 1)
