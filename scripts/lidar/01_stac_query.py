#!/usr/bin/env python3
"""
01_stac_query.py — Consulta STAC API DGT e gera manifesto de ficheiros S3.

ÚNICO método válido para descobrir ficheiros no S3 MinIO DGT.
ListObjects está BLOQUEADO → STAC API é a única alternativa.

Input:  AOI bbox em EPSG:4326 (default: Loulé + Albufeira)
Output: {output_dir}/lidar_manifest.json

Corre no DGT JupyterHub (96 cores, 503 GB RAM, CPU-only).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict

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


def query_stac_assets(
    cfg: PipelineConfig,
    collections: List[str],
    max_items: int = 500,
) -> List[Dict]:
    """
    Consulta STAC API DGT e guarda manifesto JSON.

    ÚNICO método válido para descobrir ficheiros no S3 (sem ListObjects).
    AOI Algarve: bbox = (-8.25, 37.00, -8.00, 37.15) em EPSG:4326.

    Args:
        cfg:         PipelineConfig com stac_url, aoi_bbox e manifest_path
        collections: lista de colecções STAC (ex: ['lidar', 'mdt-50cm'])
        max_items:   limite máximo de items STAC retornados

    Returns:
        Lista de dicts com id, key S3, href e collection de cada asset.
    """
    try:
        from pystac_client import Client
    except ImportError:
        raise ImportError("pip install pystac-client")

    logger.info("STAC query: %s", cfg.stac_url)
    logger.info("AOI bbox (WGS84): %s", cfg.aoi_bbox)
    logger.info("Collections: %s  max_items=%d", collections, max_items)

    client = Client.open(cfg.stac_url)
    search = client.search(
        collections=collections,
        bbox=cfg.aoi_bbox,
        max_items=max_items,
    )

    assets: List[Dict] = []
    for item in search.items():
        for key, asset in item.assets.items():
            assets.append({
                'id':         item.id,
                'key':        key,
                'href':       asset.href,
                'collection': item.collection_id,
                'datetime':   str(item.datetime) if item.datetime else None,
                'bbox':       list(item.bbox) if item.bbox else None,
            })

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(cfg.manifest_path), 'w') as f:
        json.dump({
            'stac_url':    cfg.stac_url,
            'aoi_bbox':    list(cfg.aoi_bbox),
            'collections': collections,
            'n_assets':    len(assets),
            'assets':      assets,
        }, f, indent=2)

    logger.info("STAC: %d assets guardados em %s", len(assets), cfg.manifest_path)

    # Resumo por colecção
    by_col: Dict[str, int] = {}
    for a in assets:
        by_col[a['collection']] = by_col.get(a['collection'], 0) + 1
    for col, n in sorted(by_col.items()):
        logger.info("  %s: %d assets", col, n)

    return assets


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Consulta STAC DGT — gera lidar_manifest.json')
    parser.add_argument('--output-dir', default='/home/jovyan/reef_output')
    parser.add_argument('--collections', nargs='+', default=['lidar', 'mdt-50cm'],
                        help='Colecções STAC a consultar')
    parser.add_argument('--max-items', type=int, default=500)
    parser.add_argument('--force', action='store_true',
                        help='Re-consulta mesmo que o manifesto já exista')
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=Path(args.output_dir))

    if cfg.manifest_path.exists() and not args.force:
        with open(str(cfg.manifest_path)) as f:
            existing = json.load(f)
        logger.info("Manifesto já existe: %s (%d assets) — usar --force para re-consultar",
                    cfg.manifest_path, existing.get('n_assets', '?'))
        sys.exit(0)

    assets = query_stac_assets(cfg, args.collections, args.max_items)
    logger.info("Próximo passo: python scripts/lidar/02_download_s3.py --output-dir %s",
                args.output_dir)
    sys.exit(0 if assets else 1)
