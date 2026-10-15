#!/usr/bin/env python3
"""
02_download_s3.py — Download paralelo de ficheiros LAZ/TIF do S3 MinIO DGT.

Lê o manifesto gerado por 01_stac_query.py e descarrega os ficheiros
em paralelo com ThreadPoolExecutor (32 workers por default).

NUNCA usar list_objects_v2() — ListObjects está BLOQUEADO.
Os hrefs no manifesto são os únicos S3 keys válidos.

Input:  {output_dir}/lidar_manifest.json
Output: {output_dir}/raw/*.laz    (nuvens de pontos LiDAR)
        {output_dir}/raw/*.tif    (MDT GeoTIFF 50cm)

Credenciais: AWS_ACCESS_KEY_HPC_ID1 / AWS_SECRET_KEY_HPC_ID1 (env vars Apptainer).

Corre no DGT JupyterHub (96 cores, 503 GB RAM, CPU-only).
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

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


def _parse_s3_key(href: str) -> Tuple[str, str]:
    """Extrai bucket e key de um href S3. Ex: s3://lidar/LO-216021.laz → (lidar, LO-216021.laz)."""
    p = urlparse(href)
    if p.scheme == 's3':
        return p.netloc, p.path.lstrip('/')
    # href pode ser URL HTTP do MinIO — extrair path
    parts = p.path.lstrip('/').split('/', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Não foi possível extrair bucket/key de: {href}")


def _download_one(args: Tuple) -> Dict:
    """Descarrega um asset S3. Devolve dict com resultado."""
    cfg, asset, raw_dir = args
    href       = asset['href']
    collection = asset.get('collection', '')

    # Escolher endpoint e bucket correcto
    if 'lidar' in collection or href.endswith('.laz'):
        s3 = cfg.s3_client_stor001()
        bucket = cfg.lidar_bucket
    else:
        s3 = cfg.s3_client_stor002()
        bucket = cfg.mdt_bucket

    try:
        _, key = _parse_s3_key(href)
    except ValueError:
        # href é o key directo
        key = asset.get('key', href)

    filename = Path(key).name
    out_path  = raw_dir / filename

    if out_path.exists():
        return {'filename': filename, 'status': 'skipped', 'size': out_path.stat().st_size}

    try:
        s3.download_file(bucket, key, str(out_path))
        size = out_path.stat().st_size
        return {'filename': filename, 'status': 'ok', 'size': size}
    except Exception as exc:
        logger.warning("Falha a descarregar %s: %s", key, exc)
        return {'filename': filename, 'status': 'error', 'error': str(exc)}


def download_assets(
    cfg: PipelineConfig,
    collections: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Descarrega assets do manifesto em paralelo com ThreadPoolExecutor.

    Args:
        cfg:         PipelineConfig (credenciais via env vars)
        collections: filtrar por colecção (None = todas)

    Returns:
        Lista de resultados por ficheiro (status ok/skipped/error).
    """
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {cfg.manifest_path}\n"
            f"Corra primeiro: python scripts/lidar/01_stac_query.py"
        )

    with open(str(cfg.manifest_path)) as f:
        manifest = json.load(f)

    assets = manifest['assets']
    if collections:
        assets = [a for a in assets if a.get('collection') in collections]

    raw_dir = cfg.output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Download: %d assets  workers=%d  destino=%s",
                len(assets), cfg.n_io_threads, raw_dir)

    t0    = time.time()
    args  = [(cfg, a, raw_dir) for a in assets]
    results: List[Dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_io_threads) as pool:
        futures = {pool.submit(_download_one, a): a for a in args}
        pbar    = tqdm(concurrent.futures.as_completed(futures),
                       total=len(futures), desc='Download S3', unit='file')
        for fut in pbar:
            res = fut.result()
            results.append(res)
            if res['status'] == 'ok':
                pbar.set_postfix({'last': res['filename'], 'MB': f"{res['size']/1e6:.1f}"})

    ok      = sum(1 for r in results if r['status'] == 'ok')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors  = sum(1 for r in results if r['status'] == 'error')
    total_mb = sum(r.get('size', 0) for r in results) / 1e6

    logger.info("Download concluído em %.1f s: %d ok / %d já existiam / %d erros  (%.0f MB)",
                time.time() - t0, ok, skipped, errors, total_mb)

    if errors > 0:
        for r in results:
            if r['status'] == 'error':
                logger.error("  ERRO %s: %s", r['filename'], r.get('error', ''))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download paralelo S3 — DGT HPC')
    parser.add_argument('--output-dir',  default='/home/jovyan/reef_output')
    parser.add_argument('--collections', nargs='+', default=None,
                        help='Filtrar colecções (default: todas no manifesto)')
    parser.add_argument('--workers',     type=int, default=32,
                        help='Número de threads de download paralelo')
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=Path(args.output_dir))
    cfg.n_io_threads = args.workers

    results = download_assets(cfg, args.collections)
    errors  = sum(1 for r in results if r['status'] == 'error')
    logger.info("Próximo passo: python scripts/lidar/03_process_laz.py --output-dir %s",
                args.output_dir)
    sys.exit(0 if errors == 0 else 1)
