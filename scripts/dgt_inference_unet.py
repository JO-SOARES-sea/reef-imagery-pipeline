#!/usr/bin/env python3
"""
dgt_inference_unet.py — Inferência em batch com ReefUNet no DGT HPC.

Lê tiles gerados por 05_generate_tiles.py, corre o ReefUNet treinado
(unet_reef_best.pth) em batch no CPU e guarda as máscaras de saída.

Input:  {output_dir}/tiles/tile_*.tif        (gerados por 05_generate_tiles.py)
        {output_dir}/tiles_index.json
        {model_path}                          (unet_reef_best.pth)
Output: {output_dir}/masks/tile_*.tif        (probabilidades float32 [0,1])
        {output_dir}/inference_log.json       (estatísticas por tile)

Corre no DGT JupyterHub (96 cores AMD EPYC-Milan, sem GPU).
batch_size=32 é seguro com tiles 256×256 em float32 (≈ 256 MB por batch).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('reef_pipeline.log', mode='a')],
)
logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
# Repo root: auto-detect from this file's location so the script runs wherever
# the repo is cloned. Override with REEF_PIPELINE_ROOT if needed.
_env_root = os.environ.get("REEF_PIPELINE_ROOT")
WORK = Path(_env_root) if _env_root else Path(__file__).resolve().parents[1]
if WORK.exists():
    sys.path.insert(0, str(WORK))

try:
    import torch
    torch.set_num_threads(min(os.cpu_count() or 1, 16))
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.error("PyTorch não instalado")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.error("rasterio não instalado")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = lambda x, **kw: x  # type: ignore[assignment]
    HAS_TQDM = False


# ── Configuração ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR  = Path('/home/jovyan/reef_output')
DEFAULT_MODEL_PATH  = WORK / 'models' / 'unet_reef_best.pth'
BATCH_SIZE          = 32    # tiles 256×256 float32 → ≈8 MB/tile → ≈256 MB/batch
TILE_SIZE           = 256
REEF_THRESHOLD      = 0.5   # limiar de probabilidade para classificar como recife


# ── Loader de modelo ──────────────────────────────────────────────────────────

_MODEL = None

def _get_model(model_path: Path) -> 'torch.nn.Module':
    """Carrega ReefUNet uma única vez (singleton)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not HAS_TORCH:
        raise RuntimeError("PyTorch não disponível")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pesos do modelo não encontrados: {model_path}\n"
            f"Treine primeiro: python scripts/dgt_train_unet.py"
        )
    from src.ml_unet_model import ReefUNet
    model = ReefUNet()
    state = torch.load(str(model_path), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("ReefUNet carregado: %s  (%d params / %.1f MB)",
                model_path, n_params, n_params * 4 / 1e6)
    _MODEL = model
    return model


# ── Inferência ────────────────────────────────────────────────────────────────

def _load_tile(tile_path: Path) -> Optional[np.ndarray]:
    """Lê tile GeoTIFF → array float32 (H, W). Retorna None se falhar."""
    try:
        with rasterio.open(str(tile_path)) as src:
            return src.read(1).astype(np.float32)
    except Exception as exc:
        logger.warning("Falha ao ler tile %s: %s", tile_path.name, exc)
        return None


def run_inference(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    batch_size: int  = BATCH_SIZE,
    threshold: float = REEF_THRESHOLD,
    overwrite: bool  = False,
) -> dict:
    """
    Corre inferência ReefUNet em todos os tiles e guarda máscaras.

    Args:
        output_dir: directório com sub-pastas tiles/ e masks/
        model_path: caminho para unet_reef_best.pth
        batch_size: número de tiles por batch de inferência
        threshold:  limiar sigmoid para máscara binária
        overwrite:  se False, salta tiles já processados

    Returns:
        dict com estatísticas: n_processed, n_skipped, mean_reef_fraction, elapsed_s
    """
    if not HAS_TORCH or not HAS_RASTERIO:
        raise RuntimeError("PyTorch e rasterio são obrigatórios")

    index_path = output_dir / 'tiles_index.json'
    if not index_path.exists():
        raise FileNotFoundError(
            f"Índice de tiles não encontrado: {index_path}\n"
            f"Corra primeiro: python scripts/lidar/05_generate_tiles.py"
        )

    with open(str(index_path)) as f:
        tile_index = json.load(f)

    tiles_dir = output_dir / 'tiles'
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)

    model    = _get_model(model_path)
    entries  = tile_index['tiles']
    n_total  = len(entries)
    logger.info("Inferência: %d tiles  batch=%d  threshold=%.2f  device=cpu",
                n_total, batch_size, threshold)

    stats_log: list[dict] = []
    n_processed = n_skipped = 0
    reef_fracs: list[float] = []
    t0 = time.time()

    # Processar em batches
    pbar = tqdm(total=n_total, desc="Inferência", unit="tile")
    for batch_start in range(0, n_total, batch_size):
        batch_entries = entries[batch_start: batch_start + batch_size]

        batch_tensors: list['torch.Tensor'] = []
        batch_valid_entries: list[dict] = []

        for entry in batch_entries:
            out_path = masks_dir / entry['filename']
            if not overwrite and out_path.exists():
                n_skipped += 1
                pbar.update(1)
                continue
            tile_path = tiles_dir / entry['filename']
            arr = _load_tile(tile_path)
            if arr is None:
                pbar.update(1)
                continue
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
            batch_tensors.append(t)
            batch_valid_entries.append(entry)

        if not batch_tensors:
            continue

        # Inferência em batch — CPU, sem gradientes
        with torch.no_grad():
            batch_input  = torch.cat(batch_tensors, dim=0)           # (B,1,H,W)
            batch_probs  = model(batch_input).squeeze(1).cpu().numpy()  # (B,H,W) probs

        # Guardar cada máscara com o mesmo GeoTIFF profile do tile de input
        for entry, prob_arr in zip(batch_valid_entries, batch_probs):
            tile_path = tiles_dir / entry['filename']
            out_path  = masks_dir / entry['filename']
            reef_frac = float((prob_arr > threshold).mean())
            reef_fracs.append(reef_frac)

            # Copiar profile do tile de input para manter CRS e transform.
            # Limpar layout de bloco herdado: blockxsize sem tiled=YES gera
            # "CPLE_IllegalArg: BLOCKXSIZE can only be used with TILED=YES".
            try:
                with rasterio.open(str(tile_path)) as src:
                    out_profile = src.profile.copy()
                for _k in ('blockxsize', 'blockysize', 'tiled'):
                    out_profile.pop(_k, None)
                out_profile.update(dtype='float32', count=1, compress='deflate')
                with rasterio.open(str(out_path), 'w', **out_profile) as dst:
                    dst.write(prob_arr.astype(np.float32), 1)
            except Exception as exc:
                logger.warning("Falha ao guardar máscara %s: %s", entry['filename'], exc)
                continue

            stats_log.append({
                'filename':    entry['filename'],
                'row_idx':     entry['row_idx'],
                'col_idx':     entry['col_idx'],
                'reef_frac':   round(reef_frac, 4),
                'has_reef':    reef_frac > 0.01,
            })
            n_processed += 1
            pbar.update(1)

    pbar.close()
    elapsed = time.time() - t0
    mean_reef = float(np.mean(reef_fracs)) if reef_fracs else 0.0
    tiles_with_reef = sum(1 for r in reef_fracs if r > 0.01)

    log_data = {
        'n_tiles_total':    n_total,
        'n_processed':      n_processed,
        'n_skipped':        n_skipped,
        'mean_reef_frac':   round(mean_reef, 4),
        'tiles_with_reef':  tiles_with_reef,
        'elapsed_s':        round(elapsed, 1),
        'tiles_per_sec':    round(n_processed / max(elapsed, 1), 1),
        'model':            str(model_path),
        'threshold':        threshold,
        'tiles':            stats_log,
    }
    log_path = output_dir / 'inference_log.json'
    with open(str(log_path), 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info("Inferência concluída: %d tiles  %.1f s  (%.1f t/s)",
                n_processed, elapsed, n_processed / max(elapsed, 1))
    logger.info("Tiles com recife (>1%%): %d/%d  (frac média: %.3f)",
                tiles_with_reef, n_processed, mean_reef)
    logger.info("Log: %s", log_path)
    logger.info("Próximo passo: python scripts/lidar/06_assemble_mask.py --output-dir %s",
                output_dir)
    return log_data


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Inferência ReefUNet em batch — DGT HPC (CPU-only)')
    parser.add_argument('--output-dir',  default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--model-path',  default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--batch-size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--threshold',   type=float, default=REEF_THRESHOLD)
    parser.add_argument('--overwrite',   action='store_true',
                        help='Re-processa tiles já existentes em masks/')
    args = parser.parse_args()

    result = run_inference(
        output_dir = Path(args.output_dir),
        model_path = Path(args.model_path),
        batch_size = args.batch_size,
        threshold  = args.threshold,
        overwrite  = args.overwrite,
    )
    sys.exit(0)
