#!/usr/bin/env python3
"""
dgt_auto_launch.py — Watcher que lança inferência + assembly automaticamente.

Corre em background no DGT JupyterHub (start_new_session=True → sobrevive
ao timeout do kernel Jupyter). Poleia de 60 em 60 segundos e lança
dgt_inference_unet.py seguido de 06_assemble_mask.py assim que:
  1. training_log.json reportar epoch 100 (ou treino terminado)
  2. tiles_index.json existir (tiling completo)

Log: /home/jovyan/reef-imagery-pipeline/watcher.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Repo root: auto-detect from this file's location so the script runs wherever
# the repo is cloned. Override with REEF_PIPELINE_ROOT if needed.
_env_root = os.environ.get("REEF_PIPELINE_ROOT")
WORK   = Path(_env_root) if _env_root else Path(__file__).resolve().parents[1]
OUT    = Path('/home/jovyan/reef_output')
LOG    = WORK / 'watcher.log'

INFERENCE_SCRIPT = WORK / 'scripts' / 'dgt_inference_unet.py'
ASSEMBLE_SCRIPT  = WORK / 'scripts' / 'lidar' / '06_assemble_mask.py'
BEST_WEIGHTS     = WORK / 'models' / 'unet_reef_best.pth'
TRAIN_LOG        = WORK / 'training_log.json'
TILES_INDEX      = OUT  / 'tiles_index.json'

POLL_INTERVAL = 60   # segundos entre verificações
MAX_WAIT_H    = 3    # desistir ao fim de 3 horas


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(str(LOG), 'a') as f:
        f.write(line + '\n')


def _training_done() -> tuple[bool, int, float]:
    """Retorna (done, epoch, best_iou)."""
    if not TRAIN_LOG.exists():
        return False, 0, 0.0
    try:
        h = json.loads(TRAIN_LOG.read_text())
        if not h:
            return False, 0, 0.0
        last = h[-1]
        epoch    = last.get('epoch', 0)
        best_iou = last.get('best_iou', 0.0)
        # Considera terminado se chegou a epoch 100 OU se o processo de treino morreu
        # e já temos pelo menos 80 epochs (evita lançar com treino incompleto)
        done = (epoch >= 100)
        return done, epoch, best_iou
    except Exception:
        return False, 0, 0.0


def _launch_background(cmd: list[str], log_path: Path) -> int:
    """Lança processo em background, retorna PID."""
    with open(str(log_path), 'w') as fout:
        p = subprocess.Popen(
            cmd,
            stdout=fout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return p.pid


def main() -> None:
    _log("=" * 50)
    _log("Auto-launch watcher iniciado")
    _log(f"  Aguarda: epoch=100 em {TRAIN_LOG.name} + {TILES_INDEX.name}")
    _log(f"  Poll: {POLL_INTERVAL}s  |  Timeout: {MAX_WAIT_H}h")
    _log("=" * 50)

    deadline = time.time() + MAX_WAIT_H * 3600
    check_n  = 0

    while time.time() < deadline:
        check_n += 1
        train_done, epoch, best_iou = _training_done()
        tiles_done = TILES_INDEX.exists()

        tile_count = len(list((OUT / 'tiles').glob('tile_*.tif'))) if (OUT / 'tiles').exists() else 0
        tiling_str = 'DONE' if tiles_done else f'{tile_count} tiles'
        _log(f"[#{check_n}] Treino={epoch}/100 best={best_iou:.4f}  "
             f"Tiling={tiling_str}  Weights={BEST_WEIGHTS.exists()}")

        if train_done and tiles_done and BEST_WEIGHTS.exists():
            _log("AMBAS AS CONDIÇÕES SATISFEITAS — lançando inferência!")

            inference_log = WORK / 'inference_stdout.log'
            assemble_log  = WORK / 'assemble_stdout.log'

            # Lançar inferência em foreground (blocking dentro deste processo)
            _log(f"A correr: {INFERENCE_SCRIPT}")
            try:
                r = subprocess.run(
                    [sys.executable, str(INFERENCE_SCRIPT),
                     '--output-dir', str(OUT),
                     '--model-path', str(BEST_WEIGHTS),
                     '--batch-size', '32'],
                    capture_output=False,
                    stdout=open(str(inference_log), 'w'),
                    stderr=subprocess.STDOUT,
                    timeout=7200,   # 2 horas
                )
                _log(f"Inferência concluída (exit={r.returncode})")
            except subprocess.TimeoutExpired:
                _log("TIMEOUT na inferência (2h) — verificar manualmente")
                return
            except Exception as exc:
                _log(f"ERRO na inferência: {exc}")
                return

            # Lançar assembly
            _log(f"A correr: {ASSEMBLE_SCRIPT}")
            try:
                r2 = subprocess.run(
                    [sys.executable, str(ASSEMBLE_SCRIPT),
                     '--output-dir', str(OUT)],
                    capture_output=False,
                    stdout=open(str(assemble_log), 'w'),
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
                _log(f"Assembly concluído (exit={r2.returncode})")
            except Exception as exc:
                _log(f"ERRO no assembly: {exc}")
                return

            # Verificar output final
            tif = OUT / 'recife_mask.tif'
            prob = OUT / 'recife_prob.tif'
            _log(f"Outputs: recife_mask.tif={tif.exists()}  recife_prob.tif={prob.exists()}")
            if tif.exists():
                _log(f"  recife_mask.tif: {tif.stat().st_size/1e6:.1f} MB")
            _log("Pipeline completa! Validar com 02_reef_detection_hpc.ipynb")
            return

        time.sleep(POLL_INTERVAL)

    _log(f"TIMEOUT após {MAX_WAIT_H}h sem condições satisfeitas. Verificar manualmente.")


if __name__ == '__main__':
    main()
