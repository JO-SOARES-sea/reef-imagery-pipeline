#!/usr/bin/env python3
"""
local_monitor.py — Poleia o DGT a partir do macOS e regista marcos da pipeline.

Corre em background local. A cada POLL_INTERVAL segundos abre um kernel
no JupyterHub, lê watcher.log + estado dos outputs, e imprime o progresso.
Sai com código 0 quando recife_mask.tif existe (pipeline completa) ou após
MAX_MIN minutos.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY   = REPO / '.venv' / 'bin' / 'python'
CTRL = REPO / 'scripts' / 'jupyterhub_control.py'

POLL_INTERVAL = 180   # segundos entre polls
MAX_MIN       = 90    # desistir após 90 min

PROBE = r'''
import json
from pathlib import Path
WORK = Path('/home/jovyan/reef-imagery-pipeline')
OUT  = Path('/home/jovyan/reef_output')

# Treino
ep, best = 0, 0.0
tl = WORK / 'training_log.json'
if tl.exists():
    h = json.loads(tl.read_text())
    if h:
        ep, best = h[-1]['epoch'], h[-1]['best_iou']

# Tiling
idx = OUT / 'tiles_index.json'
tiling = 'DONE' if idx.exists() else str(len(list((OUT/'tiles').glob('tile_*.tif'))) if (OUT/'tiles').exists() else 0)

# Inferência
masks = len(list((OUT/'masks').glob('tile_*.tif'))) if (OUT/'masks').exists() else 0

# Output final
final = (OUT/'recife_mask.tif').exists()
prob  = (OUT/'recife_prob.tif').exists()
gpkg  = (OUT/'recife_mask.gpkg').exists()

# Watcher tail
wl = WORK / 'watcher.log'
tail = wl.read_text().strip().split('\n')[-1] if wl.exists() else 'no-log'

print(f'PROBE|ep={ep}|best={best:.4f}|tiling={tiling}|masks={masks}|final={final}|prob={prob}|gpkg={gpkg}|watch={tail}')
'''


def probe() -> dict:
    r = subprocess.run(
        [str(PY), str(CTRL), 'run-code', PROBE],
        capture_output=True, text=True, timeout=120,
    )
    out = r.stdout
    for line in out.split('\n'):
        if line.startswith('PROBE|'):
            parts = dict(p.split('=', 1) for p in line[6:].split('|'))
            return parts
    return {}


def main() -> None:
    print(f"{time.strftime('%H:%M:%S')} | Monitor local iniciado  (poll={POLL_INTERVAL}s)")
    deadline = time.time() + MAX_MIN * 60
    n = 0
    last_milestone = ''
    while time.time() < deadline:
        n += 1
        try:
            p = probe()
        except Exception as exc:
            print(f"{time.strftime('%H:%M:%S')} | [#{n}] probe falhou: {exc}")
            time.sleep(POLL_INTERVAL)
            continue

        if not p:
            print(f"{time.strftime('%H:%M:%S')} | [#{n}] sem resposta")
            time.sleep(POLL_INTERVAL)
            continue

        print(f"{time.strftime('%H:%M:%S')} | [#{n}] treino={p['ep']}/100 best={p['best']}  "
              f"tiling={p['tiling']}  masks={p['masks']}  final={p['final']}")

        # Detectar marcos
        milestone = ''
        if p['final'] == 'True':
            milestone = 'COMPLETO'
        elif int(p['masks']) > 0:
            milestone = 'INFERENCIA'
        elif p['tiling'] == 'DONE' and p['ep'] == '100':
            milestone = 'PRONTO_INFERENCIA'
        if milestone and milestone != last_milestone:
            print(f"{time.strftime('%H:%M:%S')} | >>> MARCO: {milestone}  ({p['watch']})")
            last_milestone = milestone

        if p['final'] == 'True':
            print(f"{time.strftime('%H:%M:%S')} | ✓ PIPELINE COMPLETA — recife_mask.tif gerado")
            print(f"  prob={p['prob']}  gpkg={p['gpkg']}")
            return

        time.sleep(POLL_INTERVAL)

    print(f"{time.strftime('%H:%M:%S')} | Timeout após {MAX_MIN} min")


if __name__ == '__main__':
    main()
