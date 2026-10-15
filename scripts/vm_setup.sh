#!/usr/bin/env bash
# vm_setup.sh — Run once per container spawn on DGT JupyterHub
# Usage:
#   bash ~/work/reef_imagery_pipeline/scripts/vm_setup.sh [--train]
#
# What it does (idempotent):
#   1. Clone / pull the repo into ~/work/reef_imagery_pipeline
#   2. Install missing Python packages (torch, smp, etc.)
#   3. Extract training patches if not already present
#   4. Optionally start detached training (--train flag)
#
# Credentials needed for training → set as env vars before calling:
#   export AWS_ENDPOINT_URL2=...   AWS_ACCESS_KEY_ID2=...  AWS_SECRET_ACCESS_KEY2=...
# (These are used only if the training script is extended to push checkpoints to S3.)

set -e
REPO="${REEF_PIPELINE_ROOT:-$HOME/work/reef_imagery_pipeline}"
REPO_URL="https://github.com/JO-SOARES-sea/reef-imagery-pipeline.git"

echo "=== REEF PIPELINE VM SETUP ==="
echo "Target: $REPO"

# ── 1. Clone or pull ─────────────────────────────────────────────────────────
mkdir -p "$(dirname "$REPO")"
if [ -d "$REPO/.git" ]; then
    echo "[1/4] Pulling latest..."
    git -C "$REPO" pull --ff-only
else
    echo "[1/4] Cloning repo..."
    git clone "$REPO_URL" "$REPO"
fi

# ── 2. Install Python packages ────────────────────────────────────────────────
echo "[2/4] Installing Python packages..."
pip install -q \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install -q "numpy<2"
pip install -q segmentation-models-pytorch albumentations timm

# ── 3. Extract training patches ───────────────────────────────────────────────
IMG_DIR="$REPO/data/master_ml_dataset/images"
N_IMGS=$(ls "$IMG_DIR"/*.tif 2>/dev/null | wc -l || echo 0)
if [ "$N_IMGS" -lt 341 ]; then
    echo "[3/4] Extracting training patches ($N_IMGS/341 present)..."
    ZIP="$REPO/data/master_ml_dataset.zip"
    if [ -f "$ZIP" ]; then
        unzip -q "$ZIP" -d "$REPO/data/"
        echo "  Extracted $(ls "$IMG_DIR"/*.tif | wc -l) image patches"
    else
        echo "  ERROR: $ZIP not found. Run git pull to fetch it."
        exit 1
    fi
else
    echo "[3/4] Training patches already present ($N_IMGS images)"
fi

# ── 4. Optionally start training ──────────────────────────────────────────────
if [[ "$*" == *"--train"* ]]; then
    echo "[4/4] Starting training in background..."
    LOG="$REPO/training_stdout.log"
    nohup python "$REPO/scripts/dgt_train_unet.py" \
        > "$LOG" 2>&1 &
    TRAIN_PID=$!
    sleep 5
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "  Training PID=$TRAIN_PID  log=$LOG"
    else
        echo "  ERROR: training died immediately. Check $LOG"
        tail -20 "$LOG"
        exit 1
    fi
else
    echo "[4/4] Skipped training (pass --train to start)"
fi

echo ""
echo "=== SETUP COMPLETE ==="
echo "Repo: $REPO"
echo "To start training: OMP_NUM_THREADS=16 DL_NUM_WORKERS=8 \\"
echo "  nohup python $REPO/scripts/dgt_train_unet.py > $REPO/training_stdout.log 2>&1 &"
echo "To check progress: tail -f $REPO/training_stdout.log"
