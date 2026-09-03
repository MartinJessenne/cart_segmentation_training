#!/usr/bin/env bash
# run_seg_nano_480.sh — end-to-end resilient setup and training loop for RF-DETR Nano 480
set -euo pipefail

REPO_DIR="/root/cart_segmentation_training"
cd "$REPO_DIR"

mkdir -p logs output/seg_nano_480

echo "$(date +'%Y-%m-%d %H:%M:%S') [1/5] Applying patch_rfdetr.py..."
python3 patch_rfdetr.py

# 1. Check if re-encoded dataset exists, if not, fetch and prepare
if [ ! -f "_rfdetr_dataset_960/train/_annotations.coco.json" ]; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') [2/5] Preparing dataset..."
    if [ -f "cart_rfdetr_960.tar.gz" ] || python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='UItraviolet/cart_segmentation_coco_960', filename='cart_rfdetr_960.tar.gz', repo_type='dataset', local_dir='.')" 2>/dev/null; then
        echo "$(date +'%Y-%m-%d %H:%M:%S') [2/5] Found pre-packaged dataset archive! Extracting in seconds..."
        tar -xzf cart_rfdetr_960.tar.gz
    else
        echo "$(date +'%Y-%m-%d %H:%M:%S') [2/5] Pre-packaged archive not yet on HF, building from raw shards..."
        if [ ! -f "_rfdetr_dataset/train/_annotations.coco.json" ]; then
            echo "$(date +'%Y-%m-%d %H:%M:%S') [2a/5] Fetching RGB & masks (~31.2 GB)..."
            python3 fetch_rgb_masks.py --out _rfdetr_dataset
            echo "$(date +'%Y-%m-%d %H:%M:%S') [2b/5] Converting masks to COCO annotations..."
            python3 masks_to_coco.py --root _rfdetr_dataset --workers 4
        fi
        echo "$(date +'%Y-%m-%d %H:%M:%S') [2c/5] Re-encoding dataset to 960x600 JPEG q95..."
        python3 reencode_dataset.py --src _rfdetr_dataset --dst _rfdetr_dataset_960 --workers 4
        echo "$(date +'%Y-%m-%d %H:%M:%S') [2d/5] Publishing pre-processed dataset to Hugging Face..."
        python3 publish_processed_dataset.py || true
    fi
else
    echo "$(date +'%Y-%m-%d %H:%M:%S') [2/5] _rfdetr_dataset_960 already present."
fi

# 2. Start HF Checkpoint Sync Daemon in background if not already running
if ! pgrep -f "sync_hf.py.*seg_nano_480" >/dev/null 2>&1; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') [3/5] Starting sync_hf daemon..."
    nohup python3 -u sync_hf.py --watch-dir output/seg_nano_480 --interval 60 > logs/sync_hf.log 2>&1 &
    echo "$(date +'%Y-%m-%d %H:%M:%S') sync_hf daemon started (pid $!)"
else
    echo "$(date +'%Y-%m-%d %H:%M:%S') [3/5] sync_hf daemon already running."
fi

# 3. Launch Training Loop
echo "$(date +'%Y-%m-%d %H:%M:%S') [4/5] Starting RF-DETR Nano 480 training loop..."
WANDB_FLAG=""
if [ -n "$WANDB_API_KEY" ] || [ -f "$HOME/.netrc" ]; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') WandB key detected; enabling cloud telemetry (--wandb)."
    WANDB_FLAG="--wandb"
fi

python3 train_rfdetr.py nano \
    --resolution 480 \
    --batch-size 32 \
    --effective-batch 32 \
    --dataset-dir _rfdetr_dataset_960 \
    --eval-interval 5 \
    --patience 25 \
    --epochs 100 \
    $WANDB_FLAG

echo "$(date +'%Y-%m-%d %H:%M:%S') [5/5] Training finished. Running one final sync pass..."
python3 sync_hf.py --watch-dir output/seg_nano_480 --once

echo "$(date +'%Y-%m-%d %H:%M:%S') Pipeline complete!"
