#!/usr/bin/env bash
set -euo pipefail

# Source molab secrets if present
if [ -r "/marimo/storage/secret.sh" ]; then
  set -a
  . "/marimo/storage/secret.sh"
  set +a
fi

cd /home/marimo/cart_segmentation_training

# 1. Patch Hungarian matcher
echo "==> [1/3] Patching matcher..."
uv run python patch_matcher.py

# 2. Download golden checkpoint
mkdir -p pretrained/seg_nano_288_sensor_square_b32_frozen
CKPT="pretrained/seg_nano_288_sensor_square_b32_frozen/checkpoint_best_ema.pth"
if [ ! -s "$CKPT" ]; then
    echo "==> [2/3] Downloading golden checkpoint (133 MB)..."
    curl -sSL -H "Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/UItraviolet/cart_segmentation_rfdetr/resolve/main/seg_nano_288_sensor_square_b32_frozen/checkpoint_best_ema.pth \
        -o "$CKPT"
    echo "    Checkpoint ready: $CKPT"
else
    echo "==> [2/3] Golden checkpoint already present: $CKPT"
fi

# 3. Download and extract dataset
if [ ! -f "_rfdetr_dataset_960/train/_annotations.coco.json" ]; then
    echo "==> [3/3] Downloading dataset archive (5.3 GB)..."
    curl -sSL -H "Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/datasets/UItraviolet/cart_segmentation_coco_960/resolve/main/cart_rfdetr_960.tar.gz \
        -o cart_rfdetr_960.tar.gz
    echo "    Extracting dataset..."
    tar -xzf cart_rfdetr_960.tar.gz
    rm -f cart_rfdetr_960.tar.gz
    echo "    Dataset extracted to _rfdetr_dataset_960/"
else
    echo "==> [3/3] Dataset already present in _rfdetr_dataset_960/"
fi

echo "========================================================"
echo " PREPARATION 100% COMPLETE! Ready to launch LP-FT runs."
echo "========================================================"
