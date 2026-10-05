#!/usr/bin/env bash
set -euo pipefail

# Source secrets if present
if [ -r "/marimo/storage/secret.sh" ]; then
  set -a
  . "/marimo/storage/secret.sh"
  set +a
fi

echo "========================================================"
echo " 1. Cloning repository into /root/cart_seg..."
echo "========================================================"
if [ ! -d "/root/cart_seg/.git" ]; then
    rm -rf /root/cart_seg
    git clone https://github.com/MartinJessenne/cart_segmentation_training.git /root/cart_seg
fi

cd /root/cart_seg

echo "========================================================"
echo " 2. Syncing python environment (.venv)..."
echo "========================================================"
export UV_CONCURRENT_DOWNLOADS=4
uv sync
uv cache clean

echo "========================================================"
echo " 3. Applying RF-DETR patches..."
echo "========================================================"
uv run python patch_rfdetr.py
uv run python patch_matcher.py

echo "========================================================"
echo " 4. Downloading golden checkpoint..."
echo "========================================================"
mkdir -p pretrained/seg_nano_288_sensor_square_b32_frozen
CKPT="pretrained/seg_nano_288_sensor_square_b32_frozen/checkpoint_best_ema.pth"
if [ ! -s "$CKPT" ]; then
    curl -sSL -H "Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/UItraviolet/cart_segmentation_rfdetr/resolve/main/seg_nano_288_sensor_square_b32_frozen/checkpoint_best_ema.pth \
        -o "$CKPT"
    echo "   Golden checkpoint saved: $CKPT"
else
    echo "   Golden checkpoint already present."
fi

echo "========================================================"
echo " 5. Downloading & extracting dataset archive..."
echo "========================================================"
if [ ! -f "_rfdetr_dataset_960/train/_annotations.coco.json" ]; then
    echo "   Downloading cart_rfdetr_960.tar.gz (5.3 GB)..."
    curl -sSL -H "Authorization: Bearer $HF_TOKEN" \
        https://huggingface.co/datasets/UItraviolet/cart_segmentation_coco_960/resolve/main/cart_rfdetr_960.tar.gz \
        -o cart_rfdetr_960.tar.gz
    echo "   Extracting into _rfdetr_dataset_960/..."
    tar -xzf cart_rfdetr_960.tar.gz
    rm -f cart_rfdetr_960.tar.gz
    echo "   Dataset extraction complete!"
else
    echo "   _rfdetr_dataset_960 already present."
fi

echo "========================================================"
echo " SETUP COMPLETE! Workspace is ready in /root/cart_seg"
echo "========================================================"
