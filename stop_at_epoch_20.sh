#!/usr/bin/env bash
# Automatically stops training after epoch 19 validation completes (20 epochs total)
# and ensures final checkpoints are uploaded to Hugging Face.
set -e
cd /root/cart_segmentation_training

echo "[auto_stop] Monitoring training until Epoch 19/20 validation completes..."
while true; do
    if python3 -c "
import csv, sys
with open('output/seg_nano_480/metrics.csv') as f:
    rows = list(csv.DictReader(f))
# Check if epoch 19 validation row has been logged
val_19 = any(r.get('epoch') == '19' and r.get('val/ema_mAP_50_95') for r in rows)
sys.exit(0 if val_19 else 1)
" 2>/dev/null; then
        echo "[auto_stop] Epoch 19 validation complete! Gracefully stopping training..."
        pkill -SIGINT -f train_rfdetr.py || true
        sleep 5
        pkill -f train_rfdetr.py || true
        echo "[auto_stop] Performing final Hugging Face checkpoint sync..."
        python3 -u sync_hf.py --watch-dir output/seg_nano_480 --once
        echo "[auto_stop] All checkpoints verified and synced to Hugging Face. SUCCESS!"
        break
    fi
    sleep 10
done
