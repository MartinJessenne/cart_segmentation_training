#!/usr/bin/env bash
# Score each new best-EMA checkpoint on the REAL bag frames while training runs.
#
# Isaac mAP saturates near 0.985 and cannot separate a deployable model from an
# unusable one, so waiting for the end of a 60-epoch run to learn the only number
# that matters wastes hours. This exports whatever checkpoint exists and scores
# it on real_probe, then waits for the file to change and does it again.
set -uo pipefail
cd /root/cart_segmentation_training
RUN="${1:-output/seg_nano_288_sensor_square_b32}"
SIDE="${2:-288}"
CKPT="$RUN/checkpoint_best_ema.pth"
LAST=""

while true; do
  if [ -f "$CKPT" ]; then
    SIG=$(stat -c "%Y %s" "$CKPT")
    if [ "$SIG" != "$LAST" ]; then
      sleep 10   # let the writer finish
      SIG=$(stat -c "%Y %s" "$CKPT")
      OUT="$RUN/live_onnx"
      rm -rf "$OUT"; mkdir -p "$OUT"
      # export_ckpt.py strips torch.compile's "_orig_mod." key prefix, which
      # RF-DETR's loader does not expect and fails on with KeyError.
      if uv run python export_ckpt.py "$CKPT" "$SIDE" "$OUT" >/tmp/export_err.log 2>&1
      then
        ONNX=$(ls "$OUT"/*.onnx 2>/dev/null | head -1)
        if [ -n "$ONNX" ]; then
          EP=$(python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$RUN/metrics.csv')) if r.get('val/ema_segm_mAP_50_95')]
print(rows[-1]['epoch'] if rows else '?')" 2>/dev/null)
          RES=$(uv run python eval_real_probe.py "$ONNX" --probe real_probe 2>/dev/null | grep -E "leanflow|REAL-DOMAIN")
          echo "SCORE epoch=$EP :: $(echo "$RES" | tr '\n' ' | ')"
          # Keep the checkpoint that is best on the REAL bag. best_ema is chosen
          # on Isaac mAP, which only ever climbs, so it will overwrite a better
          # real-domain checkpoint as training continues.
          ACC=$(echo "$RES" | grep -oE "= +[0-9.]+%" | head -1 | tr -dc "0-9.")
          CONF=$(echo "$RES" | grep -oE "mean score [0-9.]+" | head -1 | awk '{print $3}')
          BEST_FILE="$RUN/best_real_accuracy.txt"
          BEST_CONF_FILE="$RUN/best_real_confidence.txt"
          BEST_ACC=$(cat "$BEST_FILE" 2>/dev/null || echo 0)
          BEST_CONF=$(cat "$BEST_CONF_FILE" 2>/dev/null || echo 0)
          if python3 -c "
import sys
acc, best_acc = float('${ACC:-0}'), float('${BEST_ACC:-0}')
conf, best_conf = float('${CONF:-0}'), float('${BEST_CONF:-0}')
is_better = (acc > best_acc) or (abs(acc - best_acc) < 1e-4 and conf > best_conf)
sys.exit(0 if is_better else 1)
"; then
            echo "$ACC" > "$BEST_FILE"
            echo "$CONF" > "$BEST_CONF_FILE"
            mkdir -p "$RUN/best_real"
            cp -f "$CKPT" "$RUN/best_real/checkpoint_best_real.pth"
            cp -f "$ONNX" "$RUN/best_real/rfdetr-seg-nano-best-real.onnx"
            echo "$EP" > "$RUN/best_real/epoch.txt"
            echo "SCORE   ^ new best on real ($ACC%, conf ${CONF:-?}), archived epoch $EP"
          fi
        fi
      fi
      LAST="$SIG"
    fi
  fi
  sleep 60
done
