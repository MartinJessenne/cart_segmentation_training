#!/usr/bin/env bash
# Train RF-DETR-Seg Nano at 288x288 square geometry with:
#   1. Lever 1: Targeted non-geometric sensor & photometric augmentation (sensor preset:
#      GaussianBlur, GaussNoise, ColorJitter, HorizontalFlip -- no Affine/Rotate).
#   2. Lever 2: DINOv2 backbone regularization (--lr-encoder 1e-5, 10x lower than head).
#   3. Lever 3: Short training horizon (15 epochs) + live real-probe scoring daemon (score_live.sh)
#      to select checkpoints based on real RealSense bag accuracy rather than saturated sim mAP.
#
# Reference: Trial Log 30.22 (2026-09-07) & Status Report (2026-09-03)
set -euo pipefail
cd /root/cart_segmentation_training

RESOLUTION=${RESOLUTION:-288}
EPOCHS=${EPOCHS:-15}
DATASET=${DATASET:-_rfdetr_dataset_960}
BATCH=${BATCH:-32}
LR=${LR:-1e-4}
LR_ENCODER=${LR_ENCODER:-1e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
AUG_PRESET=${AUG_PRESET:-sensor}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
EMA_INTERVAL=${EMA_INTERVAL:-4}
EMA_DECAY=${EMA_DECAY:-0.972}
COMPILE=${COMPILE:-1}
PROJECT=${PROJECT:-cart_segmentation}
HF_REPO=${HF_REPO:-UItraviolet/cart_segmentation_rfdetr}

RUN_NAME="seg_nano_${RESOLUTION}_${AUG_PRESET}_square_b${BATCH}_lrenc1e5"

mkdir -p logs "output/$RUN_NAME"

# 1. Patch Hungarian matcher for single-target fast-path
echo "[1/5] Patching Hungarian matcher..."
uv run python patch_matcher.py

# 2. Start Hugging Face background sync daemon
if ! pgrep -f "sync_hf.py.*$RUN_NAME" >/dev/null 2>&1; then
  echo "[2/5] Starting sync_hf daemon..."
  nohup uv run python -u sync_hf.py --watch-dir "output/$RUN_NAME" --repo "$HF_REPO" --interval 120 \
    > "logs/sync_hf_$RUN_NAME.log" 2>&1 &
fi

# 3. Start live real-probe scoring daemon
echo "[3/5] Starting background real-probe live scoring daemon..."
nohup bash score_live.sh "output/$RUN_NAME" "$RESOLUTION" \
  > "logs/score_live_$RUN_NAME.log" 2>&1 &
SCORE_PID=$!

cleanup() {
  echo -e "\n[shutdown] Terminating background scoring daemon..."
  kill "$SCORE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 4. Launch training
echo "[4/5] Training $RUN_NAME at ${RESOLUTION}x${RESOLUTION} (tokens: $(( (RESOLUTION/12)*(RESOLUTION/12) )))"
uv run python train_rfdetr.py nano \
  --square \
  --resolution "$RESOLUTION" \
  --aug-preset "$AUG_PRESET" \
  --dataset-dir "$DATASET" \
  --batch-size "$BATCH" --effective-batch "$BATCH" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --lr-encoder "$LR_ENCODER" \
  --ema-update-interval "$EMA_INTERVAL" --ema-decay "$EMA_DECAY" \
  $([ "$COMPILE" = "1" ] && echo "--compile") \
  --epochs "$EPOCHS" \
  --eval-interval "$EVAL_INTERVAL" \
  --compute-val-loss \
  --no-eval-ema-only \
  --run-name "$RUN_NAME" \
  --project "$PROJECT" \
  --no-resume \
  --wandb

# 5. Export checkpoints & score all on real probe
echo "[5/5] Exporting ONNX graphs and scoring final checkpoints on real frames..."
for ckpt in "output/$RUN_NAME"/*.pth; do
  [ -e "$ckpt" ] || continue
  echo "=== Exporting $ckpt"
  uv run python export_ckpt.py "$ckpt" "$RESOLUTION" "output/$RUN_NAME/onnx_$(basename "$ckpt" .pth)"
done

for onnx in "output/$RUN_NAME"/onnx_*/*.onnx; do
  [ -e "$onnx" ] || continue
  echo "=== Evaluating $onnx on real probe"
  uv run python eval_real_probe.py "$onnx" --probe real_probe \
    --json-out "${onnx%.onnx}_realprobe.json"
done

uv run python sync_hf.py --watch-dir "output/$RUN_NAME" --repo "$HF_REPO" --once
echo "================================================================================"
echo "RUN COMPLETE: output/$RUN_NAME"
if [ -f "output/$RUN_NAME/best_real_accuracy.txt" ]; then
  echo "Best real-domain accuracy achieved: $(cat "output/$RUN_NAME/best_real_accuracy.txt")%"
  echo "Best real checkpoint: output/$RUN_NAME/best_real/rfdetr-seg-nano-best-real.onnx"
fi
echo "================================================================================"
