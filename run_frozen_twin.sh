#!/usr/bin/env bash
# Launch two concurrent RF-DETR-Seg training runs with FROZEN DINOv2 BACKBONES:
#   Run 1: RF-DETR-Seg Nano  (288x288 square, sensor augmentations, frozen backbone)
#   Run 2: RF-DETR-Seg Small (288x288 square, sensor augmentations, frozen backbone)
#
# Hypothesis: DINOv2 was pretrained on 142M natural real-world images. Freezing the
# backbone entirely (Linear Probing / Head-Only tuning) prevents catastrophic feature
# distortion on synthetic Isaac Sim shaders while allowing the transformer query decoder
# and mask prediction heads to learn robust cart segmentation.
set -uo pipefail

# Source molab secrets if present (WANDB_API_KEY, HF_TOKEN)
if [ -r "/marimo/storage/secret.sh" ]; then
  set -a
  . "/marimo/storage/secret.sh"
  set +a
fi

cd /root/cart_segmentation_training

RESOLUTION=${RESOLUTION:-288}
EPOCHS=${EPOCHS:-15}
DATASET=${DATASET:-_rfdetr_dataset_960}
BATCH=${BATCH:-32}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-8}
AUG_PRESET=${AUG_PRESET:-sensor}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
EMA_INTERVAL=${EMA_INTERVAL:-4}
EMA_DECAY=${EMA_DECAY:-0.972}
COMPILE=${COMPILE:-1}
PROJECT=${PROJECT:-cart_segmentation}
HF_REPO=${HF_REPO:-UItraviolet/cart_segmentation_rfdetr}

RUN_NANO="seg_nano_${RESOLUTION}_${AUG_PRESET}_square_b${BATCH}_frozen"
RUN_SMALL="seg_small_${RESOLUTION}_${AUG_PRESET}_square_b${BATCH}_frozen"

mkdir -p logs "output/$RUN_NANO" "output/$RUN_SMALL"

# 1. Patch Hungarian matcher for single-target fast-path
echo "[setup] Verifying Hungarian matcher patch..."
uv run python patch_matcher.py

# 2. Start Hugging Face background sync daemons
start_sync() {
  local run_name="$1"
  if ! pgrep -f "sync_hf.py.*$run_name" >/dev/null 2>&1; then
    echo "[sync] Starting sync_hf daemon for $run_name"
    nohup uv run python -u sync_hf.py --watch-dir "output/$run_name" --repo "$HF_REPO" --interval 120 \
      > "logs/sync_hf_$run_name.log" 2>&1 &
  fi
}
start_sync "$RUN_NANO"
start_sync "$RUN_SMALL"

# 3. Start live real-probe scoring daemons
start_score() {
  local run_name="$1"
  if ! pgrep -f "score_live.sh.*$run_name" >/dev/null 2>&1; then
    echo "[score] Starting live scoring daemon for $run_name"
    nohup bash score_live.sh "output/$run_name" "$RESOLUTION" \
      > "logs/score_live_$run_name.log" 2>&1 &
  fi
}
start_score "$RUN_NANO"
start_score "$RUN_SMALL"

train_pids=()

cleanup() {
  echo -e "\n[shutdown] Terminating processes..."
  for pid in "${train_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

echo "================================================================================"
echo " Launching Dual Frozen-Backbone Training Runs on RTX PRO 6000 Blackwell"
echo "   Run 1: $RUN_NANO (Nano variant, frozen backbone)"
echo "   Run 2: $RUN_SMALL (Small variant, frozen backbone)"
echo "   Geometry: ${RESOLUTION}x${RESOLUTION} square stretch | Aug: $AUG_PRESET"
echo "   Telemetry: WandB project '$PROJECT' | Eval interval: every $EVAL_INTERVAL epoch(s)"
echo "================================================================================"

# Launch Run 1: Nano Frozen
echo "[launch] Starting Nano (frozen)..."
uv run python train_rfdetr.py nano \
  --square \
  --resolution "$RESOLUTION" \
  --aug-preset "$AUG_PRESET" \
  --dataset-dir "$DATASET" \
  --batch-size "$BATCH" --effective-batch "$BATCH" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --freeze-backbone \
  --ema-update-interval "$EMA_INTERVAL" --ema-decay "$EMA_DECAY" \
  $([ "$COMPILE" = "1" ] && echo "--compile") \
  --epochs "$EPOCHS" \
  --eval-interval "$EVAL_INTERVAL" \
  --compute-val-loss \
  --no-eval-ema-only \
  --run-name "$RUN_NANO" \
  --project "$PROJECT" \
  --no-resume \
  --wandb \
  > "logs/${RUN_NANO}.log" 2>&1 &
PID_NANO=$!
train_pids+=("$PID_NANO")
echo "[launch] Nano started (PID $PID_NANO) -> logs/${RUN_NANO}.log"

# Launch Run 2: Small Frozen
echo "[launch] Starting Small (frozen)..."
uv run python train_rfdetr.py small \
  --square \
  --resolution "$RESOLUTION" \
  --aug-preset "$AUG_PRESET" \
  --dataset-dir "$DATASET" \
  --batch-size "$BATCH" --effective-batch "$BATCH" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --freeze-backbone \
  --ema-update-interval "$EMA_INTERVAL" --ema-decay "$EMA_DECAY" \
  $([ "$COMPILE" = "1" ] && echo "--compile") \
  --epochs "$EPOCHS" \
  --eval-interval "$EVAL_INTERVAL" \
  --compute-val-loss \
  --no-eval-ema-only \
  --run-name "$RUN_SMALL" \
  --project "$PROJECT" \
  --no-resume \
  --wandb \
  > "logs/${RUN_SMALL}.log" 2>&1 &
PID_SMALL=$!
train_pids+=("$PID_SMALL")
echo "[launch] Small started (PID $PID_SMALL) -> logs/${RUN_SMALL}.log"

echo -e "\nBoth runs active on GPU. Awaiting completion..."

wait "$PID_NANO" || echo "Nano run exited with $?"
wait "$PID_SMALL" || echo "Small run exited with $?"

# Export ONNX and run final scoring
score_and_export() {
  local run_name="$1"
  local variant="$2"
  echo "=== Post-training ONNX export & scoring for $run_name ==="
  for ckpt in "output/$run_name"/*.pth; do
    [ -e "$ckpt" ] || continue
    uv run python export_ckpt.py "$ckpt" "$RESOLUTION" "output/$run_name/onnx_$(basename "$ckpt" .pth)"
  done
  for onnx in "output/$run_name"/onnx_*/*.onnx; do
    [ -e "$onnx" ] || continue
    uv run python eval_real_probe.py "$onnx" --probe real_probe --json-out "${onnx%.onnx}_realprobe.json"
  done
  uv run python sync_hf.py --watch-dir "output/$run_name" --repo "$HF_REPO" --once
}

score_and_export "$RUN_NANO" nano
score_and_export "$RUN_SMALL" small

echo "================================================================================"
echo " BOTH FROZEN RUNS FINISHED!"
echo "   Nano best real accuracy : $(cat "output/$RUN_NANO/best_real_accuracy.txt" 2>/dev/null || echo '?')%"
echo "   Small best real accuracy: $(cat "output/$RUN_SMALL/best_real_accuracy.txt" 2>/dev/null || echo '?')%"
echo "================================================================================"
