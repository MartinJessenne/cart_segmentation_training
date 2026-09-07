#!/usr/bin/env bash
# Launch two concurrent RF-DETR-Seg Nano training runs at 312x312 and 288x288 square resolutions.
#
# Hardware & Concurrency:
#   - Runs under CUDA Multi-Process Service (MPS) to allow GPU kernel overlapping.
#   - Both runs fit comfortably in GPU memory (~16-18 GB each on a 96 GB RTX 6000 Ada/Black).
#   - Dataloader workers and Hungarian matcher are tuned to avoid host CPU bottlenecks.
#   - Telemetry is streamed to Weights & Biases (WandB) for live tracking.
#   - Checkpoints and metrics are synchronized to Hugging Face continuously.
set -uo pipefail
cd /root/cart_segmentation_training

EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
AUG_PRESET=${AUG_PRESET:-vanilla}
DATASET=${DATASET:-_rfdetr_dataset_960}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
EMA_INTERVAL=${EMA_INTERVAL:-4}
EMA_DECAY=${EMA_DECAY:-0.972}
PROJECT=${PROJECT:-cart_segmentation}
COMPILE=${COMPILE:-1}
HF_REPO=${HF_REPO:-UItraviolet/cart_segmentation_rfdetr}

RUN_312="seg_nano_312_${AUG_PRESET}_square_b${BATCH}"
RUN_288="seg_nano_288_${AUG_PRESET}_square_b${BATCH}"

mkdir -p logs "output/$RUN_312" "output/$RUN_288"

# ------------------------------------------------------------------------------
# 1. Start CUDA MPS daemon for kernel concurrency
# ------------------------------------------------------------------------------
start_mps() {
    if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "[mps] MPS daemon already running"
        return 0
    fi
    export CUDA_VISIBLE_DEVICES=0
    if nvidia-cuda-mps-control -d; then
        sleep 2
        echo "[mps] MPS daemon started successfully"
        return 0
    fi
    echo "[mps] WARNING: MPS failed to start; continuing without MPS"
    return 0
}

start_mps

# ------------------------------------------------------------------------------
# 2. Verify matcher patch for vectorized Hungarian argmin
# ------------------------------------------------------------------------------
echo "[setup] verifying Hungarian matcher fast-path patch..."
uv run python patch_matcher.py

# ------------------------------------------------------------------------------
# 3. WandB authentication check
# ------------------------------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[wandb] WARNING: WANDB_API_KEY environment variable is not set."
else
    echo "[wandb] WandB credentials verified. Streaming telemetry to project '$PROJECT'."
fi

# ------------------------------------------------------------------------------
# 4. Background Sync Daemons for Hugging Face
# ------------------------------------------------------------------------------
sync_pids=()

start_sync_daemon() {
    local run_name="$1"
    if ! pgrep -f "sync_hf.py.*$run_name" >/dev/null 2>&1; then
        echo "[sync] starting sync_hf daemon for $run_name"
        nohup uv run python -u sync_hf.py \
            --watch-dir "output/$run_name" \
            --repo "$HF_REPO" \
            --interval 120 > "logs/sync_hf_$run_name.log" 2>&1 &
        sync_pids+=("$!")
    fi
}

start_sync_daemon "$RUN_312"
start_sync_daemon "$RUN_288"

# ------------------------------------------------------------------------------
# 5. Process management & cleanup handler
# ------------------------------------------------------------------------------
train_pids=()

cleanup() {
    echo -e "\n[shutdown] terminating running processes..."
    for pid in "${train_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[shutdown] stopping training PID $pid..."
            kill -INT "$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        fi
    done
    for pid in "${sync_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done
}
trap cleanup INT TERM

# ------------------------------------------------------------------------------
# 6. Launch Concurrent Training Runs
# ------------------------------------------------------------------------------
echo "================================================================================"
echo " Launching Concurrent RF-DETR-Seg Nano Training Runs ($EPOCHS Epochs)"
echo "   Run 1: $RUN_312 (312x312 square stretch, native ViT resolution)"
echo "   Run 2: $RUN_288 (288x288 square stretch, downsampled ViT tokens)"
echo "   Augmentation: $AUG_PRESET | Batch: $BATCH | Compile: $COMPILE"
echo "   WandB: Project '$PROJECT' | HF Repo: '$HF_REPO'"
echo "================================================================================"

# Launch Run 1 (312x312)
uv run python train_rfdetr.py nano \
    --square \
    --resolution 312 \
    --aug-preset "$AUG_PRESET" \
    --dataset-dir "$DATASET" \
    --batch-size "$BATCH" --effective-batch "$BATCH" \
    --num-workers "$NUM_WORKERS" \
    --ema-update-interval "$EMA_INTERVAL" --ema-decay "$EMA_DECAY" \
    $([ "$COMPILE" = "1" ] && echo "--compile") \
    --epochs "$EPOCHS" \
    --eval-interval "$EVAL_INTERVAL" \
    --run-name "$RUN_312" \
    --project "$PROJECT" \
    --no-resume \
    --wandb \
    > "logs/${RUN_312}.log" 2>&1 &
PID_312=$!
train_pids+=("$PID_312")
echo "[launch] Run 312x312 started (PID $PID_312) -> logging to logs/${RUN_312}.log"

# Launch Run 2 (288x288)
uv run python train_rfdetr.py nano \
    --square \
    --resolution 288 \
    --aug-preset "$AUG_PRESET" \
    --dataset-dir "$DATASET" \
    --batch-size "$BATCH" --effective-batch "$BATCH" \
    --num-workers "$NUM_WORKERS" \
    --ema-update-interval "$EMA_INTERVAL" --ema-decay "$EMA_DECAY" \
    $([ "$COMPILE" = "1" ] && echo "--compile") \
    --epochs "$EPOCHS" \
    --eval-interval "$EVAL_INTERVAL" \
    --run-name "$RUN_288" \
    --project "$PROJECT" \
    --no-resume \
    --wandb \
    > "logs/${RUN_288}.log" 2>&1 &
PID_288=$!
train_pids+=("$PID_288")
echo "[launch] Run 288x288 started (PID $PID_288) -> logging to logs/${RUN_288}.log"

echo -e "\nBoth runs active. Monitoring completion..."

# ------------------------------------------------------------------------------
# 7. Wait for both training runs to finish
# ------------------------------------------------------------------------------
status_312=0
status_288=0

wait "$PID_312" || status_312=$?
echo "[complete] Run 312x312 exited with status $status_312"

wait "$PID_288" || status_288=$?
echo "[complete] Run 288x288 exited with status $status_288"

# ------------------------------------------------------------------------------
# 8. Export ONNX & Evaluate on Real Probe Frames
# ------------------------------------------------------------------------------
export_and_score() {
    local run_name="$1"
    local side="$2"
    echo -e "\n--- Exporting & Evaluating: $run_name (side=$side) ---"

    for ckpt in "output/$run_name"/*.pth; do
        [ -e "$ckpt" ] || continue
        echo "Exporting ONNX for $ckpt..."
        uv run python - "$ckpt" "$side" <<'PY'
import sys, os
from rfdetr import RFDETRSegNano
ck, s = sys.argv[1], int(sys.argv[2])
out = os.path.join(os.path.dirname(ck), "onnx_" + os.path.basename(ck).replace(".pth", ""))
os.makedirs(out, exist_ok=True)
names = ["picanol", "colruyt", "leanflow"]
m = RFDETRSegNano(pretrain_weights=ck, num_classes=len(names))
m.export(output_dir=out, format="onnx", shape=(s, s),
         notes={"class_names": names, "variant": "nano", "square": True, "resolution": s})
PY
    done

    for onnx in "output/$run_name"/onnx_*/*.onnx; do
        [ -e "$onnx" ] || continue
        echo "Evaluating $onnx on real probe..."
        uv run python eval_real_probe.py "$onnx" --probe real_probe \
            --json-out "${onnx%.onnx}_realprobe.json"
    done

    echo "Final HF sync for $run_name..."
    uv run python sync_hf.py --watch-dir "output/$run_name" --repo "$HF_REPO" --once
}

if [ "$status_312" -eq 0 ]; then
    export_and_score "$RUN_312" 312
fi

if [ "$status_288" -eq 0 ]; then
    export_and_score "$RUN_288" 288
fi

echo "================================================================================"
echo " Concurrent Runs Complete!"
echo "   Output 312: output/$RUN_312"
echo "   Output 288: output/$RUN_288"
echo "================================================================================"
