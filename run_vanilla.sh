#!/usr/bin/env bash
# RF-DETR-Seg nano on RF-DETR's OWN geometry and augmentation.
#
# Three deliberate departures from run_sim2real.sh, each aimed at the measured
# failure rather than at Isaac mAP:
#
#   1. SQUARE. square_resize_div_64=True is RF-DETR's default: A.Resize(s, s),
#      a straight stretch, no letterbox and no crop. Train and inference distort
#      aspect identically, so the network stays consistent with itself. It also
#      makes every sample in a batch the same size, which is the condition
#      patch_rfdetr.py existed to work around -- so that patch is NOT applied.
#
#   2. RESOLUTION 336. Measured on the real bag with the 432x768 model: the cart
#      spans 418 of 2304 tokens at native scale and is classified correctly 0% of
#      the time; shrunk until it spans ~137 tokens it is correct 100% of the time,
#      with a smooth ramp between. A 336 square is 28x28 = 784 tokens and the cart
#      keeps its fraction of the frame, putting it near 140 tokens. Both sides
#      must stay divisible by 24 (patch_size 12 x num_windows 2); 336 = 14 x 24.
#
#   3. VANILLA AUGMENTATION. aug_config=None hands RF-DETR its own AUG_CONFIG on
#      the torchvision path. The custom Affine/ColorJitter preset moved real
#      accuracy from 0.3% to 7.4%, so it is not what is being tested here.
#
# Selection is on the REAL probe, not Isaac mAP, which saturates at 0.985 and
# cannot separate a deployable model from an unusable one.
set -euo pipefail
cd /root/cart_segmentation_training

RESOLUTION=${RESOLUTION:-336}
EPOCHS=${EPOCHS:-60}
DATASET=${DATASET:-_rfdetr_dataset_960}
# Throughput settings, chosen against the measured bottleneck rather than the
# dataloader. At batch 32 the run held 2.22 steps/s with the GPU at 56% and the
# workers at 34%: the limit is per-step main-process work -- scipy Hungarian
# matching, which synchronises the GPU, and the EMA copy of 33.6M params. Both
# are per STEP, so the lever is fewer, larger steps.
# Batch 32, and NOT larger: measured, batch 128 gave 68 img/s against 71 at
# batch 32 while using 66 GB instead of 18. Throughput is flat in batch size
# because matcher.py calls scipy's linear_sum_assignment once per image per
# query group -- group_detr is 13, so 13 calls per image however they are
# batched. Enlarging the batch cannot amortise a per-image cost; it only buys
# an LR change that would confound the comparison against the sim2real run.
BATCH=${BATCH:-32}
# RF-DETR's own rates. Left empty so the flags are omitted entirely rather than
# re-stating a default that upstream may move.
LR=${LR:-}
LR_ENCODER=${LR_ENCODER:-}
# decay' = 1 - k(1 - decay) holds the averaging horizon at ~143 steps while
# copying the weights a quarter as often.
EMA_INTERVAL=${EMA_INTERVAL:-4}
EMA_DECAY=${EMA_DECAY:-0.972}
RUN_NAME="seg_nano_${RESOLUTION}_vanilla_square_b${BATCH}"

mkdir -p logs "output/$RUN_NAME"

# One target per image in this dataset, so the Hungarian solve is a vectorised
# argmin. Without this, matcher.py makes group_detr (13) scipy calls per IMAGE,
# which is why throughput was flat in batch size.
echo "[0/3] patching matcher"
uv run python patch_matcher.py

if ! pgrep -f "sync_hf.py.*$RUN_NAME" >/dev/null 2>&1; then
  echo "[1/3] starting sync_hf daemon"
  nohup uv run python -u sync_hf.py --watch-dir "output/$RUN_NAME" --interval 120 \
    > "logs/sync_hf_$RUN_NAME.log" 2>&1 &
fi

echo "[2/3] training $RUN_NAME"
uv run python train_rfdetr.py nano \
  --square \
  --resolution "$RESOLUTION" \
  --aug-preset vanilla \
  --dataset-dir "$DATASET" \
  --batch-size "$BATCH" --effective-batch "$BATCH" \
  --num-workers "${NUM_WORKERS:-8}" \
  ${LR:+--lr "$LR"} ${LR_ENCODER:+--lr-encoder "$LR_ENCODER"} \
  --ema-update-interval "$EMA_INTERVAL" --ema-decay "$EMA_DECAY" \
  ${COMPILE:+--compile} \
  --epochs "$EPOCHS" \
  --eval-interval 5 \
  --run-name "$RUN_NAME" \
  --no-resume \
  ${WANDB_API_KEY:+--wandb}

echo "[3/3] exporting square graphs and scoring on real frames"
# The export shape must be THIS run's geometry: square in, square out.
for ckpt in "output/$RUN_NAME"/*.pth; do
  [ -e "$ckpt" ] || continue
  echo "=== $ckpt"
  uv run python - "$ckpt" "$RESOLUTION" <<'PY'
import sys, os
from rfdetr import RFDETRSegNano
ck, side = sys.argv[1], int(sys.argv[2])
out = os.path.join(os.path.dirname(ck), "onnx_" + os.path.basename(ck).replace(".pth", ""))
os.makedirs(out, exist_ok=True)
names = ["picanol", "colruyt", "leanflow"]
m = RFDETRSegNano(pretrain_weights=ck, num_classes=len(names))
m.export(output_dir=out, format="onnx", shape=(side, side),
         notes={"class_names": names, "variant": "nano", "square": True})
PY
done

for onnx in "output/$RUN_NAME"/onnx_*/*.onnx; do
  [ -e "$onnx" ] || continue
  uv run python eval_real_probe.py "$onnx" --probe real_probe \
    --json-out "${onnx%.onnx}_realprobe.json"
done

uv run python sync_hf.py --watch-dir "output/$RUN_NAME" --once
echo "VANILLA_RUN_DONE: output/$RUN_NAME"
