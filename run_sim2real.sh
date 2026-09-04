#!/usr/bin/env bash
# Train RF-DETR-Seg nano for REAL-WORLD cart classification, not for Isaac mAP.
#
# What this run changes against run_seg_nano_480.sh, and why. The previous
# configuration reached 0.9868 mAP on the Isaac validation split and classified
# a real leanflow as picanol on 296 of 297 recorded bag frames, while a YOLO
# baseline trained on the same renders scored 10/13 on those frames. Three
# differences are addressed here:
#
#   1. GEOMETRY. The Isaac camera renders 1280x800 at 90.5 deg horizontal /
#      64.5 deg vertical field of view; the D455 publishes 1280x720 at 90.8 /
#      59.4. The horizontal views already agree, so a vertical centre-crop to
#      16:9 aligns them (see reencode_dataset.py CROP_ASPECT). Training then
#      runs at 432x768, itself 16:9, so neither training nor inference stretches
#      the frame -- the old path squashed every real frame by 800/720 = 1.11x.
#
#   2. AUGMENTATION. --aug-preset sim2real restores Affine and widens the
#      photometric range; see SIM2REAL_AUG in train_rfdetr.py for why the
#      objection that removed geometric transforms does not apply to a detector
#      whose range comes from depth.
#
#   3. SCHEDULE. --epochs is honoured rather than truncated by an external
#      signal, so lr_drop lands at 75% of the budget and the run actually gets
#      its low-rate refinement phase. The old run was killed at epoch 20 of a
#      100-epoch schedule whose lr_drop was 75, so the rate never dropped.
#
# Selection is on the REAL probe, not on Isaac mAP: eval_real_probe.py scores
# every exported candidate on recorded frames. Read it together with the Isaac
# mAP printed by training -- the probe currently holds only leanflow bags, so
# it alone cannot rule out a constant predictor and the sim metric is what does.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/cart_segmentation_training}"
cd "$REPO_DIR"

RESOLUTION=${RESOLUTION:-432}       # short side; 432 x 1.7778 = 768, both /24
ASPECT=${ASPECT:-1.7778}            # 16:9, the real camera's aspect
EPOCHS=${EPOCHS:-60}
EVAL_INTERVAL=${EVAL_INTERVAL:-2}
AUG_PRESET=${AUG_PRESET:-sim2real}
DATASET=${DATASET:-_rfdetr_dataset_960x540}
RUN_NAME=${RUN_NAME:-seg_nano_${RESOLUTION}_${AUG_PRESET}}

mkdir -p logs "output/$RUN_NAME"

echo "[1/5] patching rfdetr"
python3 patch_rfdetr.py

# The FOV-matched dataset. Building it from the 16:10 archive costs one JPEG
# generation and a few minutes; rebuilding from the raw 1280x800 PNG shards
# gives one fewer resize and costs the 31 GB download. Either is correct -- the
# crop geometry is identical, because cropping 800 -> 720 rows and cropping
# 600 -> 540 rows remove the same fraction of the same field of view.
if [ ! -f "$DATASET/train/_annotations.coco.json" ]; then
    echo "[2/5] building $DATASET (16:9, FOV-matched to the D455)"
    if [ -f "_rfdetr_dataset/train/_annotations.coco.json" ]; then
        SRC=_rfdetr_dataset; TARGET_W=960
    elif [ -f "_rfdetr_dataset_960/train/_annotations.coco.json" ]; then
        SRC=_rfdetr_dataset_960; TARGET_W=960
    else
        echo "!!! no source dataset. Fetch one first:"
        echo "    python3 fetch_rgb_masks.py --out _rfdetr_dataset && python3 masks_to_coco.py --root _rfdetr_dataset"
        exit 1
    fi
    python3 reencode_dataset.py --src "$SRC" --dst "$DATASET" \
        --crop-aspect "$ASPECT" --target-w "$TARGET_W" --workers 4
else
    echo "[2/5] $DATASET already present"
fi

if ! pgrep -f "sync_hf.py.*$RUN_NAME" >/dev/null 2>&1; then
    echo "[3/5] starting sync_hf daemon"
    nohup python3 -u sync_hf.py --watch-dir "output/$RUN_NAME" --interval 60 \
        > "logs/sync_hf_$RUN_NAME.log" 2>&1 &
fi

echo "[4/5] training $RUN_NAME"
python3 train_rfdetr.py nano \
    --resolution "$RESOLUTION" \
    --aspect "$ASPECT" \
    --aug-preset "$AUG_PRESET" \
    --dataset-dir "$DATASET" \
    --batch-size 32 --effective-batch 32 \
    --eval-interval "$EVAL_INTERVAL" \
    --epochs "$EPOCHS" \
    --run-name "$RUN_NAME" \
    ${WANDB_API_KEY:+--wandb}

echo "[5/5] exporting and scoring on real frames"
# The export must use THIS run's geometry, so the shape is passed as an
# argument rather than read from the environment -- an unexported shell
# variable would silently fall back to a default and write a graph whose input
# size does not match the weights it came from.
LONG_SIDE=$(python3 -c "print(round($RESOLUTION * $ASPECT))")
for ckpt in "output/$RUN_NAME"/*.pth; do
    [ -e "$ckpt" ] || continue
    echo "=== $ckpt"
    python3 - "$ckpt" "$RESOLUTION" "$LONG_SIDE" <<'PY'
import sys, os
from rfdetr import RFDETRSegNano
ck, short_side, long_side = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = os.path.join(os.path.dirname(ck), "onnx_" + os.path.basename(ck).replace(".pth", ""))
os.makedirs(out, exist_ok=True)
names = ["picanol", "colruyt", "leanflow"]
m = RFDETRSegNano(pretrain_weights=ck, num_classes=len(names))
m.export(output_dir=out, format="onnx",
         shape=(short_side, long_side),
         notes={"class_names": names, "variant": "nano"})
PY
done
for onnx in "output/$RUN_NAME"/onnx_*/*.onnx; do
    [ -e "$onnx" ] || continue
    python3 eval_real_probe.py "$onnx" --probe real_probe \
        --json-out "${onnx%.onnx}_realprobe.json"
done
python3 sync_hf.py --watch-dir "output/$RUN_NAME" --once
echo "done: output/$RUN_NAME"
