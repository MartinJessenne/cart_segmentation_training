#!/usr/bin/env bash
# Train the nine-model matrix: three variants at three input resolutions.
#
# Runs go three at a time under CUDA MPS. A single run leaves the GPU at 39% of
# its power limit and cannot be made to fill it: roughly two thirds of a step is
# single-threaded Python in the main process -- losses, optimizer, Hungarian
# matcher, augmentation -- and a larger batch does not help, throughput being
# flat from 16 to 64. Concurrency is the only lever, and it needs MPS: without
# the daemon the driver time-slices the CUDA contexts instead of overlapping
# their kernels, and two concurrent runs came out slower than one alone (36
# images/s against 53). Starting MPS is therefore not an optimisation but a
# correctness condition for this script; a resume that skipped it would run
# slower than the sequential version it replaced, and say nothing.
#
# Groups mix one cell of each resolution rather than grouping by variant. Six of
# the lightest cell already occupy 87 GB of 96, so memory bounds concurrency
# well before throughput does, and the three 600x960 cells together would exceed
# the device. The heaviest group as arranged below peaks at 76.5 GB and gains
# 1.66x over running its three cells in sequence. Under MPS an out-of-memory
# takes down every client sharing the device, so the margin is deliberate.
#
# Each run is independently resumable -- train_rfdetr.py finds last.ckpt on its
# own -- so re-running this script after a crash continues where it stopped. A
# finished run is identified by training_summary.json, written only after
# training returns. A failing run does not stop the matrix.
set -u

GROUPS=(
    "nano:600 small:480 medium:360"
    "small:600 medium:480 nano:360"
    "medium:600 nano:480 small:360"
)

start_mps() {
    if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "$(date +%H:%M:%S) MPS already running"
        return 0
    fi
    export CUDA_VISIBLE_DEVICES=0
    if nvidia-cuda-mps-control -d; then
        sleep 3
        echo "$(date +%H:%M:%S) MPS started"
        return 0
    fi
    echo "$(date +%H:%M:%S) ABORT: MPS failed to start. Without it, three" \
         "concurrent runs are slower than one at a time."
    return 1
}

start_mps || exit 1
mkdir -p logs

for group in "${GROUPS[@]}"; do
    pids=()
    names=()
    for cell in $group; do
        variant="${cell%%:*}"
        resolution="${cell##*:}"
        run="seg_${variant}_${resolution}"
        if [ -f "output/${run}/training_summary.json" ]; then
            echo "$(date +%H:%M:%S) $run: already finished, skipped"
            continue
        fi
        uv run python train_rfdetr.py "$variant" --resolution "$resolution" \
            >> "logs/${run}.log" 2>&1 &
        pids+=("$!")
        names+=("$run")
        echo "$(date +%H:%M:%S) $run: started (pid $!)"
    done

    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "$(date +%H:%M:%S) ${names[$i]}: done"
        else
            echo "$(date +%H:%M:%S) ${names[$i]}: FAILED (see logs/${names[$i]}.log)"
        fi
    done
    echo "$(date +%H:%M:%S) group complete"
done

echo "$(date +%H:%M:%S) matrix complete"
