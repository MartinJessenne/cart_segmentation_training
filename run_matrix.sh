#!/usr/bin/env bash
# Train the nine-model matrix: three variants at three input resolutions.
#
# One GPU, so the runs are sequential. Each is independently resumable --
# train_rfdetr.py finds last.ckpt in the run's output directory by itself -- so
# this script can be re-run after a crash and picks up where it stopped. A run
# that finished is identified by its training_summary.json, which is written
# only after model.train() returns.
#
# A failing run does not stop the matrix: eight usable models beat none, and the
# failure is in its own log.
set -u

VARIANTS="${VARIANTS:-nano small medium}"
RESOLUTIONS="${RESOLUTIONS:-360 480 600}"

mkdir -p logs

for variant in $VARIANTS; do
    for resolution in $RESOLUTIONS; do
        run="seg_${variant}_${resolution}"
        if [ -f "output/${run}/training_summary.json" ]; then
            echo "$(date +%H:%M:%S) $run: already finished, skipped"
            continue
        fi
        echo "$(date +%H:%M:%S) $run: starting"
        if uv run python train_rfdetr.py "$variant" \
                --resolution "$resolution" >> "logs/${run}.log" 2>&1; then
            echo "$(date +%H:%M:%S) $run: done"
        else
            echo "$(date +%H:%M:%S) $run: FAILED (see logs/${run}.log)"
        fi
    done
done

echo "$(date +%H:%M:%S) matrix complete"
