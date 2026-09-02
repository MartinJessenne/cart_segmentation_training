"""Measure the training cost of every cell in the nine-model matrix.

Committing to the matrix commits days of GPU time, and the cost is not linear in
anything that can be read off the configuration: pixels grow 2.78x from the
smallest tier to the largest, the medium variant carries 200 queries against
100, and every cell is GPU-bound rather than input-bound, so per-cell timing is
the only honest source.

Each cell is started through the real training script -- same dataset, same
augmentation, same batch geometry -- run past the warm-up into steady state,
timed over a fixed window of optimiser steps, then stopped. A cell that fails to
start, typically for want of memory at the heavy tiers, is reported as such
rather than silently skipped; finding that here costs two minutes, and finding
it during the matrix costs whatever had run before it.

Only the training epoch is measured. Validation runs once every eval_interval
epochs and is timed separately.
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time

VARIANTS = ("nano", "small", "medium")
RESOLUTIONS = (360, 480, 600)
# metrics.csv gains one row per 50 optimiser steps.
STEPS_PER_ROW = 50


def count_rows(path):
    """Rows of metrics.csv excluding its header, or 0 before it exists."""
    try:
        with open(path) as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def measure(variant, resolution, dataset_dir, batch_size, settle_rows,
            window_rows, timeout):
    """Steps per second for one cell, or a reason it produced none."""
    output_dir = f"_bench/{variant}_{resolution}"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    metrics = os.path.join(output_dir, "metrics.csv")
    log_path = os.path.join(output_dir, "run.log")

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["uv", "run", "python", "train_rfdetr.py", variant,
             "--resolution", str(resolution),
             "--dataset-dir", dataset_dir,
             "--batch-size", str(batch_size),
             "--epochs", "1", "--output-dir", output_dir, "--no-resume"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    try:
        # Startup covers weight download, the dataset build and the first
        # compiled kernels; timing anything inside it measures the wrong thing.
        deadline = time.time() + timeout
        while count_rows(metrics) < settle_rows:
            if proc.poll() is not None:
                return None, f"exited early (code {proc.returncode})"
            if time.time() > deadline:
                return None, f"no steady state within {timeout}s"
            time.sleep(5)

        start_rows = count_rows(metrics)
        start = time.time()
        target = start_rows + window_rows
        while count_rows(metrics) < target:
            if proc.poll() is not None:
                return None, f"exited during measurement (code {proc.returncode})"
            if time.time() > deadline:
                return None, f"window did not close within {timeout}s"
            time.sleep(5)

        elapsed = time.time() - start
        rows = count_rows(metrics) - start_rows
        return rows * STEPS_PER_ROW / elapsed, None
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset_960")
    # Comma-separated, so one axis can be swept while the others are pinned.
    # Sweeping --batch-sizes answers whether throughput is limited by the work
    # itself or by the cost of launching it: a small model on a large GPU issues
    # many tiny kernels, and if that is the limit then images per second rise
    # with the batch while the Python work per step stays the same.
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--resolutions", default=",".join(str(r) for r in RESOLUTIONS))
    ap.add_argument("--batch-sizes", default="16")
    # Rows to discard before timing, and rows to time over. One row is 50 steps.
    ap.add_argument("--settle-rows", type=int, default=2)
    ap.add_argument("--window-rows", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="epoch_cost.json")
    args = ap.parse_args()

    variants = args.variants.split(",")
    resolutions = [int(r) for r in args.resolutions.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    with open(os.path.join(args.dataset_dir, "train",
                           "_annotations.coco.json")) as fh:
        train_images = len(json.load(fh)["images"])
    print(f"{train_images} training images\n")

    results = {}
    for variant in variants:
        for resolution in resolutions:
            for batch_size in batch_sizes:
                cell = f"{variant}_{resolution}_b{batch_size}"
                steps_per_epoch = math.ceil(train_images / batch_size)
                print(f"{cell}: measuring ({steps_per_epoch} steps/epoch)",
                      flush=True)
                rate, failure = measure(variant, resolution, args.dataset_dir,
                                        batch_size, args.settle_rows,
                                        args.window_rows, args.timeout)
                if failure:
                    print(f"{cell}: FAILED -- {failure}", flush=True)
                    results[cell] = {"failed": failure}
                    continue
                images = rate * batch_size
                minutes = steps_per_epoch / rate / 60
                results[cell] = {"steps_per_second": round(rate, 3),
                                 "images_per_second": round(images, 1),
                                 "train_epoch_minutes": round(minutes, 2)}
                print(f"{cell}: {rate:.2f} steps/s, {images:.1f} img/s, "
                      f"{minutes:.1f} min per training epoch", flush=True)

    with open(args.out, "w") as fh:
        json.dump({"train_images": train_images, "cells": results}, fh, indent=2)

    usable = {k: v for k, v in results.items() if "train_epoch_minutes" in v}
    print(f"\n{len(usable)}/{len(results)} cells measured")
    if len(batch_sizes) > 1:
        # Throughput against batch size is the whole question: a flat curve
        # means the work itself is the limit and a bigger batch buys nothing.
        print("\nimages/s by batch size:")
        for cell, data in usable.items():
            print(f"  {cell:28s} {data['images_per_second']:7.1f} img/s")
    else:
        total = sum(v["train_epoch_minutes"] for v in usable.values())
        print(f"one epoch across the matrix: {total:.0f} min ({total/60:.1f} h)")
        for epochs in (25, 50, 100):
            print(f"  {epochs:3d} epochs, training only: {total*epochs/60:.0f} h")


if __name__ == "__main__":
    main()
