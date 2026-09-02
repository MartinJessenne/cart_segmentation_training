"""Train one RF-DETR segmentation variant on the industrial cart dataset.

The three variants differ by more than depth: their input resolution is baked
into their pretrained weights (positional_encoding_size = resolution /
patch_size), 312 for nano, 384 for small, 432 for medium. That resolution is
therefore not a knob, it is part of the variant -- and it carries most of the
accuracy/latency trade-off this script exists to measure.

Everything else is held identical across variants (same epochs, same effective
batch, same dataset, same seed), otherwise the comparison stops measuring the
variant and starts measuring the protocol.

The script runs through to ONNX and to the Hub in one process. That is not a
convenience: the training machine has no persistent storage, so a checkpoint
that has not left it does not exist. Splitting export and upload into separate
steps would reopen a window where a variant is trained but not delivered.
"""
import argparse
import glob
import json
import os
import time

import torch
from huggingface_hub import HfApi, create_repo
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

VARIANTS = {
    "nano": (RFDETRSegNano, 312),
    "small": (RFDETRSegSmall, 384),
    "medium": (RFDETRSegMedium, 432),
}

# What deserves to outlive the machine: the retained weights, the exported
# graph, the summaries and the logs. Per-epoch checkpoints are deliberately
# excluded; they are heavy and carry no conclusion.
PUBLISHED_PATTERNS = ["*.json", "*.txt", "*.onnx", "checkpoint_best*.pth"]

# Ordered by preference: the periodic checkpoint is the most recent state, the
# best-metric one is a fallback that costs at most the epochs since it was cut.
RESUME_PATTERNS = ["checkpoint.pth", "checkpoint_best_total.pth",
                   "checkpoint_best*.pth"]


def class_names_from_dataset(dataset_dir):
    """The class order as RF-DETR derives it, read from COCO rather than retyped.

    RF-DETR builds its indices with
        {category["id"]: label for label, category in enumerate(kept)}
    over the train split's categories sorted by id. Sorting by that same id
    here reproduces the model's internal order exactly. This list goes into the
    ONNX metadata: it is the only place where the class order travels with the
    file, instead of being retyped in the C++ detector where a permutation
    passes every metric in silence.
    """
    path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    with open(path) as fh:
        coco = json.load(fh)
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def find_resume_checkpoint(output_dir):
    """Latest usable checkpoint in output_dir, or None on a fresh run."""
    for pattern in RESUME_PATTERNS:
        found = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if found:
            return found[0]
    return None


def publish(output_dir, repo, variant):
    """Upload the useful part of output_dir under seg_<variant>/ of the repo."""
    api = HfApi()
    create_repo(repo, exist_ok=True, repo_type="model", private=True)
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=f"seg_{variant}",
        repo_id=repo,
        repo_type="model",
        allow_patterns=PUBLISHED_PATTERNS,
    )
    print(f"published -> https://huggingface.co/{repo}/tree/main/seg_{variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset")
    # 100 is RF-DETR's own default and the value its schedule is built around:
    # lr_drop is 100 as well, so the learning-rate step lands at the end of the
    # budget. A shorter run never reaches it and forfeits its refinement phase.
    ap.add_argument("--epochs", type=int, default=100)
    # Effective batch = batch_size * grad_accum_steps. RF-DETR is tuned for 16
    # when fine-tuning; the card has 96 GB, so the 16 fit in a single batch and
    # grad_accum stays at 1 for all three variants.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    # 100 epochs is a ceiling, not a fixed duration: early stopping hands back
    # control once validation stops improving. The patience and delta below are
    # RF-DETR's own defaults; only the switch is off by default.
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=0.001)
    # Comparing variants only means something if everything else is held fixed.
    # The seed is part of that, and RF-DETR leaves it unset on its own.
    ap.add_argument("--seed", type=int, default=42)
    # The training machine has no persistent storage and goes down regularly; a
    # 100-epoch run outlasts its observed uptime. With no explicit path, pick up
    # from the newest checkpoint already in the output directory.
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--hf-repo", default="UItraviolet/cart_segmentation_rfdetr")
    args = ap.parse_args()

    cls, resolution = VARIANTS[args.variant]
    output_dir = args.output_dir or f"output/seg_{args.variant}"
    os.makedirs(output_dir, exist_ok=True)

    classes = class_names_from_dataset(args.dataset_dir)
    resume_from = args.resume_from
    if resume_from is None and not args.no_resume:
        resume_from = find_resume_checkpoint(output_dir)

    print(f"variant    : {args.variant}  (resolution {resolution})")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability()}")
    print(f"dataset    : {args.dataset_dir}")
    print(f"classes    : {classes}")
    print(f"resume     : {resume_from or 'fresh run'}")
    print(f"output     : {output_dir}\n")

    start = time.time()
    model = cls()
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=output_dir,
        early_stopping=True,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        seed=args.seed,
        class_names=classes,
        resume=resume_from,
    )
    duration = time.time() - start

    summary = {
        "variant": args.variant,
        "resolution": resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "resumed_from": resume_from,
        "training_seconds": round(duration, 1),
        "gpu": torch.cuda.get_device_name(0),
        "class_names": classes,
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\ntrained in {duration/60:.1f} min")

    # `notes` is the only metadata channel RF-DETR exposes at export time: its
    # contents are serialised as JSON under the ONNX file's `rfdetr_notes` key.
    # Putting the class order there makes the graph self-describing.
    onnx_path = model.export(
        output_dir=output_dir,
        format="onnx",
        notes={"class_names": classes, "variant": args.variant,
               "resolution": resolution},
    )
    print(f"exported -> {onnx_path}")

    publish(output_dir, args.hf_repo, args.variant)
    print(f"\ndone in {(time.time() - start)/60:.1f} min -> {output_dir}")


if __name__ == "__main__":
    main()
