"""Train one RF-DETR segmentation variant at one input resolution.

Two axes are being measured, and they are independent. The *variant* fixes
depth, patch size and window count; the *resolution* fixes how many pixels the
network sees. RF-DETR ships a nominal resolution per variant, but it is not
baked in: positional_encoding_size is derived from whatever resolution is asked
for, and the pretrained position table is bicubically interpolated onto the
requested patch grid at every forward pass. Overriding it is therefore
supported, and the two axes can be crossed.

Geometry is 16:10 throughout, matching the D455 colour stream at 1280x800.
`square_resize_div_64=False` selects the aspect-preserving resize, so
`--resolution` sets the *short* side and the long side follows at 1.6x: 360
gives 360x576, 480 gives 480x768, 600 gives 600x960. Training the detector on
squashed square crops while deploying on 16:10 frames would compress the
horizontal axis 1.6x more than the vertical, and the horizontal axis is the one
carrying bearing, which is the dominant error term measured on the robot.

Multi-scale is off by default. Its scale range spans +/-5 window steps around
the nominal resolution, which is wide enough that neighbouring resolution tiers
would overlap heavily and the comparison would stop measuring resolution.

The script runs through to ONNX and to the Hub in one process. The training
machine has no persistent storage, so a checkpoint that has not left it does
not exist.
"""
import argparse
import glob
import json
import os
import time

import torch
from huggingface_hub import HfApi, create_repo
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

# Nominal resolution per variant, used when --resolution is not given.
VARIANTS = {
    "nano": (RFDETRSegNano, 312),
    "small": (RFDETRSegSmall, 384),
    "medium": (RFDETRSegMedium, 432),
}

# Warehouse augmentation: RF-DETR's AUG_AGGRESSIVE without its three geometric
# transforms, plus the noise and blur from AUG_INDUSTRIAL.
#
# VerticalFlip, Rotate and Affine are deliberately absent. The camera is rigidly
# mounted 30.4 cm above the deck with a calibrated orientation, and the pose
# stage converts image position to metric position through that extrinsic.
# Rotating or shearing the frame would both invent geometry the robot never sees
# and break the link between the image's vertical axis and elevation, which is
# what carries range. A vertical flip would put the floor at the top.
#
# ColorJitter spans the illuminant range of a warehouse: sodium and metal-halide
# near 3000 K, LED near 5000 K, daylight through the doors near 6500 K. The
# amplitude is safe because cart identity rests on silhouette, not colour.
#
# GaussNoise models the two dominant sensor terms, photon shot noise and read
# noise, both Gaussian. Uniform noise would model quantisation, which is one LSB
# and negligible against them.
WAREHOUSE_AUG = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2,
                    "saturation": 0.2, "hue": 0.1, "p": 0.5},
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": [0.01, 0.05], "p": 0.3},
}

# What deserves to outlive the machine: the retained weights, the exported
# graph, the summaries and the logs. Per-epoch checkpoints are deliberately
# excluded; they are heavy and carry no conclusion.
PUBLISHED_PATTERNS = ["*.json", "*.txt", "*.csv", "*.onnx", "checkpoint_best*.pth"]

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


def publish(output_dir, repo, run_name):
    """Upload the useful part of output_dir under <run_name>/ of the repo."""
    api = HfApi()
    create_repo(repo, exist_ok=True, repo_type="model", private=True)
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=run_name,
        repo_id=repo,
        repo_type="model",
        allow_patterns=PUBLISHED_PATTERNS,
    )
    print(f"published -> https://huggingface.co/{repo}/tree/main/{run_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset")
    # Short side of the 16:10 input. The long side follows at 1.6x. Both must be
    # divisible by patch_size * num_windows (24 for small and medium), which
    # 360, 480 and 600 satisfy, giving 360x576, 480x768 and 600x960.
    ap.add_argument("--resolution", type=int, default=None,
                    help="short side; defaults to the variant's nominal value")
    # 100 is RF-DETR's own default and the value its schedule is built around:
    # lr_drop is 100 too, so the learning-rate step lands at the end of the
    # budget and a shorter run forfeits its refinement phase.
    ap.add_argument("--epochs", type=int, default=100)
    # Effective batch = batch_size * grad_accum_steps. RF-DETR is tuned for 16
    # when fine-tuning. At the higher resolution tiers the activations may not
    # fit in one batch, in which case lower batch-size and raise grad-accum to
    # hold the product at 16.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    # Off by default: the multi-scale range is wide enough that adjacent
    # resolution tiers would overlap and the comparison would lose its meaning.
    ap.add_argument("--multi-scale", action="store_true")
    # 100 epochs is a ceiling, not a fixed duration: early stopping hands back
    # control once validation stops improving. The patience and delta below are
    # RF-DETR's own defaults; only the switch is off by default.
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=0.001)
    # Comparing variants only means something if everything else is held fixed.
    # The seed is part of that, and RF-DETR leaves it unset on its own.
    ap.add_argument("--seed", type=int, default=42)
    # The machine goes down regularly and a 100-epoch run outlasts its observed
    # uptime. RF-DETR's own default writes a checkpoint every 10 epochs, which
    # would put up to 10 epochs at risk; 1 makes a crash cost one epoch.
    ap.add_argument("--checkpoint-interval", type=int, default=1)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--hf-repo", default="UItraviolet/cart_segmentation_rfdetr")
    args = ap.parse_args()

    cls, nominal = VARIANTS[args.variant]
    resolution = args.resolution or nominal
    long_side = round(resolution * 1.6)
    run_name = f"seg_{args.variant}_{resolution}"
    output_dir = args.output_dir or os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    classes = class_names_from_dataset(args.dataset_dir)
    resume_from = args.resume_from
    if resume_from is None and not args.no_resume:
        resume_from = find_resume_checkpoint(output_dir)

    print(f"variant    : {args.variant}  (nominal {nominal})")
    print(f"input      : {resolution}x{long_side}  (16:10, short side {resolution})")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability()}")
    print(f"dataset    : {args.dataset_dir}")
    print(f"classes    : {classes}")
    print(f"resume     : {resume_from or 'fresh run'}")
    print(f"output     : {output_dir}\n")

    start = time.time()
    model = cls(resolution=resolution, num_classes=len(classes))
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=output_dir,
        resolution=resolution,
        square_resize_div_64=False,
        multi_scale=args.multi_scale,
        aug_config=WAREHOUSE_AUG,
        augmentation_backend="kornia",
        checkpoint_interval=args.checkpoint_interval,
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
        "input_hw": [resolution, long_side],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "multi_scale": args.multi_scale,
        "augmentation": WAREHOUSE_AUG,
        "augmentation_backend": "kornia",
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
    # Putting the class order and the input geometry there makes the graph
    # self-describing for the C++ detector that consumes it.
    onnx_path = model.export(
        output_dir=output_dir,
        format="onnx",
        shape=(resolution, long_side),
        notes={"class_names": classes, "variant": args.variant,
               "resolution": resolution, "input_hw": [resolution, long_side]},
    )
    print(f"exported -> {onnx_path}")

    publish(output_dir, args.hf_repo, run_name)
    print(f"\ndone in {(time.time() - start)/60:.1f} min -> {output_dir}")


if __name__ == "__main__":
    main()
