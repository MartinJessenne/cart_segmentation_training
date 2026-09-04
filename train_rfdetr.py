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

Multi-scale is off, and turning it on is now a mistake rather than a choice.
Its scales run from -3 to +4 window steps around the nominal resolution, so they
reach above it; the dataset is stored at 960x600, the largest size the resize
pipeline ever requests without it, and anything above that would be upsampled
from storage. It also spans a range wide enough that neighbouring resolution
tiers would overlap and the comparison would stop measuring resolution.

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
# and negligible against them. Kornia cannot sample a per-image std and takes
# the upper bound as a fixed value, so the bound is the noise level every image
# actually receives: 0.03 is a realistic warehouse signal-to-noise ratio, where
# 0.05 would apply the worst case to every frame.
WAREHOUSE_AUG = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2,
                    "saturation": 0.2, "hue": 0.1, "p": 0.5},
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": [0.01, 0.03], "p": 0.3},
}

# Sim-to-real augmentation: everything WAREHOUSE_AUG has, plus the geometric
# transform it omits and wider photometric range.
#
# Affine is present here and absent above because the objection that removed it
# does not apply to this detector. The network emits a MASK and a CLASS; metric
# range is recovered downstream by back-projecting the masked depth through the
# camera intrinsics (nxtbot_cart_pose core/estimator.cpp), never from where the
# cart sits in the frame. So rotating or shearing the image cannot corrupt a
# range cue, because no range cue is read from image position. What it does buy
# is viewpoint diversity, which is the measured failure: the detector scores
# 18/18 on held-out Isaac renders and 0/13 on real RealSense frames of a cart
# it has to recognise, while a YOLO baseline trained on the same renders with
# mosaic and random affine scores 10/13 on those same frames.
#
# VerticalFlip stays out on its own merits: the camera is rigidly mounted
# looking forward, so a frame with the floor at the top is a view the robot
# cannot produce, and spending capacity on it buys nothing.
#
# rotate is capped at 12 deg because the mount is rigid and the deck is flat --
# the only real roll comes from suspension travel and floor slope, a few
# degrees. The cap keeps the distribution near what the robot can actually see
# while still breaking the pixel-exact vertical alignment every render shares.
#
# GaussNoise: Kornia cannot sample a per-image std and applies the UPPER bound
# of std_range to every image it touches, so the upper bound is the noise level
# actually delivered, and p is what fraction of images receive it. 0.04 sits
# between the clean-render floor and the worst RealSense frames measured on the
# bags; raising the bound further would apply the worst case universally.
SIM2REAL_AUG = {
    "HorizontalFlip": {"p": 0.5},
    "Affine": {
        "scale": (0.7, 1.4),
        "translate_percent": (-0.12, 0.12),
        "rotate": (-12, 12),
        "shear": (-6, 6),
        "p": 0.7,
    },
    "ColorJitter": {"brightness": 0.4, "contrast": 0.4,
                    "saturation": 0.4, "hue": 0.15, "p": 0.8},
    "GaussianBlur": {"blur_limit": 7, "p": 0.4},
    "GaussNoise": {"std_range": [0.01, 0.04], "p": 0.5},
}

AUG_PRESETS = {"warehouse": WAREHOUSE_AUG, "sim2real": SIM2REAL_AUG}

# What deserves to outlive the machine: the retained weights, the exported
# graph, the summaries and the logs. last.ckpt is deliberately absent -- at
# 534 MB against 133 for the stripped weights, it carries optimizer and
# scheduler state that matters only to a resume on this machine.
PUBLISHED_PATTERNS = ["*.json", "*.txt", "*.csv", "*.onnx", "*best*.pth"]

# Ordered by preference. last.ckpt is Lightning's own checkpoint, rewritten
# every epoch with optimizer and scheduler state, so resuming from it continues
# the run exactly. The best-metric files are stripped to {model, args, epoch}:
# resuming from one restarts the optimizer, losing the momentum and the
# schedule position, so they are a fallback for when last.ckpt is missing.
RESUME_PATTERNS = ["last.ckpt", "checkpoint_best_total.pth",
                   "checkpoint_best_ema.pth"]


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
    # 960x600 JPEG, produced by reencode_dataset.py. That is the largest input
    # any tier requests, so nothing upsamples, and it decodes 6.5x faster than
    # the 1280x800 PNG source -- which is what the two dataloader workers were
    # spending the whole run on.
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset_960")
    # Short side of the 16:10 input. The long side follows at 1.6x. Both must be
    # divisible by patch_size * num_windows (24 for small and medium), which
    # 360, 480 and 600 satisfy, giving 360x576, 480x768 and 600x960.
    ap.add_argument("--resolution", type=int, default=None,
                    help="short side; defaults to the variant's nominal value")
    # 100 is RF-DETR's own default and the value its schedule is built around:
    # lr_drop is 100 too, so the learning-rate step lands at the end of the
    # budget and a shorter run forfeits its refinement phase.
    ap.add_argument("--epochs", type=int, default=100)
    # Micro-batch per forward pass. The device holds far more than this -- the
    # probe reports 31 for nano at 360x576 on 96 GB -- but capacity is not the
    # criterion. batch_size="auto" cannot be used here: it returns the largest
    # micro-batch that fits and accumulates only when that falls below the
    # target, so it resolves to an effective batch of 31 at 360x576 and to
    # something else again at 600x960. The nine runs would then differ in the
    # one quantity the comparison has to hold fixed.
    ap.add_argument("--batch-size", type=int, default=16)
    # What the optimiser sees per step, and the invariant of the matrix.
    # RF-DETR is tuned for 16 when fine-tuning; grad_accum_steps is derived from
    # this and --batch-size, so lowering the micro-batch at a heavier tier
    # raises the accumulation and leaves the product untouched.
    ap.add_argument("--effective-batch", type=int, default=16)
    # Off by default: the multi-scale range is wide enough that adjacent
    # resolution tiers would overlap and the comparison would lose its meaning.
    ap.add_argument("--multi-scale", action="store_true")
    # Option B of the resize pipeline: resize the short side to 400/500/600,
    # take a square RandomSizedCrop, and resize that crop to the target. Square
    # in, square out, so it zooms without distorting aspect. It is the recipe's
    # only scale augmentation, and the deployed camera meets carts from 0.3 m to
    # several metres, so it stays on -- as it is in RF-DETR itself.
    #
    # It makes a batch mix 360x360 and 360x576 samples. The collator pads the
    # images to the batch maximum and leaves the masks alone; patch_rfdetr.py
    # supplies the missing mask padding and must be applied before training.
    ap.add_argument("--no-scale-jitter", dest="scale_jitter",
                    action="store_false",
                    help="disable the resize-crop-resize branch")
    ap.add_argument("--aug-backend", default="kornia", choices=("kornia", "cpu"))
    # Only the kornia backend's supported set is usable here (HorizontalFlip,
    # VerticalFlip, Rotate, Affine, ColorJitter, RandomBrightnessContrast,
    # GaussianBlur, GaussNoise); anything else is silently not applied. Adding
    # grayscale, motion blur or JPEG artefacts means --aug-backend cpu, which
    # moves augmentation off the GPU and onto the two dataloader workers.
    ap.add_argument("--aug-preset", default="warehouse", choices=sorted(AUG_PRESETS),
                    help="warehouse = photometric only; sim2real = adds Affine "
                         "and widens the photometric range")
    # Long side as a multiple of the short side. 1.6 (16:10) matches the Isaac
    # renders as they come out of the simulator; 1.7778 (16:9) matches the D455
    # colour stream the robot actually publishes, and is the right choice
    # whenever the dataset has been FOV-matched by reencode_dataset.py
    # --crop-aspect. Both sides must stay divisible by 24 (patch_size x
    # num_windows), which is checked below rather than left to fail inside the
    # backbone: 432x768 and 480x768 both satisfy it.
    ap.add_argument("--aspect", type=float, default=1.6,
                    help="long side / short side; 1.6 = 16:10, 1.7778 = 16:9")
    # 100 epochs is a ceiling, not a fixed duration: early stopping hands back
    # control once validation stops improving. The patience and delta below are
    # RF-DETR's own defaults; only the switch is off by default.
    # Given in epochs, converted to validation events below. RFDETREarlyStopping
    # only ever runs on eval epochs, so its own counter is in validations: left
    # at 10 alongside --eval-interval 5 it would tolerate 50 stagnant epochs and
    # no run would stop early at all.
    ap.add_argument("--patience", type=int, default=25,
                    help="epochs without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=0.001)
    # Validation costs more than training here. The mAP metric RLE-encodes every
    # predicted mask on a single CPU core -- num_select is 100, over 2470 images
    # -- and measured at 360x576 that ran about 25 minutes against 8 for the
    # training epoch. Validating every fifth epoch amortises it; Lightning skips
    # the loop entirely on the others, and RF-DETR forces one on the final epoch
    # so a run always ends on a measured checkpoint.
    ap.add_argument("--eval-interval", type=int, default=5,
                    help="epochs between validations")
    # Comparing variants only means something if everything else is held fixed.
    # The seed is part of that, and RF-DETR leaves it unset on its own.
    ap.add_argument("--seed", type=int, default=42)
    # Two rolling checkpoints and nothing else. RF-DETR registers three
    # callbacks: a "last" one written every epoch with save_top_k=1, a best-mAP
    # one, and an archive `checkpoint_{epoch}` with save_top_k=-1 that keeps
    # every file it ever writes. The archive's retention is hardcoded, so the
    # only way to silence it is an interval longer than the run -- hence the
    # default below. Setting the interval to 1 does the opposite of what it
    # looks like: it makes the archive fire every epoch (100 files, ~40 GB) and
    # it also suppresses the "last" callback, which is skipped exactly when the
    # interval equals 1.
    ap.add_argument("--checkpoint-interval", type=int, default=None,
                    help="epochs between archive checkpoints; "
                         "defaults to more than the run, keeping only last+best")
    # The detection head is rebuilt for 3 classes and, whenever --resolution
    # differs from the variant's nominal value, the pretrained position table is
    # interpolated onto a new grid. Both make the first steps noisy, and DETR
    # models are fragile there. RF-DETR ships 0.0, which starts at full rate.
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    # RF-DETR ships lr_drop equal to the epoch budget, so the step lands on the
    # last epoch and the rate is constant throughout -- the refinement phase
    # never happens. Dropping at three quarters of the budget gives the mask
    # boundaries a low-rate phase to settle in, which is what the pose stage
    # reads.
    ap.add_argument("--lr-drop", type=int, default=None,
                    help="epoch of the learning-rate step; defaults to 75%% of epochs")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--hf-repo", default="UItraviolet/cart_segmentation_rfdetr")
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="Enable Weights & Biases cloud telemetry")
    ap.add_argument("--project", default="cart_segmentation",
                    help="WandB project name")
    ap.add_argument("--run-name", default=None,
                    help="WandB run name")
    args = ap.parse_args()

    cls, nominal = VARIANTS[args.variant]
    resolution = args.resolution or nominal
    lr_drop = args.lr_drop or max(1, round(args.epochs * 0.75))
    checkpoint_interval = args.checkpoint_interval or (args.epochs + 1)
    patience_evals = max(1, round(args.patience / args.eval_interval))
    if args.effective_batch % args.batch_size:
        ap.error(f"--effective-batch {args.effective_batch} is not a multiple of "
                 f"--batch-size {args.batch_size}: the optimiser would see a "
                 "different batch from the one asked for")
    grad_accum_steps = args.effective_batch // args.batch_size
    long_side = round(resolution * args.aspect)
    for name, side in (("short", resolution), ("long", long_side)):
        if side % 24:
            ap.error(f"{name} side {side} is not divisible by 24 "
                     f"(patch_size x num_windows); 432x768 and 480x768 are valid")
    aug_config = AUG_PRESETS[args.aug_preset]
    # The preset and aspect are part of the identity: output_dir doubles as the
    # resume source, so two runs that differ only in augmentation or aspect
    # would otherwise share a directory and silently resume from each other.
    run_name = args.run_name or (
        f"seg_{args.variant}_{resolution}_{args.aug_preset}_{long_side}")
    output_dir = args.output_dir or os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    classes = class_names_from_dataset(args.dataset_dir)
    resume_from = args.resume_from
    if resume_from is None and not args.no_resume:
        resume_from = find_resume_checkpoint(output_dir)

    print(f"variant    : {args.variant}  (nominal {nominal})")
    print(f"input      : {resolution}x{long_side}  (aspect {args.aspect:.4f})")
    print(f"augment    : {args.aug_preset} on {args.aug_backend}")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability()}")
    print(f"dataset    : {args.dataset_dir}")
    print(f"classes    : {classes}")
    print(f"schedule   : {args.epochs} epochs, warmup {args.warmup_epochs}, "
          f"lr_drop {lr_drop}, archive every {checkpoint_interval}")
    print(f"batch      : {args.batch_size} micro x {grad_accum_steps} accum "
          f"= {args.effective_batch} effective")
    print(f"validation : every {args.eval_interval} epochs, EMA only, "
          f"patience {args.patience} epochs ({patience_evals} evals)")
    print(f"resume     : {resume_from or 'fresh run'}")
    print(f"output     : {output_dir}\n")

    start = time.time()
    model = cls(resolution=resolution, num_classes=len(classes))
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=grad_accum_steps,
        output_dir=output_dir,
        resolution=resolution,
        square_resize_div_64=False,
        multi_scale=args.multi_scale,
        scale_jitter=args.scale_jitter,
        aug_config=aug_config,
        augmentation_backend=args.aug_backend,
        checkpoint_interval=checkpoint_interval,
        warmup_epochs=args.warmup_epochs,
        # The scheduler's own interface. The flat lr_drop= argument still works
        # but is deprecated, and a version that drops it would take the
        # refinement phase with it without failing.
        lr_scheduler_kwargs={"lr_drop": lr_drop},
        eval_interval=args.eval_interval,
        # Validation forwards through the EMA weights only. Left False, RF-DETR
        # runs a second independent forward pass and a second full round of RLE
        # mask encoding for the base model, exactly doubling the phase that
        # already dominates the run. The EMA weights are the ones exported, so
        # the base model's metrics answer no question being asked here.
        eval_ema_only=True,
        # Nothing reads the validation loss: early stopping and best-checkpoint
        # selection both follow the EMA mAP, and the "step" schedule ignores its
        # monitor. Computing it runs the Hungarian matcher over every validation
        # mask for a number that is only ever printed.
        compute_val_loss=False,
        early_stopping=True,
        early_stopping_patience=patience_evals,
        early_stopping_min_delta=args.min_delta,
        # eval_ema_only never logs the base metric, so the default
        # max(regular, ema) comparison would read a key that is not there.
        early_stopping_use_ema=True,
        seed=args.seed,
        class_names=classes,
        resume=resume_from,
        wandb=args.wandb,
        project=args.project,
        run=run_name,
    )
    duration = time.time() - start

    summary = {
        "variant": args.variant,
        "resolution": resolution,
        "input_hw": [resolution, long_side],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch": args.effective_batch,
        "multi_scale": args.multi_scale,
        "scale_jitter": args.scale_jitter,
        "augmentation": aug_config,
        "aug_preset": args.aug_preset,
        "aspect": args.aspect,
        "augmentation_backend": args.aug_backend,
        "warmup_epochs": args.warmup_epochs,
        "lr_drop": lr_drop,
        "checkpoint_interval": checkpoint_interval,
        "eval_interval": args.eval_interval,
        "eval_ema_only": True,
        "patience_epochs": args.patience,
        "patience_evals": patience_evals,
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
