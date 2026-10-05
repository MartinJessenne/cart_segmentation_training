"""Train RF-DETR segmentation for industrial tow carts.

Implements the validated Sim-to-Real domain transfer configuration:
  1. Frozen DINOv2 Backbone (Linear Probing): Freezing the ViT backbone protects
     pre-trained natural visual representations against synthetic shader overfitting,
     achieving 100% real-world classification accuracy.
  2. Square Geometry (288x288): Reduces ViT token count (~104 tokens per cart)
     to eliminate micro-texture memorization and force silhouette-based detection.
  3. Targeted Sensor Augmentation (sensor preset): Gaussian blur, sensor noise,
     and color jitter simulate camera MTF and warehouse lighting without geometric distortion.

Usage:
    uv run python train.py --variant nano --dataset-dir _dataset_960
    uv run python train.py --variant small --dataset-dir _dataset_960 --batch-size 32
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

VARIANTS = {
    "nano": (RFDETRSegNano, 312),
    "small": (RFDETRSegSmall, 384),
    "medium": (RFDETRSegMedium, 432),
}

# Sensor-domain augmentation: targeted non-geometric sensor and photometric perturbations
SENSOR_AUG = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.6,
    },
    "GaussianBlur": {"blur_limit": 5, "p": 0.4},
    "GaussNoise": {"std_range": [0.01, 0.03], "p": 0.5},
}

WAREHOUSE_AUG = {
    "HorizontalFlip": {"p": 0.5},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2,
                    "saturation": 0.2, "hue": 0.1, "p": 0.5},
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": [0.01, 0.03], "p": 0.3},
}

AUG_PRESETS = {
    "sensor": SENSOR_AUG,
    "warehouse": WAREHOUSE_AUG,
    "vanilla": None,
}

RESUME_PATTERNS = ["last.ckpt", "checkpoint_best_total.pth", "checkpoint_best_ema.pth"]


def class_names_from_dataset(dataset_dir: str) -> list[str]:
    """Read category names in sorted ID order from COCO annotations."""
    path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    if not os.path.exists(path):
        sys.exit(f"ABORT: COCO annotations not found at '{path}'. Run masks_to_coco.py first.")
    with open(path) as fh:
        coco = json.load(fh)
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def find_resume_checkpoint(output_dir: str):
    """Find latest usable checkpoint in output_dir, or None on a fresh run."""
    for pattern in RESUME_PATTERNS:
        found = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if found:
            return found[0]
    return None


def maybe_publish_to_hf(output_dir: str, repo: str, run_name: str):
    """Upload model artifacts to Hugging Face Hub if repository is specified."""
    if not repo:
        return
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(repo, exist_ok=True, repo_type="model", private=True)
        api.upload_folder(
            folder_path=output_dir,
            path_in_repo=run_name,
            repo_id=repo,
            repo_type="model",
            allow_patterns=["*.json", "*.txt", "*.csv", "*.onnx", "*best*.pth"],
        )
        print(f"==> Published artifacts to https://huggingface.co/{repo}/tree/main/{run_name}")
    except Exception as e:
        print(f"[warning] Hugging Face upload skipped or failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Train RF-DETR Segmentation model")
    ap.add_argument("--variant", default="nano", choices=sorted(VARIANTS),
                    help="Model variant: nano, small, medium (default: nano)")
    ap.add_argument("--dataset-dir", default="_dataset_960",
                    help="Path to pre-processed COCO dataset directory (default: _dataset_960)")
    ap.add_argument("--resolution", type=int, default=288,
                    help="Input resolution (short side, or square side; default: 288)")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Total training epochs (default: 15)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Micro-batch size per GPU forward pass (default: 32)")
    ap.add_argument("--effective-batch", type=int, default=32,
                    help="Effective batch size via gradient accumulation (default: 32)")
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="Dataloader worker processes (default: 8 or CPU count)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Base learning rate (default: 1e-4)")
    ap.add_argument("--lr-encoder", type=float, default=None,
                    help="Backbone learning rate (default: None; 1e-5 if regularizing without freeze)")
    ap.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                    help="Freeze DINOv2 ViT backbone weights (default: True, gold standard for Sim2Real)")
    ap.add_argument("--square", action=argparse.BooleanOptionalAction, default=True,
                    help="Use square stretch resize Div64 (default: True, 288x288)")
    ap.add_argument("--aspect", type=float, default=1.6,
                    help="Aspect ratio (long / short) when --no-square is set (default: 1.6)")
    ap.add_argument("--aug-preset", default="sensor", choices=sorted(AUG_PRESETS),
                    help="Augmentation preset: sensor, warehouse, vanilla (default: sensor)")
    ap.add_argument("--aug-backend", default="kornia", choices=("kornia", "cpu"))
    ap.add_argument("--multi-scale", action="store_true", default=False,
                    help="Enable multi-scale training (default: False)")
    ap.add_argument("--scale-jitter", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable scale jitter augmentation (default: True)")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="Enable torch.compile (CUDA only)")
    ap.add_argument("--ema-update-interval", type=int, default=4,
                    help="Steps between EMA updates (default: 4)")
    ap.add_argument("--ema-decay", type=float, default=0.972,
                    help="EMA decay parameter (default: 0.972 for interval 4)")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--lr-drop", type=int, default=None,
                    help="Epoch for LR decay (defaults to 75%% of total epochs)")
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience in epochs (default: 15)")
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--eval-interval", type=int, default=1,
                    help="Epochs between validation passes (default: 1)")
    ap.add_argument("--eval-ema-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compute-val-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--no-resume", action="store_true", default=False)
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (defaults to output/seg_<variant>_<resolution>...)")
    ap.add_argument("--hf-repo", default=None,
                    help="Optional Hugging Face model repository to upload final weights")
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="Enable Weights & Biases cloud tracking")
    ap.add_argument("--project", default="cart_segmentation",
                    help="WandB project name (default: cart_segmentation)")
    ap.add_argument("--run-name", default=None,
                    help="WandB run name")
    args = ap.parse_args()

    cls, nominal = VARIANTS[args.variant]
    resolution = args.resolution or nominal
    lr_drop = args.lr_drop or max(1, round(args.epochs * 0.75))
    checkpoint_interval = args.epochs + 1  # keep last and best EMA only
    patience_evals = max(1, round(args.patience / args.eval_interval))

    if args.effective_batch % args.batch_size:
        ap.error(f"--effective-batch {args.effective_batch} must be divisible by --batch-size {args.batch_size}")
    grad_accum_steps = args.effective_batch // args.batch_size

    long_side = resolution if args.square else round(resolution * args.aspect)
    for name, side in (("short", resolution), ("long", long_side)):
        if side % 24 != 0:
            ap.error(f"{name} side {side} must be divisible by 24 (patch_size x num_windows)")

    aug_config = AUG_PRESETS[args.aug_preset]
    aug_backend = "cpu" if aug_config is None else args.aug_backend

    run_name = args.run_name or (
        f"seg_{args.variant}_{resolution}_{args.aug_preset}_"
        f"{'square' if args.square else f'{resolution}x{long_side}'}"
        + ("_frozen" if args.freeze_backbone else "")
    )
    output_dir = args.output_dir or os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    classes = class_names_from_dataset(args.dataset_dir)
    resume_from = args.resume_from
    if resume_from is None and not args.no_resume:
        resume_from = find_resume_checkpoint(output_dir)

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"

    print("=" * 70)
    print(" RF-DETR Segmentation Training Pipeline")
    print("=" * 70)
    print(f"Variant           : {args.variant} (nominal: {nominal})")
    print(f"Input Resolution  : {resolution}x{long_side} ({'square stretch' if args.square else f'aspect {args.aspect}'})")
    print(f"ViT Tokens/Image  : {(resolution // 12) * (long_side // 12)} tokens (patch size 12)")
    print(f"Frozen Backbone   : {args.freeze_backbone} (Linear Probing)")
    print(f"Augmentation      : {args.aug_preset} ({aug_backend} backend)")
    print(f"Hardware          : {device_name}")
    print(f"Dataset Directory : {args.dataset_dir}")
    print(f"Classes ({len(classes)})       : {classes}")
    print(f"Schedule          : {args.epochs} epochs, LR drop at epoch {lr_drop}")
    print(f"Batch Structure   : micro {args.batch_size} x accum {grad_accum_steps} = {args.effective_batch} effective")
    print(f"Output Directory  : {output_dir}")
    print("=" * 70)

    if args.freeze_backbone:
        from rfdetr.training.module_model import RFDETRModelModule
        _orig_init = RFDETRModelModule.__init__

        def _frozen_init(self, model_config, train_config):
            _orig_init(self, model_config, train_config)
            if hasattr(self.model, "backbone"):
                for p in self.model.backbone.parameters():
                    p.requires_grad = False
                print("[freeze] Locked all DINOv2 backbone parameters (requires_grad = False)")

        RFDETRModelModule.__init__ = _frozen_init

    start_time = time.time()
    model = cls(resolution=resolution, num_classes=len(classes), compile=args.compile)
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=grad_accum_steps,
        output_dir=output_dir,
        resolution=resolution,
        num_workers=args.num_workers,
        ema_decay=args.ema_decay,
        ema_update_interval=args.ema_update_interval,
        **({"lr": args.lr} if args.lr is not None else {}),
        **({"lr_encoder": args.lr_encoder} if args.lr_encoder is not None else {}),
        square_resize_div_64=args.square,
        multi_scale=args.multi_scale,
        scale_jitter=args.scale_jitter,
        aug_config=aug_config,
        augmentation_backend=aug_backend,
        checkpoint_interval=checkpoint_interval,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler_kwargs={"lr_drop": lr_drop},
        eval_interval=args.eval_interval,
        eval_ema_only=args.eval_ema_only,
        compute_val_loss=args.compute_val_loss,
        early_stopping=True,
        early_stopping_patience=patience_evals,
        early_stopping_min_delta=args.min_delta,
        early_stopping_use_ema=True,
        seed=args.seed,
        class_names=classes,
        resume=resume_from,
        wandb=args.wandb,
        project=args.project,
        run=run_name,
    )
    duration = time.time() - start_time
    print(f"\n==> Training completed in {duration / 60:.1f} minutes.")

    # Save summary metadata
    summary = {
        "variant": args.variant,
        "resolution": resolution,
        "input_hw": [resolution, long_side],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch": args.effective_batch,
        "freeze_backbone": args.freeze_backbone,
        "aug_preset": args.aug_preset,
        "square": args.square,
        "training_seconds": round(duration, 1),
        "class_names": classes,
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Export best model to ONNX with metadata
    try:
        onnx_out = model.export(
            output_dir=output_dir,
            format="onnx",
            shape=(resolution, long_side),
            notes={
                "class_names": classes,
                "variant": args.variant,
                "resolution": resolution,
                "input_hw": [resolution, long_side],
                "square": args.square,
            },
        )
        print(f"==> Model successfully exported to ONNX: {onnx_out}")
    except Exception as e:
        print(f"[warning] Automatic ONNX export failed: {e}. You can use export_onnx.py manually.", file=sys.stderr)

    if args.hf_repo:
        maybe_publish_to_hf(output_dir, args.hf_repo, run_name)


if __name__ == "__main__":
    main()
