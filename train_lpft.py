"""Linear Probing then Fine-Tuning (LP-FT) for RF-DETR-Seg.

Starts from a 100% real-bag golden checkpoint (e.g. frozen DINOv2 backbone)
and trains for exactly 3 epochs to close the synthetic mask boundary mAP gap
without collapsing real-domain classification accuracy.

Variants:
  Option A (--mode global):
    Entire DINOv2 ViT backbone unfrozen with lr_encoder = 1.0e-6, lr = 1.0e-5.
  Option B (--mode partial):
    ViT blocks 0..9 strictly frozen; blocks 10..11, layernorm, projector, and
    heads unfrozen with lr_encoder = 2.0e-6, lr = 1.0e-5.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch
from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium
from rfdetr.training.module_model import RFDETRModelModule

VARIANTS = {
    "nano": (RFDETRSegNano, 312),
    "small": (RFDETRSegSmall, 384),
    "medium": (RFDETRSegMedium, 432),
}

# Targeted non-geometric sensor perturbations (Trial Log 30.22, Lever 1)
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

PREFIX = "_orig_mod."


def strip_compile_prefix(path: str) -> str:
    """Strip _orig_mod. prefix added by torch.compile to prevent KeyError in load_pretrain_weights."""
    if not os.path.isfile(path):
        return path
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("model")
    if not isinstance(model, dict) or not any(k.startswith(PREFIX) for k in model):
        return path
    print(f"[weights] Stripping '{PREFIX}' prefix from checkpoint keys...")
    ck["model"] = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v
                   for k, v in model.items()}
    fd, clean = tempfile.mkstemp(suffix=".pth", dir=os.path.dirname(path) or ".")
    os.close(fd)
    torch.save(ck, clean)
    print(f"[weights] Sanitized checkpoint saved to: {clean}")
    return clean


def class_names_from_dataset(dataset_dir: str):
    path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    with open(path) as fh:
        coco = json.load(fh)
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def hook_lpft_module(mode: str):
    """Monkey-patch RFDETRModelModule.__init__ to apply the exact LP-FT unfreezing schedule."""
    _orig_module_init = RFDETRModelModule.__init__

    def _lpft_module_init(self, model_config, train_config):
        _orig_module_init(self, model_config, train_config)
        if not hasattr(self.model, "backbone"):
            return

        if mode == "global":
            # Option A: Full unfreezing across all modules
            for p in self.model.parameters():
                p.requires_grad = True
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[LP-FT Option A: Global] Entire model unfrozen ({total_params:,} trainable params).")

        elif mode == "partial":
            # Option B: Partial ViT unfreezing
            # 1. Lock all backbone parameters
            for p in self.model.backbone.parameters():
                p.requires_grad = False

            # 2. Unlock blocks 10 and 11
            w_enc = self.model.backbone[0].encoder.encoder.encoder
            for layer in w_enc.layer[10:]:
                for p in layer.parameters():
                    p.requires_grad = True

            # 3. Unlock backbone LayerNorm
            w_dino = self.model.backbone[0].encoder.encoder
            if hasattr(w_dino, "layernorm"):
                for p in w_dino.layernorm.parameters():
                    p.requires_grad = True

            # 4. Unlock multi-scale feature projector
            if hasattr(self.model.backbone[0], "projector"):
                for p in self.model.backbone[0].projector.parameters():
                    p.requires_grad = True

            # 5. Unlock all non-backbone modules (decoder, classification & mask heads)
            for name, m in self.model.named_children():
                if name != "backbone":
                    for p in m.parameters():
                        p.requires_grad = True

            frozen_bb = sum(p.numel() for p in self.model.backbone.parameters() if not p.requires_grad)
            trainable_bb = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)
            total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[LP-FT Option B: Partial] Backbone: {frozen_bb:,} frozen (blocks 0..9), "
                  f"{trainable_bb:,} trainable (blocks 10..11 + LN + projector).")
            print(f"[LP-FT Option B: Partial] Total trainable parameters: {total_trainable:,}")

    RFDETRModelModule.__init__ = _lpft_module_init


def main():
    ap = argparse.ArgumentParser(description="LP-FT Fine-Tuning for RF-DETR-Seg")
    ap.add_argument("--mode", choices=["global", "partial"], default="global",
                    help="LP-FT mode: 'global' (Option A) or 'partial' (Option B)")
    ap.add_argument("--variant", default="nano", choices=sorted(VARIANTS),
                    help="RF-DETR variant (nano, small, medium)")
    ap.add_argument("--weights", required=True,
                    help="Path to starting golden checkpoint (.pth)")
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset_960",
                    help="Path to pre-processed COCO dataset")
    ap.add_argument("--resolution", type=int, default=288,
                    help="Input resolution (short side, default 288)")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Exact training epochs (default 3)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Micro-batch size per step (default 32)")
    ap.add_argument("--effective-batch", type=int, default=32,
                    help="Effective batch size (default 32)")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Head / decoder learning rate (default 1e-5)")
    ap.add_argument("--lr-encoder", type=float, default=None,
                    help="Backbone learning rate (default 1e-6 for global, 2e-6 for partial)")
    ap.add_argument("--ema-decay", type=float, default=0.972,
                    help="EMA decay (default 0.972)")
    ap.add_argument("--ema-update-interval", type=int, default=4,
                    help="EMA update interval in steps (default 4)")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Dataloader workers (default 8)")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="Enable torch.compile (disabled by default for stability)")
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="Explicitly disable torch.compile")
    ap.add_argument("--output-dir", default=None,
                    help="Custom output directory")
    ap.add_argument("--run-name", default=None,
                    help="Custom WandB run name")
    ap.add_argument("--project", default="cart_segmentation",
                    help="WandB project name")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", default=True,
                    help="Disable WandB logging")
    args = ap.parse_args()

    # Set default encoder learning rate based on mode
    if args.lr_encoder is None:
        args.lr_encoder = 1e-6 if args.mode == "global" else 2e-6

    # Determine default run name and output directory
    if args.run_name is None:
        mode_suffix = "global1e6" if args.mode == "global" else "last2blocks2e6"
        args.run_name = f"lpft_opt_{args.mode}_{mode_suffix}_b{args.batch_size}"
    if args.output_dir is None:
        args.output_dir = os.path.join("output", args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    cls, nominal = VARIANTS[args.variant]
    resolution = args.resolution
    classes = class_names_from_dataset(args.dataset_dir)
    grad_accum_steps = args.effective_batch // args.batch_size

    print("=" * 80)
    print(f" LP-FT Runner: {args.run_name}")
    print(f" Mode        : Option {'A' if args.mode == 'global' else 'B'} ({args.mode.upper()})")
    print(f" Weights     : {args.weights}")
    print(f" Variant     : {args.variant} ({resolution}x{resolution} square stretch)")
    print(f" Schedule    : {args.epochs} epochs (No early abortion)")
    print(f" LR          : Head/Decoder {args.lr:g} | Backbone {args.lr_encoder:g}")
    print(f" Batch       : {args.batch_size} micro x {grad_accum_steps} accum = {args.effective_batch} effective")
    print(f" Augmentation: SENSOR_AUG (non-geometric photometric)")
    print(f" Output Dir  : {args.output_dir}")
    print("=" * 80)

    # 1. Apply LP-FT module monkey-patch
    hook_lpft_module(args.mode)

    # 2. Sanitize checkpoint weights (strip _orig_mod.)
    clean_weights = strip_compile_prefix(args.weights)

    try:
        # 3. Instantiate model with starting weights
        model = cls(
            pretrain_weights=clean_weights,
            resolution=resolution,
            num_classes=len(classes),
            compile=args.compile,
        )

        start_time = time.time()
        # 4. Train model
        model.train(
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=grad_accum_steps,
            output_dir=args.output_dir,
            resolution=resolution,
            num_workers=args.num_workers,
            ema_decay=args.ema_decay,
            ema_update_interval=args.ema_update_interval,
            lr=args.lr,
            lr_encoder=args.lr_encoder,
            square_resize_div_64=True,
            multi_scale=False,
            scale_jitter=True,
            aug_config=SENSOR_AUG,
            augmentation_backend="kornia",
            checkpoint_interval=1,
            warmup_epochs=0,
            eval_interval=1,
            eval_ema_only=False,
            compute_val_loss=True,
            early_stopping=False,
            seed=42,
            class_names=classes,
            wandb=args.wandb,
            project=args.project,
            run=args.run_name,
        )
        duration = time.time() - start_time
        print(f"\n[LP-FT] Completed 3 epochs in {duration/60:.1f} minutes!")

        # 5. Export to ONNX
        print("[export] Exporting final checkpoint to ONNX...")
        model.export(
            output_dir=args.output_dir,
            format="onnx",
            shape=(resolution, resolution),
            notes={"class_names": classes, "variant": args.variant, "square": True, "lpft_mode": args.mode},
        )
        print("[export] ONNX export complete.")

    finally:
        if clean_weights != args.weights and os.path.exists(clean_weights):
            os.unlink(clean_weights)


if __name__ == "__main__":
    main()
