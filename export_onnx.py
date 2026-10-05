"""Export trained RF-DETR checkpoint to ONNX format.

Handles:
  1. Stripping torch.compile '_orig_mod.' state_dict prefix if compiled.
  2. Embedding deployment metadata (class_names, variant, input resolution) into ONNX custom_metadata_map.
  3. Supporting Nano, Small, and Medium variants.

Usage:
    uv run python export_onnx.py --checkpoint output/seg_nano/checkpoint_best_ema.pth --output-dir exported/
"""
import argparse
import os
import sys
import tempfile

import torch
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

VARIANTS = {
    "nano": RFDETRSegNano,
    "small": RFDETRSegSmall,
    "medium": RFDETRSegMedium,
}

PREFIX = "_orig_mod."
DEFAULT_CLASSES = ["picanol", "colruyt", "leanflow"]


def strip_compile_prefix(path: str) -> str:
    """Strip '_orig_mod.' prefix created by torch.compile from checkpoint keys."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("model")
    if not isinstance(model, dict) or not any(k.startswith(PREFIX) for k in model):
        return path

    ck["model"] = {
        (k[len(PREFIX):] if k.startswith(PREFIX) else k): v
        for k, v in model.items()
    }
    fd, clean_path = tempfile.mkstemp(suffix=".pth", dir=os.path.dirname(path) or ".")
    os.close(fd)
    torch.save(ck, clean_path)
    return clean_path


def main():
    ap = argparse.ArgumentParser(description="Export RF-DETR model checkpoint to ONNX")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained PyTorch checkpoint (.pth or .ckpt)")
    ap.add_argument("--output-dir", default=None,
                    help="Directory to save exported ONNX model (defaults to <checkpoint_dir>/onnx)")
    ap.add_argument("--variant", default="nano", choices=sorted(VARIANTS),
                    help="Model variant: nano, small, medium (default: nano)")
    ap.add_argument("--resolution", type=int, default=288,
                    help="Model resolution (short side or square side; default: 288)")
    ap.add_argument("--square", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether model was trained with square stretch Div64 (default: True)")
    ap.add_argument("--aspect", type=float, default=1.6,
                    help="Aspect ratio if not square (default: 1.6)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="List of class names in ascending ID order (default: picanol colruyt leanflow)")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"ABORT: Checkpoint '{args.checkpoint}' not found.")

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint) or ".", "onnx")
    os.makedirs(output_dir, exist_ok=True)

    long_side = args.resolution if args.square else round(args.resolution * args.aspect)
    shape = (args.resolution, long_side)

    print(f"==> Exporting checkpoint: {args.checkpoint}")
    print(f"    Variant: {args.variant} | Input Shape: {shape} | Classes: {args.classes}")

    clean_checkpoint = strip_compile_prefix(args.checkpoint)
    try:
        model_cls = VARIANTS[args.variant]
        model = model_cls(pretrain_weights=clean_checkpoint, num_classes=len(args.classes))
        onnx_file = model.export(
            output_dir=output_dir,
            format="onnx",
            shape=shape,
            notes={
                "class_names": args.classes,
                "variant": args.variant,
                "resolution": args.resolution,
                "input_hw": list(shape),
                "square": args.square,
            },
        )
        print(f"==> Export successful: {onnx_file}")
    finally:
        if clean_checkpoint != args.checkpoint and os.path.exists(clean_checkpoint):
            os.unlink(clean_checkpoint)


if __name__ == "__main__":
    main()
