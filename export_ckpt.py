"""Export a checkpoint to ONNX, tolerating torch.compile's key prefix.

A model wrapped by torch.compile is an OptimizedModule, and its state_dict keys
all carry an "_orig_mod." prefix. RF-DETR's load_pretrain_weights reads
checkpoint["model"]["class_embed.bias"] directly, so it raises KeyError on such a
checkpoint. The weights are unaffected -- only the names are -- so stripping the
prefix into a temporary copy is enough, and is what this does.
"""
import os
import sys
import tempfile

import torch
from rfdetr import RFDETRSegNano

PREFIX = "_orig_mod."
NAMES = ["picanol", "colruyt", "leanflow"]


def strip_compile_prefix(path: str) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("model")
    if not isinstance(model, dict) or not any(k.startswith(PREFIX) for k in model):
        return path
    ck["model"] = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v
                   for k, v in model.items()}
    fd, clean = tempfile.mkstemp(suffix=".pth", dir=os.path.dirname(path) or ".")
    os.close(fd)
    torch.save(ck, clean)
    return clean


def main() -> None:
    ckpt, side, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    clean = strip_compile_prefix(ckpt)
    try:
        model = RFDETRSegNano(pretrain_weights=clean, num_classes=len(NAMES))
        model.export(output_dir=out, format="onnx", shape=(side, side),
                     notes={"class_names": NAMES, "variant": "nano", "square": True})
    finally:
        if clean != ckpt:
            os.unlink(clean)
    print("EXPORT OK")


if __name__ == "__main__":
    main()
