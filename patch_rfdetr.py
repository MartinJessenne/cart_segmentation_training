"""Make RF-DETR pad instance masks along with the images it batches.

`_collate_with_block_size` pads the images of a batch up to the batch maximum
through `nested_tensor_from_tensor_list`, then returns the targets untouched.
For detection that is harmless: a box is four numbers and needs no padding. For
segmentation it is a defect. Any augmentation that yields a different size per
sample -- the resize jitter's square crop, or multi-scale -- then hands the
model an image of one size and a mask of another, and training stops:

    kornia_transforms.collate_masks
        Target sizes: [1, 360, 576].  Tensor sizes: [360, 360]
    models/matcher.py, torch.cat([v["masks"] for v in targets])
        Expected size 576 but got size 360

Both backends fail because both read masks that the collate never padded. The
default configuration never triggers it: `square_resize_div_64=True` makes every
branch emit the same square, so sizes never differ inside a batch. Asking for
the camera's 16:10 geometry is what makes them differ.

This script edits the installed package in place and is idempotent, so it is
safe to re-run. It has to be re-run after any `uv sync` that reinstalls rfdetr.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

MARKER = "_pad_target_masks"

ANCHOR = """    columns = list(zip(*batch))
    images = nested_tensor_from_tensor_list(list(columns[0]), block_size=block_size)
    return (images, *columns[1:])
"""

REPLACEMENT = """    columns = list(zip(*batch))
    images = nested_tensor_from_tensor_list(list(columns[0]), block_size=block_size)
    _pad_target_masks(columns[1], images.tensors.shape[-2], images.tensors.shape[-1])
    return (images, *columns[1:])
"""

HELPER = '''

def _pad_target_masks(targets, height: int, width: int) -> None:
    """Pad instance masks to the batch's padded image size, top-left aligned.

    The images of a batch are padded to the batch maximum, so a mask that came
    out of augmentation at a different size no longer covers the same pixels as
    its image. Masks are aligned exactly the way the images are in
    ``nested_tensor_from_tensor_list``: copied into the top-left corner, zero
    everywhere else, which is the region the NestedTensor marks as padding.

    Mutates the target dicts in place. They are per-sample dicts owned by this
    batch, so no caller sees the change.
    """
    for target in targets:
        if not isinstance(target, dict):
            continue
        masks = target.get("masks")
        if masks is None or masks.ndim != 3:
            continue
        h, w = masks.shape[-2], masks.shape[-1]
        if h == height and w == width:
            continue
        padded = masks.new_zeros((masks.shape[0], height, width))
        copy_h, copy_w = min(h, height), min(w, width)
        padded[:, :copy_h, :copy_w] = masks[:, :copy_h, :copy_w]
        target["masks"] = padded

'''


def target_file() -> Path:
    spec = importlib.util.find_spec("rfdetr.utilities.tensors")
    if spec is None or spec.origin is None:
        sys.exit("ABORT: rfdetr is not installed in this environment")
    return Path(spec.origin)


def self_test() -> None:
    """Batch two samples of different sizes and check the masks come back aligned."""
    code = """
import torch
from rfdetr.utilities.tensors import make_collate_fn

collate = make_collate_fn(block_size=24)
batch = [
    (torch.zeros(3, 360, 576), {"masks": torch.ones(2, 360, 576, dtype=torch.bool)}),
    (torch.zeros(3, 360, 360), {"masks": torch.ones(1, 360, 360, dtype=torch.bool)}),
]
images, targets = collate(batch)
h, w = images.tensors.shape[-2:]
assert (h, w) == (360, 576), (h, w)
for t in targets:
    assert t["masks"].shape[-2:] == (h, w), t["masks"].shape
# The smaller sample keeps its content top-left and is zero in the padded strip.
small = targets[1]["masks"]
assert small[:, :, :360].all(), "content lost"
assert not small[:, :, 360:].any(), "padding is not empty"
print("self-test ok: masks padded to", tuple(images.tensors.shape[-2:]))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report whether the patch is applied, change nothing")
    args = ap.parse_args()

    path = target_file()
    source = path.read_text()
    applied = MARKER in source

    if args.check:
        print(f"{path}: {'patched' if applied else 'NOT patched'}")
        sys.exit(0 if applied else 1)

    if applied:
        print(f"already patched: {path}")
        self_test()
        return

    if ANCHOR not in source:
        sys.exit(f"ABORT: the expected collate body is not in {path}. rfdetr has "
                 "changed; re-read _collate_with_block_size before patching.")

    marker = "def _collate_with_block_size("
    source = source.replace(marker, HELPER.lstrip("\n") + "\n" + marker, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    path.write_text(source)
    print(f"patched: {path}")
    self_test()


if __name__ == "__main__":
    main()
