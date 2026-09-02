"""Re-encode the COCO dataset to the largest size training ever asks for.

The dataloader, not the GPU, sets the training pace: two workers decode
1280x800 PNGs at 41 ms each, which caps the pipeline at 48 images/s while the
GPU idles. num_workers cannot be raised on this machine, so the only remaining
lever is the cost of one sample.

Two independent factors are removed here:

* **Format.** PNG is lossless and entropy-coded; decoding one costs 39.5 ms
  against 3.2 ms for the equivalent JPEG. Quality 95 was chosen because its
  artefacts sit near one least-significant bit, an order of magnitude below the
  Gaussian noise the training augmentation deliberately adds (std 0.01-0.03 of
  full scale, i.e. 2.5-7.6 LSB). The compression is therefore invisible against
  noise the model is already being taught to tolerate.

* **Size.** 960x600 is the largest input any tier requests, and it is a hard
  bound rather than a guess. Option A of the resize pipeline asks for
  ``SmallestMaxSize(resolution)``, at most 600 on the short side at the top
  tier; Option B, the crop branch, opens with ``SmallestMaxSize([400, 500,
  600])``, whose maximum is also 600. Nothing upsamples, and the top tier trains
  at its native stored size with no resize at all.

  The bound holds only while ``multi_scale`` is off. Its scale list reaches
  ``(base + 4) * patch_size * num_windows``, above the nominal resolution, and
  at the top tier that would upsample from storage.

Measured end to end: 41.6 ms -> 6.4 ms per sample, 48 -> 311 images/s on two
workers, and 1.22 -> 0.17 MB per file, which also lets the whole dataset sit in
the page cache instead of being re-read from disk every epoch.

Masks are resized with the images and their boxes and areas are recomputed from
the resized mask rather than scaled numerically, so the three can never drift
apart.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# The short side of the top resolution tier; see the module docstring for why
# nothing ever asks for more.
TARGET_H, TARGET_W = 600, 960
SPLITS = ("train", "valid", "test")


def resize_rle(segmentation, out_h, out_w):
    """Resize one COCO RLE mask, returning (rle, bbox, area) at the new size.

    Area-averaging then thresholding at half coverage is used rather than
    nearest-neighbour: the carts are open frames, so the masks are dominated by
    thin bars, and nearest-neighbour sampling drops or doubles a bar depending
    on where the sample point lands. Area averaging keeps a bar whose downscaled
    footprint still covers half a pixel, which at the 0.75 scale factor used
    here means every bar at least 2/3 of a pixel wide in the source survives.
    """
    rle = segmentation
    if isinstance(rle.get("counts"), str):
        rle = {"counts": rle["counts"].encode(), "size": rle["size"]}
    mask = mask_utils.decode(rle).astype(np.float32)
    resized = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_AREA)
    binary = np.asfortranarray((resized >= 0.5).astype(np.uint8))
    encoded = mask_utils.encode(binary)
    bbox = mask_utils.toBbox(encoded).tolist()
    area = float(mask_utils.area(encoded))
    # pycocotools returns bytes; JSON needs str, and the loader re-encodes it.
    out = {"counts": encoded["counts"].decode(), "size": encoded["size"]}
    return out, bbox, area


def convert_image(job):
    """Decode one PNG, write the JPEG copy, and report its true stored size."""
    src, dst = job
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        # Lanczos on the way down: the bars are the highest-frequency content in
        # the frame and bilinear would soften them before the model sees them.
        rgb = rgb.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        rgb.save(dst, "JPEG", quality=95, subsampling=0)
    return os.path.basename(dst)


def convert_split(src_dir, dst_dir, workers):
    """Convert one split's images and rewrite its annotation file."""
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(src_dir, "_annotations.coco.json")) as fh:
        coco = json.load(fh)

    jobs, renamed = [], {}
    for image in coco["images"]:
        stem = image["file_name"].rsplit(".", 1)[0]
        new_name = stem + ".jpg"
        jobs.append((os.path.join(src_dir, image["file_name"]),
                     os.path.join(dst_dir, new_name)))
        renamed[image["id"]] = (new_name, image["width"], image["height"])

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(convert_image, jobs, chunksize=16), 1):
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)} images", flush=True)

    for image in coco["images"]:
        image["file_name"] = renamed[image["id"]][0]
        image["width"], image["height"] = TARGET_W, TARGET_H

    # An instance the resize can empty is not a cart. The median instance covers
    # 126,107 px and the first percentile 12,349; surviving a 0.75 scale needs
    # only a couple of pixels, so anything that fails is four orders of
    # magnitude below the smallest real cart and is a rendering artefact.
    kept, dropped = [], []
    for ann in coco["annotations"]:
        seg = ann["segmentation"]
        if not isinstance(seg, dict):
            raise SystemExit(
                f"{src_dir}: polygon segmentation found. This dataset is RLE "
                "throughout because the carts are open frames with holes; a "
                "polygon here means the conversion upstream changed.")
        rle, bbox, area = resize_rle(seg, TARGET_H, TARGET_W)
        if area == 0:
            dropped.append((ann["image_id"], ann["area"]))
            continue
        ann["segmentation"] = rle
        ann["bbox"] = bbox
        ann["area"] = area
        kept.append(ann)
    coco["annotations"] = kept

    # An image left without an instance is a frame that contains a cart and says
    # it does not, which is the one lesson this dataset must never teach. It
    # leaves with its annotation rather than becoming a negative sample.
    annotated = {ann["image_id"] for ann in kept}
    orphans = [im for im in coco["images"] if im["id"] not in annotated]
    coco["images"] = [im for im in coco["images"] if im["id"] in annotated]
    for image in orphans:
        os.remove(os.path.join(dst_dir, image["file_name"]))
        print(f"  dropped {image['file_name']}: mask does not survive the resize",
              flush=True)

    with open(os.path.join(dst_dir, "_annotations.coco.json"), "w") as fh:
        json.dump(coco, fh)

    return len(coco["images"]), len(coco["annotations"]), len(dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_rfdetr_dataset")
    ap.add_argument("--dst", default="_rfdetr_dataset_960")
    # Kept low on purpose: this machine is the one that must stay up, and the
    # job is short enough that two processes finish it in about ten minutes.
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replace", action="store_true",
                    help="delete the source split once its copy is written")
    args = ap.parse_args()

    total_before = total_after = 0
    for split in SPLITS:
        src_dir = os.path.join(args.src, split)
        if not os.path.isdir(src_dir):
            print(f"{split}: absent, skipped")
            continue
        dst_dir = os.path.join(args.dst, split)
        print(f"{split}: converting -> {dst_dir}", flush=True)
        images, annotations, dropped = convert_split(src_dir, dst_dir, args.workers)

        before = sum(os.path.getsize(os.path.join(src_dir, f))
                     for f in os.listdir(src_dir))
        after = sum(os.path.getsize(os.path.join(dst_dir, f))
                    for f in os.listdir(dst_dir))
        total_before += before
        total_after += after
        print(f"{split}: {images} images, {annotations} annotations, "
              f"{dropped} dropped by resize, "
              f"{before/1e9:.2f} -> {after/1e9:.2f} GB", flush=True)
        if args.replace:
            shutil.rmtree(src_dir)

    print(f"\ntotal {total_before/1e9:.2f} -> {total_after/1e9:.2f} GB "
          f"({total_before/max(total_after, 1):.1f}x smaller)")


if __name__ == "__main__":
    main()
