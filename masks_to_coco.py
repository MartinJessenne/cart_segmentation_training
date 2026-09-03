"""Convert the Isaac colour masks into COCO instance annotations.

RLE encoding, not polygons. An industrial cart is an open frame: the background
shows through it, and COCO polygons are additive, unable to represent a hole.
An external contour would fill the frame in and overstate the mask.
pycocotools' compressed RLE represents the openings exactly, and RF-DETR's
loader decodes it natively (convert_coco_poly_to_mask accepts both forms).

Category ids are 1..3, not 0..2. RF-DETR derives its class indices with
   {category["id"]: label for label, category in enumerate(kept)}
over the train split's annotated categories only, sorted by id. Ascending ids
therefore reproduce CLASS_MAPPING exactly, and id 0 stays free for the parent
node the Roboflow convention places there.

A class missing from the train split would shift every other class's index, so
the script aborts unless all three are present there.

Produces <root>/<split>/_annotations.coco.json, the layout
build_roboflow_from_coco expects.
"""
import argparse
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pycocotools.mask as coco_mask
from PIL import Image

# Mask colour per internal id, as written by generate_dataset_6_0_1.py.
SEMANTIC_COLORS = {
    0: (220, 50, 50),    # picanol
    1: (50, 200, 50),    # colruyt
    2: (50, 100, 220),   # leanflow
}
CLASS_NAMES = {0: "picanol", 1: "colruyt", 2: "leanflow"}

# Offset between the generator's internal id and the COCO category id.
COCO_ID_OFFSET = 1

SPLITS = ("train", "valid", "test")


def annotations_for(args):
    """One mask -> one RLE annotation per class present.

    Returns (file name, width, height, list of annotations without ids).
    """
    img_path, mask_path, label_path = args

    with open(label_path) as fh:
        labels = json.load(fh)

    mask = np.array(Image.open(mask_path).convert("RGB"))
    height, width = mask.shape[:2]

    out = []
    for sem_id_str in labels:
        sem_id = int(sem_id_str)
        colour = SEMANTIC_COLORS[sem_id]
        # The colours are written by direct assignment, with no resampling, so
        # exact equality is the right comparison. A tolerance would absorb an
        # encoding error instead of reporting it.
        binary = np.all(mask == colour, axis=-1)
        if not binary.any():
            continue
        rle = coco_mask.encode(np.asfortranarray(binary.astype(np.uint8)))
        # counts is bytes; JSON requires text. RF-DETR decodes this form back.
        rle["counts"] = rle["counts"].decode("ascii")
        x, y, w, h = (float(v) for v in coco_mask.toBbox(rle))
        out.append({
            "category_id": sem_id + COCO_ID_OFFSET,
            "segmentation": rle,
            "area": float(coco_mask.area(rle)),
            "bbox": [x, y, w, h],
            "iscrowd": 0,
        })

    return os.path.basename(img_path), width, height, out


def build_split(root, split, workers):
    img_dir = os.path.join(root, split)
    msk_dir = os.path.join(root, "_masks", split)
    if not os.path.isdir(img_dir):
        return None

    names = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    if not names:
        return None

    tasks = []
    for name in names:
        stem = name[:-len(".png")]
        mask_path = os.path.join(msk_dir, name)
        label_path = os.path.join(msk_dir, stem + ".json")
        if not (os.path.exists(mask_path) and os.path.exists(label_path)):
            sys.exit(f"ABORT: mask or labels missing for {split}/{name}")
        tasks.append((os.path.join(img_dir, name), mask_path, label_path))

    images, annotations = [], []
    per_class = {cid: 0 for cid in CLASS_NAMES}
    empty = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for image_id, (file_name, width, height, anns) in enumerate(
                pool.map(annotations_for, tasks, chunksize=16)):
            images.append({"id": image_id, "file_name": file_name,
                           "width": width, "height": height})
            if not anns:
                empty += 1
            for ann in anns:
                ann["id"] = len(annotations)
                ann["image_id"] = image_id
                per_class[ann["category_id"] - COCO_ID_OFFSET] += 1
                annotations.append(ann)
            if (image_id + 1) % 2000 == 0:
                print(f"  {split}: {image_id + 1}/{len(tasks)}")

    coco = {
        "info": {"description": "industrial carts, Isaac Sim, instance masks"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid + COCO_ID_OFFSET, "name": CLASS_NAMES[cid],
                        "supercategory": "cart"}
                       for cid in sorted(CLASS_NAMES)],
    }

    out_path = os.path.join(img_dir, "_annotations.coco.json")
    with open(out_path, "w") as fh:
        json.dump(coco, fh)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{split}: {len(images)} images, {len(annotations)} instances, "
          f"{empty} without annotation, {size_mb:.0f} MB")
    for cid, n in sorted(per_class.items()):
        print(f"    {CLASS_NAMES[cid]:9s} {n:6d}")
    del images, annotations, coco
    gc.collect()
    return per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="_rfdetr_dataset")
    # Kept to 4 max: gVisor sentry cannot handle 20 concurrent Python image-decoding
    # workers without thrashing IPC, VFS file descriptors, and triggering host eviction.
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()

    print(f"{args.workers} processes\n")
    train_counts = None
    for split in SPLITS:
        counts = build_split(args.root, split, args.workers)
        if split == "train":
            train_counts = counts

    if train_counts is None:
        sys.exit("ABORT: no train split, RF-DETR has nothing to build its class "
                 "mapping from")
    missing = [CLASS_NAMES[c] for c, n in train_counts.items() if n == 0]
    if missing:
        sys.exit(f"ABORT: classes absent from the train split: {missing}. "
                 "RF-DETR derives its indices from the categories annotated in "
                 "train alone, so their absence would shift the other classes.")
    print("\nthree classes present in train, RF-DETR indices stable")


if __name__ == "__main__":
    main()
