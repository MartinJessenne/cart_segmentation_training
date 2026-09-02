"""Write annotated previews, to check by eye that class and geometry line up.

Two mistakes pass every metric without leaving a trace:
 - a permutation of the class names, which yields a perfectly trained model that
   answers with the wrong name;
 - a shifted or filled-in mask, whose mAP stays respectable as long as the box
   lands correctly.

Both are immediately visible on an annotated image, and on nothing else. Each
preview carries the class name taken from the COCO file, the mask decoded from
its RLE and the box, so the name can be checked against the cart actually
visible in the frame.
"""
import argparse
import json
import os

import cv2
import pycocotools.mask as coco_mask

# Overlay tint per class name, independent of the source mask colours: the
# preview has to stay readable even when the mask itself is wrong.
OVERLAY = {"picanol": (60, 60, 230), "colruyt": (60, 210, 60),
           "leanflow": (230, 120, 60)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="_rfdetr_dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", default="_preview")
    args = ap.parse_args()

    split_dir = os.path.join(args.root, args.split)
    with open(os.path.join(split_dir, "_annotations.coco.json")) as fh:
        coco = json.load(fh)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    by_class = {cid: [] for cid in names}
    for ann in coco["annotations"]:
        by_class[ann["category_id"]].append(ann)

    os.makedirs(args.out, exist_ok=True)
    for cid, anns in by_class.items():
        if not anns:
            print(f"{names[cid]}: no instance")
            continue
        # Spread across the whole split rather than taking the first few: the
        # shards are ordered, so the beginning shows only a handful of scenes.
        step = max(1, len(anns) // args.per_class)
        for k, ann in enumerate(anns[::step][:args.per_class]):
            meta = images[ann["image_id"]]
            img = cv2.imread(os.path.join(split_dir, meta["file_name"]),
                             cv2.IMREAD_COLOR)
            rle = dict(ann["segmentation"])
            rle["counts"] = rle["counts"].encode("ascii")
            mask = coco_mask.decode(rle).astype(bool)

            colour = OVERLAY[names[cid]]
            layer = img.copy()
            layer[mask] = colour
            img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)

            x, y, w, h = (int(v) for v in ann["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
            text = f"{names[cid]}  area={int(ann['area'])}px"
            cv2.putText(img, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 5)
            cv2.putText(img, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        colour, 2)

            target = os.path.join(args.out, f"{names[cid]}_{k}.png")
            cv2.imwrite(target, img)
            print(f"{target}  <- {meta['file_name']}")


if __name__ == "__main__":
    main()
