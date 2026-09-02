"""Measure accuracy and latency of a trained variant, on the test split.

Accuracy: COCO mAP for boxes AND for masks. Both are needed -- a correct box
around a filled-in mask gives a respectable box mAP and a poor mask mAP, and it
is the mask that feeds the pose stage.

Latency: batch of 1, after warm-up, with explicit CUDA synchronisation. Without
torch.cuda.synchronize the CUDA calls are asynchronous and what gets timed is
the enqueue, not the compute -- the mistake yields absurdly low latencies.

This latency is the development card's, not the target Jetson Orin Nano's. The
ordering between variants usually survives, the absolute values do not: the
final choice is confirmed on the target, under TensorRT.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pycocotools.mask as coco_mask
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

VARIANTS = {"nano": RFDETRSegNano, "small": RFDETRSegSmall,
            "medium": RFDETRSegMedium}

# Low confidence threshold: mAP integrates the precision/recall curve and needs
# the unsure detections to describe its tail. A deployment threshold (0.5) would
# truncate the curve and understate the model.
MAP_THRESHOLD = 0.05


def find_checkpoint(output_dir):
    for pattern in ("checkpoint_best_total.pth", "checkpoint_best*.pth",
                    "checkpoint.pth", "*.pth"):
        found = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if found:
            return found[0]
    raise SystemExit(f"ABORT: no checkpoint in {output_dir}")


def detect_offset(model, paths, gt_ids):
    """Compare predicted class ids to annotated ids and deduce the offset.

    The model reasons in contiguous indices, the COCO file in category ids.
    Guessing that offset would collapse the mAP in silence, so observe it.
    """
    seen = set()
    for path in paths[:200]:
        det = model.predict(path, threshold=MAP_THRESHOLD)
        if det.class_id is not None and len(det.class_id):
            seen.update(int(c) for c in det.class_id)
    if not seen:
        raise SystemExit("ABORT: no detection on the sample, class offset "
                         "cannot be determined")
    if seen <= gt_ids:
        print(f"predicted ids {sorted(seen)} already within annotated ids "
              f"{sorted(gt_ids)}: no offset")
        return 0
    offset = min(gt_ids) - min(seen)
    print(f"predicted ids {sorted(seen)} against annotated {sorted(gt_ids)}: "
          f"offset applied {offset:+d}")
    return offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--latency-iters", type=int, default=100)
    ap.add_argument("--latency-warmup", type=int, default=20)
    args = ap.parse_args()

    output_dir = args.output_dir or f"output/seg_{args.variant}"
    checkpoint = find_checkpoint(output_dir)
    test_dir = os.path.join(args.dataset_dir, "test")
    ann_file = os.path.join(test_dir, "_annotations.coco.json")

    coco_gt = COCO(ann_file)
    gt_ids = {int(c) for c in coco_gt.getCatIds()}
    images = coco_gt.loadImgs(coco_gt.getImgIds())
    paths = [os.path.join(test_dir, im["file_name"]) for im in images]

    print(f"variant    : {args.variant}")
    print(f"checkpoint : {checkpoint}")
    print(f"test images: {len(images)}\n")

    model = VARIANTS[args.variant](pretrain_weights=checkpoint,
                                   num_classes=len(gt_ids))
    offset = detect_offset(model, paths, gt_ids)

    # --- accuracy ------------------------------------------------------------
    results = []
    for n, (meta, path) in enumerate(zip(images, paths), 1):
        det = model.predict(path, threshold=MAP_THRESHOLD)
        if det.mask is None:
            raise SystemExit("ABORT: the model returns no mask; check that this "
                             "really is a Seg variant")
        for k in range(len(det.xyxy)):
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[k])
            rle = coco_mask.encode(
                np.asfortranarray(det.mask[k].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            results.append({
                "image_id": meta["id"],
                "category_id": int(det.class_id[k]) + offset,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": rle,
                "score": float(det.confidence[k]),
            })
        if n % 500 == 0:
            print(f"  {n}/{len(images)}")

    if not results:
        raise SystemExit("ABORT: no detection on the test split")

    summary = {"variant": args.variant, "checkpoint": checkpoint,
               "test_images": len(images), "detections": len(results)}

    coco_dt = coco_gt.loadRes(results)
    for iou_type in ("bbox", "segm"):
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        summary[f"mAP_{iou_type}"] = round(float(ev.stats[0]), 4)
        summary[f"mAP50_{iou_type}"] = round(float(ev.stats[1]), 4)

    # --- latency -------------------------------------------------------------
    # A single image reused: this measures the model, not the disk read.
    img = Image.open(paths[0]).convert("RGB")
    for _ in range(args.latency_warmup):
        model.predict(img, threshold=0.5)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.latency_iters):
        model.predict(img, threshold=0.5)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) / args.latency_iters * 1000

    summary["latency_ms_batch1"] = round(latency_ms, 2)
    summary["gpu"] = torch.cuda.get_device_name(0)

    summary_path = os.path.join(output_dir, "bench_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{args.variant}: box mAP {summary['mAP_bbox']:.4f} | "
          f"mask mAP {summary['mAP_segm']:.4f} | "
          f"latency {latency_ms:.1f} ms")
    print(f"written to {summary_path}")


if __name__ == "__main__":
    main()
