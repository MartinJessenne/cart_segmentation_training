"""
Validation pass: sample random YOLO-seg label files, redraw the polygon and
its derived bounding box over the RGB image, and save a contact sheet so you
can visually sanity-check the conversion before launching training.

Also reports basic label-file health stats (empty files, malformed lines,
point counts) and class distribution.

Usage:
    python validate_labels.py --dataset-dir out_ds --split train --n 30 --out contact_sheet.png
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np


def load_label(path):
    """Return list of (class_id, [(x,y), ...]) polygons in a label file."""
    polys = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 7 or len(parts) % 2 == 0:
                # class_id + at least 3 (x,y) pairs -> odd total token count
                raise ValueError(f"malformed line (token count={len(parts)}): {path}")
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = list(zip(coords[0::2], coords[1::2]))
            polys.append((class_id, pts))
    return polys


def draw_sample(img_path, label_path, class_names):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {img_path}")
    h, w = img.shape[:2]

    polys = load_label(label_path)
    for class_id, norm_pts in polys:
        pts_px = np.array([[x * w, y * h] for x, y in norm_pts], dtype=np.int32)
        cv2.polylines(img, [pts_px], isClosed=True, color=(0, 255, 0), thickness=2)

        x_min, y_min = pts_px[:, 0].min(), pts_px[:, 1].min()
        x_max, y_max = pts_px[:, 0].max(), pts_px[:, 1].max()
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)

        label_text = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.putText(img, label_text, (x_min, max(y_min - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def build_contact_sheet(images, tile_size=256, cols=6):
    n = len(images)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * tile_size, cols * tile_size, 3), 40, dtype=np.uint8)
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(im, (tile_size, tile_size))
        sheet[r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size] = resized
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO-seg labels visually")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=30, help="number of random samples to check")
    parser.add_argument("--out", default="contact_sheet.png")
    parser.add_argument("--class-names", nargs="+", default=["colruyt", "picanol", "leanflow"])
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset_dir, "images", args.split)
    lbl_dir = os.path.join(args.dataset_dir, "labels", args.split)

    label_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    if not label_paths:
        print(f"No label files found in {lbl_dir}")
        return

    # --- health stats over ALL labels in the split ---
    empty_count = 0
    malformed = []
    class_counts = {}
    point_counts = []

    for lp in label_paths:
        if os.path.getsize(lp) == 0:
            empty_count += 1
            continue
        try:
            polys = load_label(lp)
        except ValueError as e:
            malformed.append(str(e))
            continue
        for class_id, pts in polys:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            point_counts.append(len(pts))

    print(f"=== Label health report: split={args.split} ===")
    print(f"total label files: {len(label_paths)}")
    print(f"empty label files: {empty_count}")
    print(f"malformed label files: {len(malformed)}")
    if malformed[:5]:
        print("  first few malformed:", malformed[:5])
    print(f"class distribution: { {args.class_names[k] if k < len(args.class_names) else k: v for k, v in class_counts.items()} }")
    if point_counts:
        print(f"polygon point count: min={min(point_counts)} max={max(point_counts)} "
              f"avg={sum(point_counts)/len(point_counts):.1f}")

    # --- visual contact sheet on a random sample ---
    sample_n = min(args.n, len(label_paths))
    chosen = random.sample(label_paths, sample_n)

    tiles = []
    for lp in chosen:
        sample_id = os.path.splitext(os.path.basename(lp))[0]
        img_path = os.path.join(img_dir, f"{sample_id}.png")
        if not os.path.exists(img_path):
            print(f"WARNING: missing image for label {lp}")
            continue
        try:
            tiles.append(draw_sample(img_path, lp, args.class_names))
        except ValueError as e:
            print(f"WARNING: {e}")

    if tiles:
        sheet = build_contact_sheet(tiles)
        cv2.imwrite(args.out, sheet)
        print(f"\nWrote contact sheet with {len(tiles)} samples to {args.out}")
    else:
        print("No valid samples to render.")


if __name__ == "__main__":
    main()