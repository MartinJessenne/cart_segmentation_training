"""
Convert an Isaac-Sim-generated parquet dataset (columns: rgb, semantic, semantic_labels)
into a YOLO-seg compatible dataset on disk (images/ + labels/ + data.yaml).

Design notes
------------
- One object instance per sample. The `semantic` image is an RGBA mask where the
  background is fully transparent (alpha == 0) and the object is any pixel with
  alpha > 0. There is no need to disambiguate colors/instances within a sample.
- The `rgb` bytes are written to disk verbatim (no decode/re-encode) since we never
  need to touch RGB pixel data.
- Only the `semantic` image is decoded, to extract mask geometry.
- An instance can appear as several disconnected blobs (occlusion). All blobs
  belonging to the single instance in a sample are merged into ONE polygon via a
  "bridging" (keyhole) technique, so that:
    * YOLO-seg gets one polygon line per sample (still one object),
    * the bounding box Ultralytics derives from that polygon is the correct union
      bbox over all blobs, not just the largest blob.
- Class id resolution is a placeholder (`get_class_id`) — wire in real logic later.
- Parallelization is done with a process pool (CPU-bound work: findContours /
  approxPolyDP do not benefit from threads because of the GIL).
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Fixed dataset config
# ---------------------------------------------------------------------------

CLASS_NAMES = ["colruyt", "picanol", "leanflow"]
SPLITS = ["train", "validation", "test"]  # expected parquet filenames (without ext)
SPLIT_OUT_NAMES = {"split=train": "train", "split=validation": "val", "split=test": "test"}  # yolo dir names

MIN_CONTOUR_AREA = 15.0          # px^2, drops anti-aliasing / noise specks
EPS_RATIO = 0.002                # approxPolyDP epsilon as a fraction of contour perimeter
MIN_EPS = 1.0                    # floor for epsilon in pixels
MIN_POLY_POINTS = 4              # never simplify a blob below this many points


# ---------------------------------------------------------------------------
# Class assignment — PLACEHOLDER, fill in real logic later.
# ---------------------------------------------------------------------------
def keys_to_int(obj):
    return {int(k) if k.isdigit() else k: v for k, v in obj.items()}

def get_class_id(semantic_labels_value) -> int:
    """
    TODO: replace with real class resolution logic based on `semantic_labels`.
    Must return an int in [0, len(CLASS_NAMES) - 1].
    """
    label_dict = json.loads(semantic_labels_value, object_hook=keys_to_int)
    label_list = list(label_dict)
    return label_list[0]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _simplify_contour(contour: np.ndarray) -> np.ndarray:
    """Simplify a single contour with perimeter-relative epsilon, with a floor
    on both epsilon and resulting point count."""
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(EPS_RATIO * perimeter, MIN_EPS)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < MIN_POLY_POINTS:
        # fall back to the raw (unsimplified) contour if simplification collapsed it
        pts = contour.reshape(-1, 2)
    return pts


def _bridge(poly_pts: list, blob_pts: list) -> list:
    """
    Merge `blob_pts` into `poly_pts` via a keyhole bridge: find the closest
    pair of points between the two rings, cut both rings open there, and
    stitch them into a single closed ring connected by a thin double edge.
    This lets a multi-blob (occluded) instance be expressed as ONE YOLO-seg
    polygon whose bounding box is the true union bbox.
    """
    poly_arr = np.asarray(poly_pts, dtype=np.float64)
    blob_arr = np.asarray(blob_pts, dtype=np.float64)

    # pairwise distances, find the closest connecting pair (i in poly, j in blob)
    diff = poly_arr[:, None, :] - blob_arr[None, :, :]
    dists = np.einsum("ijk,ijk->ij", diff, diff)  # squared distances, cheaper
    i, j = np.unravel_index(np.argmin(dists), dists.shape)

    # rotate the blob ring so it starts (and ends) at the bridge point j
    blob_rot = blob_pts[j:] + blob_pts[:j + 1]

    # splice: poly up to and including i -> bridge into blob -> back to i -> rest of poly
    new_poly = poly_pts[:i + 1] + [blob_pts[j]] + blob_rot + [poly_pts[i]] + poly_pts[i + 1:]
    return new_poly


def mask_to_single_polygon(mask: np.ndarray):
    """
    mask: uint8 binary mask (0/255), single instance, possibly split into
    several disconnected blobs due to occlusion.

    Returns a single list of (x, y) pixel-coordinate points forming one
    closed polygon representing the whole instance, merging blobs as needed.
    Returns None if no valid contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not contours:
        return None

    simplified = [_simplify_contour(c) for c in contours]

    if len(simplified) == 1:
        return simplified[0].tolist()

    # merge largest-first so the bridge network stays compact
    simplified.sort(key=lambda pts: -cv2.contourArea(np.asarray(pts, dtype=np.int32)))
    merged = simplified[0].tolist()
    for blob in simplified[1:]:
        merged = _bridge(merged, blob.tolist())
    return merged


# ---------------------------------------------------------------------------
# Per-sample worker (runs in a separate process)
# ---------------------------------------------------------------------------

def process_sample(args):
    """
    args: tuple(
        sample_id: str,
        rgb_bytes: bytes,
        semantic_bytes: bytes,
        class_id: int,
        img_dir: str,
        lbl_dir: str,
    )
    Returns (sample_id, success: bool, message: str)
    """
    sample_id, rgb_bytes, semantic_bytes, class_id, img_dir, lbl_dir = args
    try:
        # 1. write RGB bytes verbatim -- no decode/re-encode needed
        img_path = os.path.join(img_dir, f"{sample_id}.png")
        with open(img_path, "wb") as f:
            f.write(rgb_bytes)

        # 2. decode semantic mask (need actual pixels here)
        buf = np.frombuffer(semantic_bytes, dtype=np.uint8)
        sem = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if sem is None:
            os.remove(img_path)
            return (sample_id, False, "semantic_decode_failed")
        if sem.ndim < 3 or sem.shape[2] < 4:
            os.remove(img_path)
            return (sample_id, False, "semantic_missing_alpha_channel")

        alpha = sem[:, :, 3]
        h, w = alpha.shape[:2]
        mask = (alpha > 0).astype(np.uint8) * 255

        polygon = mask_to_single_polygon(mask)
        if polygon is None:
            os.remove(img_path)
            return (sample_id, False, "no_valid_contour")

        # normalize to [0, 1]
        norm_flat = []
        for x, y in polygon:
            norm_flat.append(min(max(x / w, 0.0), 1.0))
            norm_flat.append(min(max(y / h, 0.0), 1.0))

        line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_flat) + "\n"
        lbl_path = os.path.join(lbl_dir, f"{sample_id}.txt")
        with open(lbl_path, "w") as f:
            f.write(line)

        return (sample_id, True, "ok")
    except Exception as e:  # noqa: BLE001 - report and keep the pool alive
        if os.path.exists(img_path):
            os.remove(img_path)
        return (sample_id, False, f"exception: {e!r}")


# ---------------------------------------------------------------------------
# Parquet streaming
# ---------------------------------------------------------------------------

def _extract_bytes(value):
    """
    HF-exported image columns are usually one of:
      - raw bytes
      - dict-like {'bytes': b'...', 'path': '...'}  (pyarrow gives this as a dict)
    Handle both.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        b = value.get("bytes")
        if b is not None:
            return bytes(b)
        path = value.get("path")
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    raise ValueError(f"Unrecognized image column value type: {type(value)}")


def iter_parquet_rows(parquet_path: str, batch_size: int = 256):
    """Stream rows from a parquet file without loading it all into memory."""
    pf = pq.ParquetFile(parquet_path)
    columns = ["rgb", "semantic", "semantic_labels"]
    available = set(pf.schema_arrow.names)
    columns = [c for c in columns if c in available]

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        d = batch.to_pydict()
        n = len(d[columns[0]])
        for i in range(n):
            row = {c: d[c][i] for c in columns}
            yield row


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_split(parquet_path, split_key, out_root, num_workers, batch_size, log_dir):
    out_split = SPLIT_OUT_NAMES[split_key]
    img_dir = os.path.join(out_root, "images", out_split)
    lbl_dir = os.path.join(out_root, "labels", out_split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    failures = []
    submitted = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, row in enumerate(iter_parquet_rows(parquet_path, batch_size=batch_size)):
            sample_id = f"{out_split}_{idx:06d}"
            rgb_bytes = _extract_bytes(row["rgb"])
            semantic_bytes = _extract_bytes(row["semantic"])
            class_id = get_class_id(row.get("semantic_labels"))

            futures.append(
                executor.submit(
                    process_sample,
                    (sample_id, rgb_bytes, semantic_bytes, class_id, img_dir, lbl_dir),
                )
            )
            submitted += 1

        for fut in tqdm(as_completed(futures), total=submitted, desc=f"{split_key}", unit="img"):
            sample_id, ok, msg = fut.result()
            if not ok:
                failures.append((sample_id, msg))

    elapsed = time.time() - t0
    print(f"[{split_key}] done: {submitted - len(failures)}/{submitted} ok "
          f"in {elapsed:.1f}s ({submitted / max(elapsed, 1e-6):.1f} img/s)")

    if failures:
        os.makedirs(log_dir, exist_ok=True)
        fail_path = os.path.join(log_dir, f"{split_key}_failures.json")
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"[{split_key}] {len(failures)} failures logged to {fail_path}")

    return submitted, len(failures)


def write_data_yaml(out_root):
    yaml_path = os.path.join(out_root, "data.yaml")
    lines = [
        f"path: {os.path.abspath(out_root)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {json.dumps(CLASS_NAMES)}",
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet cart dataset to YOLO-seg format")
    parser.add_argument("--parquet-dir", required=True,
                         help="Directory containing train.parquet / validation.parquet / test.parquet")
    parser.add_argument("--out-dir", required=True, help="Output dataset root directory")
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256, help="Parquet read batch size")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                         help="Which splits to process (must match parquet filenames without extension)")
    args = parser.parse_args()

    log_dir = os.path.join(args.out_dir, "_logs")
    total_ok, total_fail = 0, 0

    DATASET_NAME = "data_0"
    split_keys = ["split=train", "split=validation", "split=test"]

    for split_key in split_keys:
        parquet_path = os.path.join(args.parquet_dir, split_key, f"{DATASET_NAME}.parquet")
        if not os.path.exists(parquet_path):
            print(f"WARNING: {parquet_path} not found, skipping", file=sys.stderr)
            continue
        submitted, failed = process_split(
            parquet_path, split_key, args.out_dir, args.num_workers, args.batch_size, log_dir
        )
        total_ok += submitted - failed
        total_fail += failed

    write_data_yaml(args.out_dir)
    print(f"\nTOTAL: {total_ok} ok, {total_fail} failed")


if __name__ == "__main__":
    main()