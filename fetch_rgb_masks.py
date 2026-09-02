"""Pull only the columns the RGB training needs from the Hub.

Parquet is columnar, so selecting just rgb, semantic and semantic_labels means
the depth column's byte ranges are never requested from the server. Measured on
one shard: rgb + semantic weigh 1.29 MB per frame against 3.3 MB for a full
row, so the 24705 frames of both ranges come to ~32 GB instead of the 82 GB of
the whole repository.

Work proceeds shard by shard and each one is marked with a .done file, so a
dropped connection resumes where it stopped instead of starting over.

Layout produced, the one RF-DETR expects (the masks are an intermediate and can
be deleted once the COCO file is written):

    <out>/train/<name>.png            RGB image
    <out>/valid/<name>.png
    <out>/test/<name>.png
    <out>/_masks/<split>/<name>.png   class-coloured mask
    <out>/_masks/<split>/<name>.json  class id -> class name
"""
import argparse
import json
import os
import sys

import duckdb
from huggingface_hub import HfApi

REPO = "UItraviolet/industrial_cart"

# The Hub's "validation" split is called "valid" on the RF-DETR side.
SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}

# The three columns carrying RGB supervision. rgb and semantic are `datasets`
# Image() features, stored in parquet as struct<bytes, path>; struct_extract
# avoids the ambiguity between "column.field" and "table.column".
QUERY = """
SELECT struct_extract(rgb, 'bytes')      AS rgb_bytes,
       struct_extract(semantic, 'bytes') AS sem_bytes,
       semantic_labels                   AS labels
FROM read_parquet('hf://datasets/{repo}/{path}')
"""


def shard_split(name):
    """The split is the first segment of the shard name, by construction."""
    return name.rsplit("/", 1)[-1].split("-")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_rfdetr_dataset")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="process only N of them (debugging); 0 = all")
    args = ap.parse_args()

    api = HfApi()
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet"))
    if not shards:
        sys.exit("ABORT: no parquet shard found in the repository")

    unknown = {shard_split(s) for s in shards} - set(SPLIT_DIRS)
    if unknown:
        sys.exit(f"ABORT: unexpected splits {sorted(unknown)}")

    if args.limit_shards:
        shards = shards[:args.limit_shards]

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    done_dir = os.path.join(args.out, "_done")
    os.makedirs(done_dir, exist_ok=True)
    for d in set(SPLIT_DIRS.values()):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
        os.makedirs(os.path.join(args.out, "_masks", d), exist_ok=True)

    total_rows = 0
    total_bytes = 0
    for n, path in enumerate(shards, 1):
        stem = os.path.basename(path)[:-len(".parquet")]
        marker = os.path.join(done_dir, stem)
        if os.path.exists(marker):
            print(f"[{n}/{len(shards)}] {stem} already done")
            continue

        split_dir = SPLIT_DIRS[shard_split(path)]
        img_dir = os.path.join(args.out, split_dir)
        msk_dir = os.path.join(args.out, "_masks", split_dir)

        reader = con.execute(QUERY.format(repo=REPO, path=path)).to_arrow_reader(64)
        rows = 0
        written = 0
        for batch in reader:
            cols = batch.to_pydict()
            for rgb, sem, labels in zip(cols["rgb_bytes"], cols["sem_bytes"],
                                        cols["labels"]):
                if rgb is None or sem is None:
                    sys.exit(f"ABORT: missing bytes in {stem}, row {rows}")
                name = f"{stem}_{rows:03d}"
                for blob, target in ((rgb, os.path.join(img_dir, name + ".png")),
                                     (sem, os.path.join(msk_dir, name + ".png"))):
                    with open(target, "wb") as fh:
                        fh.write(blob)
                    written += len(blob)
                with open(os.path.join(msk_dir, name + ".json"), "w") as fh:
                    fh.write(labels if isinstance(labels, str) else json.dumps(labels))
                rows += 1

        open(marker, "w").close()
        total_rows += rows
        total_bytes += written
        print(f"[{n}/{len(shards)}] {stem} -> {split_dir}  {rows} frames  "
              f"{written/1e6:.0f} MB  (running total {total_bytes/1e9:.2f} GB)")

    print(f"\n{total_rows} frames written, {total_bytes/1e9:.2f} GB on disk")


if __name__ == "__main__":
    main()
