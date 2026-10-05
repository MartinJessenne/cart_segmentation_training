"""Stream and extract RGB images and semantic masks from Hugging Face Hub.

Queries parquet shards directly over HTTPFS using DuckDB columnar reads,
downloading only 'rgb', 'semantic', and 'semantic_labels' while skipping the
heavy depth column (~1.3 MB/frame vs ~3.3 MB/frame raw).

Idempotent and resumable: records completed shards in <out>/_done/ so
interrupted downloads resume seamlessly without re-fetching.

Layout produced:
    <out>/train/<shard_stem>_<idx>.png          RGB images
    <out>/valid/<shard_stem>_<idx>.png
    <out>/test/<shard_stem>_<idx>.png
    <out>/_masks/<split>/<shard_stem>_<idx>.png Mask images
    <out>/_masks/<split>/<shard_stem>_<idx>.json Class ID -> Name mapping
"""
import argparse
import gc
import json
import os
import sys
import time
from huggingface_hub import HfApi

SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}

QUERY = """
SELECT struct_extract(rgb, 'bytes')      AS rgb_bytes,
       struct_extract(semantic, 'bytes') AS sem_bytes,
       semantic_labels                   AS labels
FROM read_parquet('hf://datasets/{repo}/{path}')
"""


def shard_split(name: str) -> str:
    """Extract split name from shard path convention (e.g. data/train-00000-...)."""
    return name.rsplit("/", 1)[-1].split("-")[0]


def main():
    ap = argparse.ArgumentParser(description="Stream RGB images and masks from Hugging Face dataset")
    ap.add_argument("--repo", required=True,
                    help="Hugging Face dataset repository ID (e.g. 'org/dataset_name')")
    ap.add_argument("--out", default="_dataset_raw",
                    help="Output directory for downloaded images and masks")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="Limit processing to N shards (for testing/debugging; 0 = all)")
    ap.add_argument("--token", default=None,
                    help="Hugging Face access token (defaults to HF_TOKEN env var)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"==> Querying shards from Hugging Face repo: {args.repo}...")
    try:
        shards = sorted(f for f in api.list_repo_files(args.repo, repo_type="dataset")
                        if f.startswith("data/") and f.endswith(".parquet"))
    except Exception as e:
        sys.exit(f"ABORT: Failed to list files in repo '{args.repo}': {e}")

    if not shards:
        sys.exit(f"ABORT: No parquet shards found in 'data/' inside repo '{args.repo}'")

    unknown = {shard_split(s) for s in shards} - set(SPLIT_DIRS)
    if unknown:
        sys.exit(f"ABORT: Unexpected splits found in shards: {sorted(unknown)}")

    if args.limit_shards > 0:
        shards = shards[:args.limit_shards]

    print(f"==> Found {len(shards)} shards to process.")

    import duckdb

    def make_connection():
        c = duckdb.connect()
        c.execute("INSTALL httpfs; LOAD httpfs;")
        c.execute("SET http_timeout=30; SET http_retries=3; SET http_keep_alive=false;")
        c.execute("SET max_memory='2GB';")
        c.execute("SET preserve_insertion_order=false;")
        if token:
            c.execute(f"SET hf_token='{token}';")
        return c

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
            print(f"[{n}/{len(shards)}] {stem} already downloaded, skipping.")
            continue

        split_dir = SPLIT_DIRS[shard_split(path)]
        img_dir = os.path.join(args.out, split_dir)
        msk_dir = os.path.join(args.out, "_masks", split_dir)

        shard_done = False
        for attempt in range(1, 4):
            con = make_connection()
            try:
                reader = con.execute(QUERY.format(repo=args.repo, path=path)).to_arrow_reader(64)
                rows = 0
                written = 0
                for batch in reader:
                    cols = batch.to_pydict()
                    for rgb, sem, labels in zip(cols["rgb_bytes"], cols["sem_bytes"], cols["labels"]):
                        if rgb is None or sem is None:
                            sys.exit(f"ABORT: Missing image bytes in {stem}, row {rows}")
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
                print(f"[{n}/{len(shards)}] {stem} -> {split_dir} ({rows} frames, "
                      f"{written/1e6:.1f} MB, running total: {total_bytes/1e9:.2f} GB)")
                shard_done = True
                break
            except Exception as e:
                print(f"[{n}/{len(shards)}] {stem} error on attempt {attempt}/3: {e}, retrying...", flush=True)
                time.sleep(3)
            finally:
                try:
                    con.close()
                except Exception:
                    pass
                del con
                gc.collect()

        if not shard_done:
            sys.exit(f"ABORT: Failed to fetch {stem} after 3 attempts")

    print(f"\n==> Finished! {total_rows} frames downloaded ({total_bytes/1e9:.2f} GB on disk) -> {args.out}")


if __name__ == "__main__":
    main()
