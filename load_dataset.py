"""
Download selected columns from a Hugging Face parquet dataset using DuckDB,
preserving the train/validation/test split.

Only the 'rgb', 'semantic', and 'semantic_labels' columns are pulled and
written to ./cart_dataset/ as parquet, partitioned by split.

Setup:
    pip install duckdb python-dotenv
    cp .env.example .env   # then paste your HF token inside

Fill in DATASET_REPO / DATASET_GLOB below before running, then:
    python download_hf_dataset.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# CONFIG — edit these two lines for your dataset
# --------------------------------------------------------------------------
DATASET_REPO = "UItraviolet/industrial_cart"   # HF repo id, e.g. "org/dataset"
# Glob for the parquet files inside the repo. "**/*.parquet" grabs everything
# in every split/config. We rely on HF's standard shard naming convention
# (e.g. "train-00000-of-00001.parquet", "validation-...", "test-...") to
# recover the split — check "Files and versions" on HF to confirm your
# dataset follows this convention before relying on the regex below.
DATASET_GLOB = "**/*.parquet"

COLUMNS = ["rgb", "semantic", "semantic_labels"]
OUTPUT_DIR = Path("cart_dataset")

# Maps a chunk of the source filename to a normalized split name.
# Order matters: check longer/more specific patterns first if you add more.
SPLIT_PATTERNS = {
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
}

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf-duckdb")


def load_hf_token() -> str:
    load_dotenv()  # reads .env in the current working directory
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not found. Create a .env file (see .env.example) "
                   "with HF_TOKEN=hf_xxx, or export it in your shell.")
        sys.exit(1)
    log.info("Loaded HF_TOKEN from environment (%d chars).", len(token))
    return token


def build_connection(token: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    log.info("Installing/loading DuckDB extensions (httpfs)...")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # Register the HF token as a DuckDB secret so hf:// URIs authenticate.
    con.execute(f"""
        CREATE OR REPLACE SECRET hf_token (
            TYPE HUGGINGFACE,
            TOKEN '{token}'
        );
    """)
    log.info("Hugging Face secret registered with DuckDB.")

    # Progress bar in the terminal while queries run.
    con.execute("SET enable_progress_bar = true;")
    con.execute("SET enable_progress_bar_print = true;")
    return con


def build_split_case_expr() -> str:
    """
    Builds a SQL CASE expression that derives a normalized split name from
    the source filename, using SPLIT_PATTERNS. Falls back to 'unknown' if
    no pattern matches, so nothing silently vanishes.
    """
    lines = ["CASE"]
    for pattern, split_name in SPLIT_PATTERNS.items():
        lines.append(
            f"    WHEN regexp_matches(filename, '(^|[-_/])({pattern})([-_.]|$)') "
            f"THEN '{split_name}'"
        )
    lines.append("    ELSE 'unknown'")
    lines.append("END")
    return "\n".join(lines)


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = load_hf_token()
    con = build_connection(token)

    source_uri = f"hf://datasets/{DATASET_REPO}/{DATASET_GLOB}"
    col_list = ", ".join(COLUMNS)
    split_expr = build_split_case_expr()

    log.info("Source: %s", source_uri)
    log.info("Columns requested: %s", col_list)
    log.info("Output dir (partitioned by split): %s", OUTPUT_DIR.resolve())

    # filename=true exposes the source parquet shard path as a column,
    # which is what lets us recover the split without needing a 'split'
    # column to already exist in the data itself.
    read_expr = f"read_parquet('{source_uri}', filename=true)"

    # Quick peek: row count per derived split, before doing the full copy.
    # This also validates the regex against your actual filenames early,
    # instead of discovering a mismatch after paying for the full download.
    log.info("Checking dataset (row count per split)...")
    t0 = time.time()
    try:
        rows = con.execute(f"""
            SELECT
                {split_expr} AS split,
                COUNT(*) AS n
            FROM {read_expr}
            GROUP BY split
            ORDER BY split
        """).fetchall()
    except duckdb.Error as e:
        log.error("Failed to read dataset schema/rows: %s", e)
        sys.exit(1)

    if not rows:
        log.error("Query returned no rows — check DATASET_REPO/DATASET_GLOB.")
        sys.exit(1)

    for split_name, n in rows:
        log.info("  split=%-12s rows=%s", split_name, f"{n:,}")
    log.info("Row count check done in %.1fs.", time.time() - t0)

    if any(split_name == "unknown" for split_name, _ in rows):
        log.warning(
            "Some rows had a filename that didn't match any pattern in "
            "SPLIT_PATTERNS — they were tagged 'unknown'. Inspect "
            "%s to see the raw filenames.",
            f"SELECT DISTINCT filename FROM {read_expr}",
        )

    log.info("Starting download + column selection (progress bar below)...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT
                {col_list},
                {split_expr} AS split
            FROM {read_expr}
        ) TO '{OUTPUT_DIR.as_posix()}' (
            FORMAT PARQUET,
            PARTITION_BY (split),
            OVERWRITE_OR_IGNORE
        );
    """)
    elapsed = time.time() - t0

    total_size_mb = sum(
        f.stat().st_size for f in OUTPUT_DIR.rglob("*.parquet")
    ) / (1024 * 1024)
    log.info(
        "Done in %.1fs. Wrote %.2f MB across split partitions under %s",
        elapsed, total_size_mb, OUTPUT_DIR,
    )
    for split_dir in sorted(OUTPUT_DIR.glob("split=*")):
        n_files = len(list(split_dir.glob("*.parquet")))
        log.info("  %s -> %d file(s)", split_dir.name, n_files)

    con.close()


if __name__ == "__main__":
    download()