"""
Download selected columns from a Hugging Face parquet dataset using DuckDB.

Only the 'rgb', 'semantic', and 'semantic_labels' columns are pulled and
written to ./cart_dataset/ as parquet.

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
# in every split/config; narrow it (e.g. "data/train-*.parquet") if you only
# want one split. Check the "Files and versions" tab on HF to see the layout.
DATASET_GLOB = "**/*.parquet"

COLUMNS = ["rgb", "semantic", "semantic_labels"]
OUTPUT_DIR = Path("cart_dataset")
OUTPUT_FILE = OUTPUT_DIR / "data.parquet"

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


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = load_hf_token()
    con = build_connection(token)

    source_uri = f"hf://datasets/{DATASET_REPO}/{DATASET_GLOB}"
    col_list = ", ".join(COLUMNS)

    log.info("Source: %s", source_uri)
    log.info("Columns requested: %s", col_list)
    log.info("Output: %s", OUTPUT_FILE.resolve())

    # Quick peek at row count first (also validates the query before the
    # full copy runs).
    log.info("Checking dataset (row count)...")
    t0 = time.time()
    try:
        count = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{source_uri}')
        """).fetchone()[0]
    except duckdb.Error as e:
        log.error("Failed to read dataset schema/rows: %s", e)
        sys.exit(1)
    log.info("Dataset has %s rows (checked in %.1fs).", f"{count:,}", time.time() - t0)

    log.info("Starting download + column selection (progress bar below)...")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT {col_list}
            FROM read_parquet('{source_uri}')
        ) TO '{OUTPUT_FILE.as_posix()}' (FORMAT PARQUET);
    """)
    elapsed = time.time() - t0

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    log.info("Done in %.1fs. Wrote %.2f MB to %s", elapsed, size_mb, OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    download()