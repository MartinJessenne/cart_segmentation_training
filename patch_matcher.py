"""Skip the Hungarian solve when every image has exactly one target.

`_assign_compact_cost_matrix` calls scipy's `linear_sum_assignment` once per
image per query group. With `group_detr=13` that is 13 solver round-trips per
image, and the count scales with images rather than steps -- measured, raising
the batch from 32 to 128 moved throughput 71 -> 68 img/s, because a per-image
cost cannot be amortised by batching.

When an image has a single target the cost matrix has one column, and the
minimum-cost assignment of a one-column matrix is its argmin. The solver is
computing an answer that `torch.argmin` gives exactly, so for a batch whose
images all have one target the whole group collapses to one vectorised argmin
over the (queries x images) block.

This is exact, not an approximation. Batches with any other target count fall
through to scipy unchanged.

Idempotent; re-run after any `uv sync` that reinstalls rfdetr.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

MARKER = "_single_target_fast_path"

ANCHOR = """            group_indices = [
                linear_sum_assignment(grouped_cost_matrix[:, target_offsets[index] : target_offsets[index + 1]])
                for index in range(len(sizes))
            ]
"""

REPLACEMENT = """            if _single_target_fast_path(sizes):
                # One column per image: argmin IS the optimal assignment, so a
                # single vectorised reduction replaces len(sizes) solver calls.
                best_rows = grouped_cost_matrix.argmin(dim=0).tolist()
                group_indices = [
                    (np.array([best_rows[index]], dtype=np.int64), _SINGLE_TARGET_COLUMN)
                    for index in range(len(sizes))
                ]
            else:
                group_indices = [
                    linear_sum_assignment(grouped_cost_matrix[:, target_offsets[index] : target_offsets[index + 1]])
                    for index in range(len(sizes))
                ]
"""

HELPER = '''

_SINGLE_TARGET_COLUMN = np.zeros(1, dtype=np.int64)


def _single_target_fast_path(sizes) -> bool:
    """True when every image in the batch has exactly one target.

    The caller then owns a cost matrix with one column per image, whose optimal
    assignment is its argmin -- no Hungarian solve required.
    """
    return bool(sizes) and all(size == 1 for size in sizes)

'''


def target_file() -> Path:
    spec = importlib.util.find_spec("rfdetr.models.matcher")
    if spec is None or spec.origin is None:
        sys.exit("ABORT: rfdetr is not installed in this environment")
    return Path(spec.origin)


def self_test() -> None:
    """Random cost matrices, one target per image: fast path must equal scipy."""
    code = """
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from rfdetr.models.matcher import _single_target_fast_path

torch.manual_seed(0)
for trial in range(200):
    n_images = int(torch.randint(1, 9, (1,)))
    n_queries = int(torch.randint(2, 40, (1,)))
    cost = torch.randn(n_queries, n_images)
    sizes = [1] * n_images
    assert _single_target_fast_path(sizes)
    fast = cost.argmin(dim=0).tolist()
    for i in range(n_images):
        rows, cols = linear_sum_assignment(cost[:, i:i + 1])
        assert list(cols) == [0], cols
        assert rows[0] == fast[i], (trial, i, rows[0], fast[i])
assert not _single_target_fast_path([1, 2, 1])
assert not _single_target_fast_path([])
print("self-test ok: fast path matches scipy on 200 random batches")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = target_file()
    source = path.read_text()
    if MARKER in source:
        print(f"already patched: {path}")
        if not args.check:
            self_test()
        return
    if args.check:
        print(f"{path}: NOT patched")
        sys.exit(1)
    if ANCHOR not in source:
        sys.exit(f"ABORT: expected assignment loop not found in {path}; "
                 "rfdetr has changed, re-read _assign_compact_cost_matrix.")

    marker = "class HungarianMatcher("
    if marker not in source:
        marker = "def _assign_compact_cost_matrix("
    source = source.replace(marker, HELPER.lstrip("\n") + "\n" + marker, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    path.write_text(source)
    print(f"patched: {path}")
    self_test()


if __name__ == "__main__":
    main()
