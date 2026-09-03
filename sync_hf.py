"""Automated Hugging Face checkpoint synchronization daemon.

Continuously monitors the training output directory (default: output/seg_nano_480)
and synchronizes last.ckpt, checkpoint_best_ema.pth, and metrics files to
Hugging Face (default: UItraviolet/cart_segmentation_rfdetr) at a regular
interval (default: 120s). Survives network dropouts and spot terminations.
"""
import argparse
import fnmatch
import os
import signal
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo, get_token

SYNC_PATTERNS = [
    "last.ckpt",
    "checkpoint_best_ema.pth",
    "checkpoint_best_total.pth",
    "metrics.csv",
    "training_summary.json",
    "*.onnx",
]


def match_any(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)


def main():
    ap = argparse.ArgumentParser(description="Hugging Face checkpoint sync daemon")
    ap.add_argument("--watch-dir", default="output/seg_nano_480",
                    help="Directory to watch for checkpoints")
    ap.add_argument("--repo", default="UItraviolet/cart_segmentation_rfdetr",
                    help="Target Hugging Face model repository")
    ap.add_argument("--interval", type=int, default=120,
                    help="Sync interval in seconds (default: 120)")
    ap.add_argument("--once", action="store_true",
                    help="Run one sync pass and exit")
    args = ap.parse_args()

    watch_path = Path(args.watch_dir)
    run_name = watch_path.name
    token = os.getenv("HF_TOKEN") or get_token()

    if not token:
        print("[sync_hf] WARNING: No HF_TOKEN found in env or cache. Ensure authentication is set.",
              file=sys.stderr)

    api = HfApi(token=token)
    try:
        create_repo(args.repo, exist_ok=True, repo_type="model", private=True, token=token)
        print(f"[sync_hf] Target repo verified: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"[sync_hf] Target repo check/creation note: {e}", file=sys.stderr)

    print(f"[sync_hf] Monitoring '{watch_path}' every {args.interval}s -> {args.repo}/{run_name}")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        print(f"\n[sync_hf] Received signal {sig}, exiting gracefully after current cycle...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Tracks {filepath: (mtime, size)} to prevent redundant uploads
    uploaded_state: dict[Path, tuple[float, int]] = {}

    while running:
        if watch_path.exists() and watch_path.is_dir():
            for entry in watch_path.rglob("*"):
                if not entry.is_file():
                    continue
                if not match_any(entry.name, SYNC_PATTERNS):
                    continue

                try:
                    stat = entry.stat()
                    mtime = stat.st_mtime
                    size = stat.st_size

                    # Ignore empty files or files modified in the last 5 seconds (prevent uploading during active write)
                    if size == 0 or (time.time() - mtime < 5):
                        continue

                    # Verify file size stability (ensure process finished writing before upload starts)
                    time.sleep(1)
                    if entry.stat().st_size != size:
                        continue

                    prev_state = uploaded_state.get(entry)
                    if prev_state and prev_state == (mtime, size):
                        continue  # unchanged

                    rel_path = entry.relative_to(watch_path)
                    path_in_repo = f"{run_name}/{rel_path.as_posix()}"

                    print(f"[sync_hf] Uploading {entry.name} ({size / (1024*1024):.1f} MB) -> {path_in_repo}...", flush=True)
                    api.upload_file(
                        path_or_fileobj=str(entry),
                        path_in_repo=path_in_repo,
                        repo_id=args.repo,
                        repo_type="model",
                        token=token,
                    )
                    uploaded_state[entry] = (mtime, size)
                    print(f"[sync_hf] Successfully uploaded {entry.name}", flush=True)
                except Exception as e:
                    print(f"[sync_hf] Upload failed for {entry.name}: {e}", file=sys.stderr, flush=True)

        if args.once or not running:
            break

        time.sleep(args.interval)

    print("[sync_hf] Daemon stopped.")


if __name__ == "__main__":
    main()
