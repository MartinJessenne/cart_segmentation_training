"""Publish pre-processed 960x600 COCO dataset archive to Hugging Face.

Packages the clean, re-encoded dataset (_rfdetr_dataset_960) into a single
tar.gz archive (~4.5-5.3 GB) and uploads it to Hugging Face Hub dataset repo.
This makes bootstrapping on fresh ephemeral instances instantaneous (~60s),
eliminating 50 minutes of raw parquet downloads and CPU image re-encoding.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, get_token

DEFAULT_REPO = "UItraviolet/cart_segmentation_coco_960"
ARCHIVE_NAME = "cart_rfdetr_960.tar.gz"


def main():
    ap = argparse.ArgumentParser(description="Package and publish RF-DETR COCO dataset to HF Hub")
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset_960",
                    help="Pre-processed dataset directory to bundle")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="Target Hugging Face dataset repository")
    ap.add_argument("--archive", default=ARCHIVE_NAME,
                    help="Output tar.gz archive filename")
    ap.add_argument("--private", action="store_true",
                    help="Create private HF dataset repo instead of public")
    args = ap.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        sys.exit(f"ABORT: dataset directory '{dataset_path}' does not exist.")

    train_coco = dataset_path / "train" / "_annotations.coco.json"
    if not train_coco.exists():
        sys.exit(f"ABORT: '{train_coco}' missing. Dataset is incomplete or un-converted.")

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"==> Packaging '{dataset_path}' into '{archive_path}' (pigz/gzip)...")
        # Use pigz if available for fast multi-core compression, otherwise standard tar -czf
        has_pigz = subprocess.run(["which", "pigz"], capture_output=True).returncode == 0
        if has_pigz:
            tar_cmd = ["tar", "-I", "pigz", "-cf", str(archive_path), "-C", str(dataset_path.parent), dataset_path.name]
        else:
            tar_cmd = ["tar", "-czf", str(archive_path), "-C", str(dataset_path.parent), dataset_path.name]

        subprocess.run(tar_cmd, check=True)
        size_gb = archive_path.stat().st_size / (1024 ** 3)
        print(f"==> Archive created successfully: {archive_path.name} ({size_gb:.2f} GB)")
    else:
        size_gb = archive_path.stat().st_size / (1024 ** 3)
        print(f"==> Found existing archive '{archive_path.name}' ({size_gb:.2f} GB), skipping tar step.")

    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        sys.exit("ABORT: HF_TOKEN not found in environment or ~/.cache/huggingface/token.")

    api = HfApi(token=token)
    print(f"==> Ensuring Hugging Face dataset repository '{args.repo}' exists...")
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    print(f"==> Uploading '{archive_path.name}' ({size_gb:.2f} GB) to 'hf://datasets/{args.repo}'...")
    api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=archive_path.name,
        repo_id=args.repo,
        repo_type="dataset",
        token=token,
    )

    print(f"\nSUCCESS! Dataset published: https://huggingface.co/datasets/{args.repo}")
    print("\nOn any fresh remote machine, setup takes 60 seconds:")
    print(f"  hf download {args.repo} {archive_path.name} --repo-type dataset --local-dir .")
    print(f"  tar -xzf {archive_path.name}")


if __name__ == "__main__":
    main()
