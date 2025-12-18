#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import get_token, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a checkpoint from the Hugging Face Hub into model_checkpoints/profam-1"
    )
    parser.add_argument(
        "--repo-id",
        default="judewells/ProFam-1",
        help="Repository id on HF Hub, e.g. judewells/ProFam-1",
    )
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset"],
        default="model",
        help="HF repo type (default: model)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit to download (default: main)",
    )
    parser.add_argument(
        "--target-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "model_checkpoints" / "profam-1"
        ),
        help="Destination directory (default: model_checkpoints/profam-1)",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional allow patterns to limit files (e.g. *.ckpt config.yaml)",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*/.git*", "*/.DS_Store"],
        help="Optional ignore patterns",
    )
    return parser.parse_args()


def ensure_token() -> str | None:
    return os.environ.get("HF_TOKEN") or get_token()


def main() -> None:
    args = parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    token = ensure_token()

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    # For clarity, print where files landed
    print(f"Downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
