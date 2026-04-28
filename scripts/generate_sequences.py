"""Backward-compatibility wrapper for the pre-package ``generate_sequences``
script.

This file preserves the legacy CLI contract (notably
``--checkpoint_dir``, the demo default ``--file_path``, and the
``hf_download_checkpoint.py`` hint on a missing checkpoint). All core
logic lives in :mod:`profam.cli.generate_sequences`, which is itself a
thin wrapper over :meth:`profam.ProFam.generate`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

from profam.cli.generate_sequences import main as _cli_main

_DEFAULT_CHECKPOINT_DIR = "model_checkpoints/profam-1"
_DEFAULT_FILE_PATH = "data/generate_sequences_example/4_1_1_39_cluster.filtered.fasta"


def _split_legacy_args(
    argv: Sequence[str],
) -> tuple[str | None, List[str]]:
    """Pull the legacy ``--checkpoint_dir`` flag out of ``argv``.

    Returns the checkpoint directory (or ``None`` if not provided) and
    the remaining args to forward to the CLI. Supports both
    ``--checkpoint_dir X`` and ``--checkpoint_dir=X``.
    """
    checkpoint_dir: str | None = None
    remaining: List[str] = []
    it = iter(argv)
    for arg in it:
        if arg == "--checkpoint_dir":
            checkpoint_dir = next(it, None)
        elif arg.startswith("--checkpoint_dir="):
            checkpoint_dir = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
    return checkpoint_dir, remaining


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    if "-h" in raw or "--help" in raw:
        parser = argparse.ArgumentParser(
            description=(
                "Legacy wrapper. Accepts --checkpoint_dir in addition to the flags "
                "exposed by `profam generate`; all core logic lives in "
                "profam.cli.generate_sequences."
            ),
            add_help=True,
        )
        parser.add_argument(
            "--checkpoint_dir",
            default=_DEFAULT_CHECKPOINT_DIR,
            help=(
                "Run directory containing `checkpoints/last.ckpt`. The wrapper "
                "resolves this to a full .ckpt path before delegating."
            ),
        )
        parser.parse_known_args(raw)

    checkpoint_dir, forwarded = _split_legacy_args(raw)
    if checkpoint_dir is None:
        checkpoint_dir = _DEFAULT_CHECKPOINT_DIR

    ckpt_path = os.path.join(checkpoint_dir, "checkpoints/last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run `python scripts/hf_download_checkpoint.py` to download the checkpoint."
        )

    if not any(a == "--checkpoint" or a.startswith("--checkpoint=") for a in forwarded):
        forwarded.extend(["--checkpoint", ckpt_path])

    # The new CLI renamed --file_path to --prompt_file; translate the
    # legacy flag so existing callers of this wrapper keep working.
    forwarded = [
        ("--prompt_file" if a == "--file_path" else a)
        if not a.startswith("--file_path=")
        else "--prompt_file=" + a.split("=", 1)[1]
        for a in forwarded
    ]
    if not any(
        a == "--prompt_file" or a.startswith("--prompt_file=") for a in forwarded
    ):
        forwarded.extend(["--prompt_file", _DEFAULT_FILE_PATH])

    if not any(a in ("--auto_download", "--no-auto_download") for a in forwarded):
        forwarded.append("--no-auto_download")

    return _cli_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
