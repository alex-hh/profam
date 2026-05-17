"""CLI entry point for ``profam generate``.

Thin wrapper over :meth:`profam.ProFam.generate`: loads the model once,
iterates over the input FASTA(s), and delegates the sampler and
preprocessor wiring to the public Python API. The CLI adds orchestration
concerns only — glob/file discovery, task sharding, and writing
generated sequences (plus the family-context prompts that were actually
used) to disk.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Sequence

import torch

from profam.api import ProFam
from profam.constants import resolve_runtime_path
from profam.sequence.fasta import read_fasta
from profam.utils.utils import seed_all


def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequences from family prompts"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_checkpoints/profam-1/checkpoints/last.ckpt",
        help="Path to the .ckpt checkpoint file",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path or glob for input sequence files (FASTA / a2m / a3m)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs",
        help="Directory to save generated FASTA files",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="single",
        choices=["ensemble", "single"],
        help="Sampler type: ensemble or single",
    )
    parser.add_argument("--num_prompts_in_ensemble", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_generated_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability mass (0<p<=1)",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean_probs",
        choices=["mean_probs", "sum_log_probs"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--max_sequence_length_multiplier",
        type=float,
        default=1.2,
        help=(
            "Limits the maximum generated length to be no longer than this factor "
            "longer than the longest sequence in the prompt"
        ),
    )
    parser.add_argument(
        "--minimum_sequence_length_proportion",
        type=float,
        default=0.5,
        help=(
            "Discard sequences that end with length < this proportion times the minimum "
            "sequence length in the prompt"
        ),
    )
    parser.add_argument(
        "--minimum_sequence_identity",
        type=float,
        default=None,
        help="Discard sequences whose aligned identity is below this threshold",
    )
    parser.add_argument(
        "--maximum_retries",
        type=int,
        default=5,
        help=(
            "If a sequence is aborted by filters, retry up to this many times before "
            "returning the last attempt"
        ),
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=1,
        help=(
            "Number of sequences generated in parallel per model.generate() "
            "call (single sampler only). 1 (default) reproduces sequential "
            "generation; values > 1 trade memory for throughput"
        ),
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="Task index when passing multiple files to process",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Number of tasks when passing multiple files to process",
    )
    parser.add_argument(
        "--disable_repeat_guard",
        action="store_true",
        default=False,
        help="Disable repeat guard retries when too many repeats are detected",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Override attention implementation before model init",
    )
    parser.add_argument(
        "--auto_download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically download the default checkpoint if missing",
    )
    return parser.parse_args(argv)


def _resolve_input_files(file_path: str) -> list[str]:
    """Expand a glob (or bare path) into a sorted list of existing files."""
    pattern = os.path.expanduser(file_path)
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0 and not glob.has_magic(pattern):
        resolved = resolve_runtime_path(pattern)
        if resolved.exists():
            matches = [str(resolved)]
    return matches


def _shard_files(
    files: list[str], task_index: int | None, num_tasks: int | None
) -> list[str]:
    if task_index is None or num_tasks is None:
        return files
    batch_size = len(files) // num_tasks
    start = task_index * batch_size
    end = len(files) if task_index == num_tasks - 1 else start + batch_size
    shard = files[start:end]
    print(f"Processing {len(shard)} files in task {task_index} of {num_tasks}")
    for fpath in shard:
        print(fpath)
    return shard


def _prompt_from_fasta(path: str) -> tuple[list[str], list[str]]:
    """Read a FASTA/MSA file into (accessions, sequences) with insertions kept."""
    names, seqs = read_fasta(path, keep_insertions=True, to_upper=True, keep_gaps=False)
    return names, seqs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repeat_guard = not args.disable_repeat_guard
    if args.max_tokens > 8192:
        raise ValueError(
            "Max tokens must be less than or equal to 8192: model was only trained "
            "up to 8192 tokens."
        )

    seed_all(args.seed)

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    input_files = _resolve_input_files(args.prompt_file)
    input_files = _shard_files(input_files, args.task_index, args.num_tasks)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No input files matched pattern: {args.prompt_file}")

    pf = ProFam(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        auto_download=args.auto_download,
    )

    for fasta_path in input_files:
        base = os.path.splitext(os.path.basename(fasta_path))[0]
        out_path = save_dir / f"{base}_generated_{args.sampler}.fasta"
        if out_path.exists():
            print(f"Skipping {fasta_path} because {out_path} already exists")
            continue
        try:
            accessions, sequences = _prompt_from_fasta(fasta_path)
            if len(sequences) == 0:
                print(f"Skipping {fasta_path}: no sequences found")
                continue

            result = pf.generate(
                prompt=sequences,
                prompt_accessions=accessions,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                max_generated_length=args.max_generated_length,
                max_sequence_length_multiplier=args.max_sequence_length_multiplier,
                temperature=args.temperature,
                top_p=args.top_p,
                sampler=args.sampler,
                num_prompts_in_ensemble=args.num_prompts_in_ensemble,
                reduction=args.reduction,
                minimum_sequence_length_proportion=args.minimum_sequence_length_proportion,
                minimum_sequence_identity=args.minimum_sequence_identity,
                maximum_retries=args.maximum_retries,
                repeat_guard=repeat_guard,
                generation_batch_size=args.generation_batch_size,
                seed=args.seed,
            )

            if result.conditioning_prompts is not None:
                if args.sampler == "ensemble":
                    for pi, cond in enumerate(result.conditioning_prompts):
                        cond_path = save_dir / f"{base}_conditioning_prompt_{pi}.fasta"
                        write_fasta(cond.sequences, cond.accessions, str(cond_path))
                    print(
                        f"Wrote {len(result.conditioning_prompts)} ensemble "
                        f"conditioning prompts -> {save_dir}"
                    )
                else:
                    cond = result.conditioning_prompts[0]
                    cond_path = save_dir / f"{base}_conditioning_prompt.fasta"
                    write_fasta(cond.sequences, cond.accessions, str(cond_path))
                    print(
                        f"Wrote {len(cond.sequences)} conditioning sequences -> "
                        f"{cond_path}"
                    )

            out_accessions = [
                f"{base}_sample_{i}_log_likelihood_{score:.3f}"
                for i, score in enumerate(result.scores)
            ]
            write_fasta(result.sequences, out_accessions, str(out_path))
            print(f"Wrote {len(result.sequences)} sequences -> {out_path}")
        except Exception as e:
            print(f"Error processing {fasta_path}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
