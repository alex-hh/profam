"""CLI entry point for ``profam score``.

Thin wrapper over :class:`profam.ProFam.score`: loads the model once
and delegates scoring to the public Python API. The conditioning MSA
file is passed through unchanged; ``ProFam.score`` infers whether it
is aligned (so homology diversity weights are usable) or unaligned.
Diversity weights are cached on disk next to the MSA file so repeated
runs skip the Hamming step.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from profam.api import ProFam, _resolve_prompt
from profam.constants import resolve_runtime_path
from profam.sequence.fasta import read_fasta
from profam.utils.utils import seed_all


def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute conditional likelihoods of candidate sequences given conditioning sequences"
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
        help="Path to conditioning sequence file (FASTA / a2m / a3m)",
    )
    parser.add_argument(
        "--candidates_file",
        type=str,
        required=True,
        help="Path to candidate FASTA or CSV file with a 'mutated_sequence' column",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs", help="Directory to save output files"
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Token budget (prompt+completion) used for batch size heuristics",
    )
    parser.add_argument(
        "--scoring_max_tokens",
        type=int,
        default=64000,
        help="Token budget used only to dynamically set the scoring batch size",
    )
    parser.add_argument(
        "--ensemble_number",
        type=int,
        default=3,
        help="Number of prompts used to generate the ensemble score",
    )
    parser.add_argument(
        "--use_diversity_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample conditioning sequences with homology-based diversity weights",
    )
    parser.add_argument(
        "--diversity_theta",
        type=float,
        default=0.2,
        help="Theta used for homology neighbor definition when computing diversity weights",
    )
    parser.add_argument(
        "--recompute_diversity_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore any on-disk cached weights and recompute",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Override attention implementation before model init",
    )
    return parser.parse_args(argv)


def _load_conditioning_views(
    prompt_file: str, use_diversity_weights: bool
) -> Tuple[List[str], Optional[List[str]]]:
    """Load conditioning views from an MSA file, failing fast on bad input.

    Returns ``(conditioning, aligned_or_None)``. If ``use_diversity_weights``
    is requested but the file is not an aligned MSA, exits with a clear
    error pointing at ``--no-use_diversity_weights``.
    """
    try:
        conditioning, aligned, _ = _resolve_prompt(prompt_file, use_diversity_weights)
    except ValueError as exc:
        raise SystemExit(
            f"Cannot use diversity weights with {prompt_file!r}: "
            "it is not a valid aligned MSA (sequences have different "
            "lengths after stripping insertions). Either provide an "
            "aligned MSA file (FASTA / a2m / a3m with equal-length "
            "sequences after insertions are stripped), or re-run with "
            "--no-use_diversity_weights to score without weighting."
        ) from exc

    if aligned is None and not use_diversity_weights:
        print(
            f"{prompt_file!r} is not an aligned MSA; loading as "
            "unaligned sequences (diversity weights disabled).",
            file=sys.stderr,
        )
    return conditioning, aligned


def _load_candidates(candidates_file: str):
    """Return (names, sequences, dms_scores) from a CSV or FASTA file."""
    dms_scores = None
    if candidates_file.endswith(".csv"):
        df = pd.read_csv(candidates_file)
        if "mutated_sequence" not in df.columns:
            raise ValueError("CSV must have 'mutated_sequence' column")
        cand_seqs = df["mutated_sequence"].astype(str).str.upper().tolist()
        if "mutant" in df.columns:
            cand_names = df["mutant"].astype(str).tolist()
        else:
            cand_names = [f"seq_{i}" for i in range(len(cand_seqs))]
        if "DMS_score" in df.columns:
            dms_scores = df["DMS_score"].values
    else:
        cand_names, cand_seqs = read_fasta(
            candidates_file, keep_insertions=False, to_upper=True
        )

    if len(cand_seqs) == 0:
        raise ValueError("No candidate sequences found")

    return cand_names, cand_seqs, dms_scores


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    seed_all(args.seed)

    prompt_file = str(resolve_runtime_path(args.prompt_file))
    candidates_file = str(resolve_runtime_path(args.candidates_file))
    save_dir = Path(args.save_dir).expanduser().resolve()

    # Validate the conditioning input up-front so we fail before loading
    # the model, and grab the conditioning view for the output dump.
    conditioning, _ = _load_conditioning_views(prompt_file, args.use_diversity_weights)
    cand_names, cand_seqs, dms_scores = _load_candidates(candidates_file)

    print(
        f"Loaded {len(conditioning)} conditioning sequences "
        f"and {len(cand_seqs)} candidates.",
        file=sys.stderr,
    )
    if args.use_diversity_weights:
        print(
            f"Computing diversity (homology) weights for {prompt_file}...",
            file=sys.stderr,
        )

    pf = ProFam(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    result = pf.score(
        sequences=cand_seqs,
        prompt=prompt_file,
        ensemble_size=args.ensemble_number,
        max_tokens=args.max_tokens,
        scoring_max_tokens=args.scoring_max_tokens,
        use_diversity_weights=args.use_diversity_weights,
        diversity_theta=args.diversity_theta,
        cache_weights=True,
        recompute_cached_weights=args.recompute_diversity_weights,
        seed=args.seed,
    )
    lls = result.scores

    save_dir.mkdir(parents=True, exist_ok=True)

    prompt_basename = os.path.splitext(os.path.basename(prompt_file))[0]
    cond_used_path = save_dir / f"{prompt_basename}_conditioning_used.fasta"
    write_fasta(
        conditioning,
        [f"cond_{i}" for i in range(len(conditioning))],
        str(cond_used_path),
    )
    print(
        f"Wrote {len(conditioning)} conditioning sequences -> {cond_used_path}",
        file=sys.stderr,
    )

    candidate_basename = os.path.splitext(os.path.basename(candidates_file))[0]
    csv_path = save_dir / f"{candidate_basename}_scores.csv"
    json_path = save_dir / f"{candidate_basename}_metadata.json"

    df_out = pd.DataFrame(
        {"id": cand_names, "mutated_sequence": cand_seqs, "score": lls.tolist()}
    )
    if dms_scores is not None:
        df_out["DMS_score"] = dms_scores
    df_out.to_csv(csv_path, index=False)

    print(df_out[["id", "mutated_sequence", "score"]].to_csv(index=False))
    print(f"Scores saved to {csv_path}", file=sys.stderr)

    corr = None
    if dms_scores is not None:
        corr, _ = spearmanr(lls, dms_scores)
        print(f"Spearman correlation: {corr}", file=sys.stderr)

    metadata: Dict[str, object] = {
        "n_sequences_evaluated": len(cand_seqs),
        "ensemble_number": args.ensemble_number,
        "timestamp": datetime.now().isoformat(),
        "prompt_file": prompt_file,
        "n_conditioning_sequences": len(conditioning),
        "candidates_file": candidates_file,
        "mean_likelihood_score": float(np.mean(lls)),
        "spearman_correlation": float(corr) if corr is not None else None,
        "checkpoint": args.checkpoint,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
