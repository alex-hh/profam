import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

"""
Script to compute conditional likelihoods of candidate sequences given conditioning sequences.
Inputs: conditioning_sequences.fasta, candidate_sequences.fasta or .csv
Outputs: prints per-sequence mean log-likelihoods to stdout as CSV

Backwards-compatibility wrapper over the ``profam`` package. The CLI matches
the previous ``scripts/score_sequences.py`` contract; loading and scoring
delegate to ``profam.checkpoint.load_model`` / ``profam.scoring``.

The conditioning MSA is loaded with two co-registered views: an aligned
view (used to compute homology diversity weights) and an unaligned,
insertions-kept view (tokenized and fed to the model as the conditioning
prompt). This matches what ``ProFam.score`` does in the Python API; the
only functional difference is that this script caches diversity weights
on disk so large MSAs don't pay the Hamming cost on every run.
"""

from profam.checkpoint import load_model
from profam.cli.score_sequences import _load_conditioning_views
from profam.data.msa_subsampling import compute_homology_sequence_weights_with_cache
from profam.scoring import score_variants_ensemble
from profam.sequence.fasta import read_fasta
from profam.utils.utils import seed_all


def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute conditional likelihoods of candidate sequences given conditioning sequences"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="model_checkpoints/profam-1",
        help="Checkpoint run directory (contains checkpoints/last.ckpt)",
    )
    parser.add_argument(
        "--conditioning_fasta",
        type=str,
        default="data/score_sequences_example/CCDB_ECOLI_Adkar_2012.a3m",
        help="Path to conditioning FASTA/MSA file",
    )
    parser.add_argument(
        "--candidates_file",
        type=str,
        default="data/score_sequences_example/CCDB_ECOLI_Adkar_2012_subsample_250.csv",
        help="Path to candidate sequences FASTA file or csv file with columns: 'mutated_sequence', and optionally 'DMS_score'",
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
        help=(
            "Token budget used ONLY to dynamically set the scoring batch size to stay within memory "
            "constraints. This is typically higher than --max_tokens. "
        ),
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
        help="If set, sample conditioning sequences with homology-based diversity weights (1/#neighbors).",
    )
    parser.add_argument(
        "--diversity_theta",
        type=float,
        default=0.2,
        help="Theta used for homology neighbor definition when computing diversity weights.",
    )
    parser.add_argument(
        "--recompute_diversity_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, ignore any on-disk cached weights and recompute.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Override attention implementation before model init (e.g. flash_attention_2)",
    )
    args = parser.parse_args()

    seed_all(args.seed)

    # Load the conditioning MSA with both co-registered views (aligned for
    # weights, unaligned-with-insertions for model conditioning) before the
    # expensive model load, so bad input fails fast. Falls back to an
    # unaligned-only prompt if the file is ragged and diversity weighting is
    # disabled; raises with a helpful message otherwise.
    conditioning, aligned = _load_conditioning_views(
        args.conditioning_fasta, args.use_diversity_weights
    )

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run `python scripts/hf_download_checkpoint.py` to download the checkpoint."
        )

    model = load_model(
        checkpoint=ckpt_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        auto_download=False,
    )

    weights: Optional[np.ndarray] = None
    if args.use_diversity_weights:
        print(
            f"Computing diversity (homology) weights for {args.conditioning_fasta}...",
            file=sys.stderr,
        )
        weights = compute_homology_sequence_weights_with_cache(
            msa_file=args.conditioning_fasta,
            sequences=aligned,
            theta=args.diversity_theta,
            force_recalc=args.recompute_diversity_weights,
        )

    print(
        f"Tokenizing {len(conditioning)} conditioning sequences...",
        file=sys.stderr,
    )
    tokenized_conditioning_sequences = [
        model.tokenizer(seq, add_special_tokens=False)["input_ids"]
        for seq in conditioning
    ]

    # Read candidates
    dms_scores = None
    if args.candidates_file.endswith(".csv"):
        df = pd.read_csv(args.candidates_file)
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
            args.candidates_file, keep_insertions=False, to_upper=True
        )

    if len(cand_seqs) == 0:
        raise ValueError("No candidate sequences found")

    # Encode completions with BOS/EOS = [SEP]
    comp_tok = model.tokenizer.encode_completions(
        cand_seqs,
        bos_token=model.tokenizer.sep_token,
        eos_token=model.tokenizer.sep_token,
    )
    completion_ids = (
        torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(model.device)
    )  # (1, n, L)

    with torch.no_grad():
        lls = score_variants_ensemble(
            model=model,
            completion_ids=completion_ids,
            tokenized_conditioning_sequences=tokenized_conditioning_sequences,
            ensemble_size=args.ensemble_number,
            scoring_max_tokens=args.scoring_max_tokens,
            start_tokens=[47, 63],
            max_tokens_override=args.max_tokens,
            weights=weights,
            seed=args.seed,
        )

    # Output handling
    os.makedirs(args.save_dir, exist_ok=True)
    candidate_basename = os.path.splitext(os.path.basename(args.candidates_file))[0]

    csv_path = os.path.join(args.save_dir, f"{candidate_basename}_scores.csv")
    json_path = os.path.join(args.save_dir, f"{candidate_basename}_metadata.json")

    df_out = pd.DataFrame(
        {"id": cand_names, "mutated_sequence": cand_seqs, "score": lls.tolist()}
    )
    if dms_scores is not None:
        df_out["DMS_score"] = dms_scores
    df_out.to_csv(csv_path, index=False)

    print(df_out[["id", "mutated_sequence", "score"]].to_csv(index=False))
    print(f"Scores saved to {csv_path}...")

    # Calculate metrics
    corr = None
    if dms_scores is not None:
        corr, _ = spearmanr(lls, dms_scores)
        print(f"Spearman correlation: {corr}", file=sys.stderr)

    metadata = {
        "n_sequences_evaluated": len(cand_seqs),
        "ensemble_number": args.ensemble_number,
        "timestamp": datetime.now().isoformat(),
        "conditioning_fasta": args.conditioning_fasta,
        "n_conditioning_sequences": len(conditioning),
        "candidates_file": args.candidates_file,
        "mean_likelihood_score": float(np.mean(lls)),
        "spearman_correlation": float(corr) if corr is not None else None,
        "checkpoint": args.checkpoint_dir,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
