import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from profam.constants import resolve_runtime_path
from profam.data.msa_subsampling import compute_homology_sequence_weights_with_cache
from profam.data.objects import ProteinDocument
from profam.scoring import score_variants_ensemble
from profam.sequence.fasta import read_fasta
from profam.utils.utils import seed_all


def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def build_pool_from_fasta(path: str) -> ProteinDocument:
    names, seqs = read_fasta(path, keep_insertions=True, to_upper=True, keep_gaps=False)
    rep = names[0] if len(names) > 0 else "representative"
    return ProteinDocument(
        sequences=seqs,
        accessions=names,
        identifier=os.path.basename(path),
        representative_accession=rep,
    )


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
        "--conditioning_fasta",
        type=str,
        required=True,
        help="Path to conditioning FASTA/MSA file",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    seed_all(args.seed)

    from profam.checkpoint import load_model

    model = load_model(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    conditioning_fasta = str(resolve_runtime_path(args.conditioning_fasta))
    candidates_file = str(resolve_runtime_path(args.candidates_file))
    save_dir = Path(args.save_dir).expanduser().resolve()

    cond_doc = build_pool_from_fasta(conditioning_fasta)

    weights = None
    if args.use_diversity_weights:
        print(
            f"Computing diversity (homology) weights for {conditioning_fasta}...",
            file=sys.stderr,
        )
        _, aligned_sequences = read_fasta(
            conditioning_fasta,
            keep_insertions=False,
            to_upper=True,
            keep_gaps=True,
        )
        weights = compute_homology_sequence_weights_with_cache(
            msa_file=conditioning_fasta,
            sequences=aligned_sequences,
            theta=args.diversity_theta,
            force_recalc=args.recompute_diversity_weights,
        )

    print(
        f"Tokenizing {len(cond_doc.sequences)} conditioning sequences...",
        file=sys.stderr,
    )
    tokenized_conditioning_sequences = [
        model.tokenizer(
            seq.upper().replace("-", "").replace(".", ""), add_special_tokens=False
        )["input_ids"]
        for seq in cond_doc.sequences
    ]

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

    comp_tok = model.tokenizer.encode_completions(
        cand_seqs,
        bos_token=model.tokenizer.sep_token,
        eos_token=model.tokenizer.sep_token,
    )
    completion_ids = (
        torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(model.device)
    )

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
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the conditioning sequences that were used as context
    cond_basename = os.path.splitext(os.path.basename(conditioning_fasta))[0]
    prompt_path = save_dir / f"{cond_basename}_conditioning_used.fasta"
    write_fasta(
        list(cond_doc.sequences),
        list(cond_doc.accessions)
        if cond_doc.accessions
        else [f"cond_{i}" for i in range(len(cond_doc.sequences))],
        str(prompt_path),
    )
    print(
        f"Wrote {len(cond_doc.sequences)} conditioning sequences -> {prompt_path}",
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
        "conditioning_fasta": conditioning_fasta,
        "n_conditioning_sequences": len(cond_doc.sequences),
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
