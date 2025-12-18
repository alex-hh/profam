import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rootutils
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

"""
Script to compute conditional likelihoods of candidate sequences given conditioning sequences.
Inputs: conditioning_sequences.fasta, candidate_sequences.fasta or .csv
Outputs: prints per-sequence mean log-likelihoods to stdout as CSV
"""

from src.data.msa_subsampling import compute_homology_sequence_weights_with_cache
from src.data.objects import ProteinDocument
from src.models.llama import LlamaLitModule
from src.sequence.fasta import read_fasta
from src.utils.utils import seed_all


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


def score_variants_ensemble(
    model: LlamaLitModule,
    completion_ids: torch.Tensor,
    tokenized_conditioning_sequences: List[List[int]],
    ensemble_size: int,
    scoring_max_tokens: int,
    start_tokens: Optional[list[int]] = None,
    max_tokens_override: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
):
    """
    Computes the mean log-likelihood of candidate sequences using an ensemble of prompts
    sampled from the conditioning sequences (context).
    """
    if start_tokens is None:
        start_tokens = [47, 63]
    random.seed(42)
    rng = random.Random(42)
    rng_np = np.random.default_rng(42)

    # Pre-calculate stats
    seq_lengths = [len(seq) for seq in tokenized_conditioning_sequences]
    total_seqs = len(seq_lengths)
    completion_length = completion_ids.shape[-1]

    max_tokens = (
        max_tokens_override if max_tokens_override is not None else model.max_tokens
    )
    max_context_tokens = (max_tokens - completion_length) - 5

    avg_seq_len = sum(seq_lengths) / len(seq_lengths) if len(seq_lengths) > 0 else 0
    min_seq_len = min(seq_lengths) if len(seq_lengths) > 0 else 0
    assumed_seq_len = (min_seq_len + avg_seq_len) / 2

    max_n_by_tokens = (
        max(0, min(int(max_context_tokens // assumed_seq_len) + 2, total_seqs))
        if avg_seq_len > 0
        else 0
    )

    # find range of n_opt values that are in the target likelihood range (heuristic range here):
    lower_bound = min(max_n_by_tokens, 2)
    upper_bound = min(max_n_by_tokens, total_seqs)
    vals_in_range = list(np.arange(lower_bound, upper_bound + 1, dtype=int))
    if len(vals_in_range) == 0:
        vals_in_range = [0]

    n_opt = int(rng.choice(vals_in_range))
    n_seqs_list = []
    variant_lls: List[np.ndarray] = []
    token_count_attempts = 100

    if completion_length + 2 > max_tokens:
        n_opt = 0
        repeats = 1
    else:
        repeats = min(ensemble_size, total_seqs) if total_seqs > 0 else 1

    sep_token_id = model.tokenizer.sep_token_id
    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        p = (w / s) if s > 0 else None
    for rep in tqdm(
        range(repeats),
        desc="Scoring sequences",
        unit="prompt",
        file=sys.stderr,
    ):
        while True:
            if n_opt == 0 and 0 in n_seqs_list:
                if len(vals_in_range) > 0:
                    n_opt = int(random.choice(vals_in_range))
                else:
                    n_opt = 0  # Stuck at 0
                    break

            if total_seqs > 0:
                idxs = rng_np.choice(
                    np.arange(total_seqs),
                    size=min(n_opt, total_seqs),
                    replace=False,
                    p=p,
                ).tolist()
                rng.shuffle(idxs)
                tok_cnt = sum(seq_lengths[i] for i in idxs)
            else:
                idxs = []
                tok_cnt = 0

            prompt_len_estimate = len(start_tokens) + tok_cnt + len(idxs)

            if prompt_len_estimate + completion_length <= max_tokens:
                break
            else:
                n_opt = max(0, n_opt - 1)  # Try a smaller number of sequences

        # Build prompt
        if n_opt == 0 or len(idxs) == 0:
            prompt_ids_list = []  # No context
        else:
            prompt_ids_list = list(start_tokens)
            for i, idx in enumerate(idxs):
                prompt_ids_list.extend(tokenized_conditioning_sequences[idx])
                if i < len(idxs) - 1:
                    prompt_ids_list.append(sep_token_id)

        if len(prompt_ids_list) > 0:
            input_ids = torch.tensor(
                prompt_ids_list, dtype=torch.long, device=model.device
            ).unsqueeze(0)
        else:
            input_ids = None

        L = completion_ids.shape[-1]
        L_prompt = 0 if input_ids is None else input_ids.shape[-1]

        completion_ids_device = completion_ids.to(model.device)

        lls = model.score_seqs(
            input_ids,
            completion_ids_device,
            use_cache=getattr(model, "use_kv_cache_for_scoring", True),
            batch_size=max(
                int(scoring_max_tokens) // (L + L_prompt),
                1,
            )
            if getattr(model, "use_kv_cache_for_scoring", True)
            else 1,
        )

        variant_lls.append(lls)
        n_seqs_list.append(n_opt)

        if len(vals_in_range) > 0:
            n_opt = rng.choice(vals_in_range)

    lls_array = np.stack(variant_lls, axis=0)
    # Return per-sequence mean log-likelihood across variants
    mean_lls_per_sequence = lls_array.mean(axis=0)

    return mean_lls_per_sequence


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
        default="data/score_sequences_example/CCDB_ECOLI_Adkar_2012.csv",
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

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run `python scripts/hf_download_checkpoint.py` to download the checkpoint."
        )
    attn_impl = args.attn_implementation

    try:
        import flash_attn
    except ImportError:
        if attn_impl == "flash_attention_2":
            raise ImportError(
                "Flash attention is not installed. "
                "select an alternative attention implementation such as:\n`--attn_implementation sdpa`.\n"
                "Or install it with:\n`pip install flash-attn --no-build-isolation`. "
            )

    try:
        ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hyper_params = ckpt_blob.get("hyper_parameters", {})
        cfg_obj = hyper_params.get("config", None)
        if cfg_obj is None:
            raise RuntimeError(
                "Could not find 'config' in checkpoint hyper_parameters to override attn implementation"
            )
        setattr(cfg_obj, "attn_implementation", attn_impl)
        setattr(cfg_obj, "_attn_implementation", attn_impl)
        # We handle ensemble size explicitly now, but setting it here doesn't hurt
        if hasattr(cfg_obj, "gym_subsamples_per_n"):
            setattr(cfg_obj, "gym_subsamples_per_n", args.ensemble_number)
        model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(
            ckpt_path, config=cfg_obj, strict=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to override attention implementation: {e}")
    model.eval()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model.to(args.device, dtype=dtype_map[args.dtype])

    # Build ProteinDocument objects (just to read sequences nicely)
    cond_doc = build_pool_from_fasta(args.conditioning_fasta)

    weights = None
    if args.use_diversity_weights:
        print(
            f"Computing diversity (homology) weights for {args.conditioning_fasta}...",
            file=sys.stderr,
        )
        _, aligned_sequences = read_fasta(
            args.conditioning_fasta,
            keep_insertions=False,
            to_upper=True,
            keep_gaps=True,
        )
        weights = compute_homology_sequence_weights_with_cache(
            msa_file=args.conditioning_fasta,
            sequences=aligned_sequences,
            theta=args.diversity_theta,
            force_recalc=args.recompute_diversity_weights,
        )

    # Tokenize conditioning sequences individually
    print(
        f"Tokenizing {len(cond_doc.sequences)} conditioning sequences...",
        file=sys.stderr,
    )
    # Using the tokenizer directly on strings to get IDs.
    # NOTE: verify if we need spaces or not. The tokenizer in debug worked on "ACDEFGH".
    tokenized_conditioning_sequences = [
        model.tokenizer(
            seq.upper().replace("-", "").replace(".", ""), add_special_tokens=False
        )["input_ids"]
        for seq in cond_doc.sequences
    ]

    # Read candidates
    dms_scores = None
    if args.candidates_file.endswith(".csv"):
        df = pd.read_csv(args.candidates_file)
        if "mutated_sequence" not in df.columns:
            raise ValueError("CSV must have 'mutated_sequence' column")
        # Ensure upper case
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
        "n_conditioning_sequences": len(cond_doc.sequences),
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
