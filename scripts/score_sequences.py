import argparse
import os
import torch
import random
import numpy as np
from typing import Dict, List
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

"""
Script to compute conditional likelihoods of candidate sequences given conditioning sequences.
Inputs: conditioning_sequences.fasta, candidate_sequences.fasta
Outputs: prints per-sequence mean log-likelihoods to stdout
"""

from src.data.objects import ProteinDocument
from src.data.processors.preprocessing import ProteinDocumentPreprocessor, AlignedProteinPreprocessingConfig
from src.sequence.fasta import read_fasta
from src.models.llama import LlamaLitModule
from src.utils.utils import seed_all



def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def build_pool_from_fasta(path: str) -> ProteinDocument:
    names, seqs = read_fasta(path, keep_insertions=False, to_upper=True)
    rep = names[0] if len(names) > 0 else None
    return ProteinDocument(
        sequences=seqs,
        accessions=names,
        identifier=os.path.basename(path),
        representative_accession=rep,
    )


def _evaluate_and_save_variants_v10(
    model: LlamaLitModule,
    batch: Dict[str, torch.Tensor],
    start_tokens: list[int] = [47, 63],
    coverage_multiplier: float = 0.0,
    seq_sim_multiplier: float = 0.0,
    precomputed_multiplier: float = 1.0,
    resample_downweighter: float = 0.6
):
    """
    re-implementation of v9 to remove the log forward search
    and restrict minimal n_opt
    """
    random.seed(42)
    rng = random.Random(42)
    rng_np = np.random.default_rng(42)

    seq_starts, seq_ends, seq_lengths, total_seqs, completion_length = model._prepare_prompt_and_stats(
        batch, start_tokens
    )
    max_context_tokens = (model.max_tokens - completion_length) - 5
    avg_seq_len = sum(seq_lengths) / len(seq_lengths) if len(seq_lengths) > 0 else 0
    min_seq_len = min(seq_lengths)
    assumed_seq_len = (min_seq_len + avg_seq_len) / 2
    max_n_by_tokens = max(0, min(int(max_context_tokens // assumed_seq_len) + 2, total_seqs)) if avg_seq_len > 0 else 0
    # find range of n_opt values that are in the target likelihood range:
    lower_bound = min(max_n_by_tokens, 2)
    upper_bound = min(max_n_by_tokens, total_seqs)
    vals_in_range = list(np.arange(lower_bound, upper_bound + 1, dtype=int))
    if len(vals_in_range) == 0:
        vals_in_range = [0]
    n_opt = int(rng.choice(vals_in_range))

    # compute likelihoods for each n_opt value in the range:
    spearman_list = []
    variant_lls: List[np.ndarray] = []
    n_seqs_list = []
    tok_cnt_list: List[int] = []
    min_cov_list: List[float] = []
    # Additional metrics to mirror v5
    min_length_ratio_list: List[float] = []
    min_sequence_similarity_list: List[float] = []
    mean_sequence_similarity_list: List[float] = []
    max_sequence_similarity_list: List[float] = []
    min_coverage_list: List[float] = []
    mean_coverage_list: List[float] = []
    max_coverage_list: List[float] = []
    
    token_count_attempts = 100
    if completion_length + 2 > model.max_tokens:
        n_opt = 0
        repeats = 1
    else:
        repeats = min(model.gym_subsamples_per_n, total_seqs)
    weights = np.ones(total_seqs) / total_seqs
    for rep in range(repeats):
        fail_count = 0
        while True:
            if n_opt == 0 and 0 in n_seqs_list:
                n_opt = int(random.choice(vals_in_range))
            idxs = rng_np.choice(np.arange(total_seqs), size=min(n_opt, total_seqs), replace=False, p=weights).tolist()
            # Downweight the probability of re-sampling chosen indices and renormalise
            weights[idxs] *= resample_downweighter
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights[:] = 1.0 / len(weights)
            rng.shuffle(idxs)
            tok_cnt = sum(seq_lengths[i] for i in idxs)
            if tok_cnt + completion_length <= model.max_tokens:
                fail_count = 0
                break
            else:
                fail_count += 1
                if fail_count > token_count_attempts:
                    n_opt = max(0, n_opt - 1)
                    fail_count = 0
        if n_opt == 0:
            # No context sequences selected; use empty prompt
            idxs = []
            var_batch = model._clone_batch(batch)
            var_batch["input_ids"] = None
        else:
            var_batch = model._make_truncated_batch_from_indices(
                batch, idxs, seq_starts, seq_ends, start_tokens, include_optional_meta=True
            )
        var_batch_device = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in var_batch.items()}
        L = var_batch_device["completion_ids"].shape[-1]
        L_prompt = 0 if var_batch_device["input_ids"] is None else var_batch_device["input_ids"].shape[-1]
        lls = model.score_seqs(
            var_batch_device["input_ids"],
            var_batch_device["completion_ids"],
            use_cache=getattr(model, "use_kv_cache_for_scoring", True),
            batch_size=max((getattr(model, "scoring_max_tokens", 32000)) // (L + L_prompt), 1)
            if getattr(model, "use_kv_cache_for_scoring", True)
            else 1,
        )
        
        variant_lls.append(lls)
        n_opt = rng.choice(vals_in_range)


    # Stack and persist
    lls_array = np.stack(variant_lls, axis=0)
    # Return per-sequence mean log-likelihood across variants
    mean_lls_per_sequence = lls_array.mean(axis=0)
    return mean_lls_per_sequence

def encode_prompt_without_trailing_sep(model: LlamaLitModule, proteins: ProteinDocument, document_token: str):
    pre_cfg = AlignedProteinPreprocessingConfig(
        document_token=document_token,
        defer_sampling=True,
        padding="do_not_pad",
        shuffle_proteins_in_document=False,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
        max_tokens_per_example=None,
    )
    preprocessor = ProteinDocumentPreprocessor(cfg=pre_cfg)
    prepared = preprocessor.apply_transforms(proteins, model.tokenizer)
    tokenized = model.tokenizer.encode(
        prepared,
        document_token=document_token,
        padding="do_not_pad",
        add_final_sep=False,  # important: avoid double [SEP] with completion BOS=[SEP]
        allow_unk=getattr(pre_cfg, "allow_unk", False),
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Compute conditional likelihoods of candidate sequences given conditioning sequences")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="model_checkpoints/abyoeovl",
        help="Checkpoint run directory (contains checkpoints/last.ckpt)",
    )
    parser.add_argument("--conditioning_fasta", type=str, required=True, help="Path to conditioning FASTA/MSA file")
    parser.add_argument("--candidates_fasta", type=str, required=True, help="Path to candidate sequences FASTA file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=8192, help="Token budget (prompt+completion) used for batch size heuristics")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Override attention implementation before model init (e.g. flash_attention_2)",
    )
    args = parser.parse_args()

    seed_all(args.seed)

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    override_attn_impl = args.attn_implementation or None
    if override_attn_impl is not None:
        try:
            ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            hyper_params = ckpt_blob.get("hyper_parameters", {})
            cfg_obj = hyper_params.get("config", None)
            if cfg_obj is None:
                raise RuntimeError("Could not find 'config' in checkpoint hyper_parameters to override attn implementation")
            setattr(cfg_obj, "attn_implementation", override_attn_impl)
            setattr(cfg_obj, "_attn_implementation", override_attn_impl)
            model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(ckpt_path, config=cfg_obj, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to override attention implementation: {e}")
    else:
        migrated_ckpt_path = ckpt_path.replace("last.ckpt", "last_migrated.ckpt")
        model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model.to(args.device, dtype=dtype_map[args.dtype])

    # Build ProteinDocument objects
    cond_doc = build_pool_from_fasta(args.conditioning_fasta)
    cand_names, cand_seqs = read_fasta(args.candidates_fasta, keep_insertions=False, to_upper=True)
    if len(cand_seqs) == 0:
        raise ValueError("No candidate sequences found")

    # Encode prompt without trailing [SEP]
    document_token = "[RAW]"
    prompt_tokenized = encode_prompt_without_trailing_sep(model, cond_doc, document_token)
    input_ids = torch.as_tensor(prompt_tokenized["input_ids"], dtype=torch.long).unsqueeze(0).to(model.device)

    # Encode completions with BOS/EOS = [SEP]
    comp_tok = model.tokenizer.encode_completions(cand_seqs, bos_token=model.tokenizer.sep_token, eos_token=model.tokenizer.sep_token)
    completion_ids = torch.as_tensor(comp_tok["input_ids"], dtype=torch.long).unsqueeze(0).to(model.device)  # (1, n, L)

    # Determine batch size heuristic like in validation
    L_prompt = int(input_ids.shape[-1])
    L_comp = int(completion_ids.shape[-1]) if completion_ids.ndim == 3 else int(completion_ids.shape[-1])
    scoring_max_tokens = getattr(model, "scoring_max_tokens", 32000)
    bs = max((int(scoring_max_tokens) // (L_prompt + L_comp)), 1)

    # Build a batch dict compatible with v10 helper and run ensemble-style likelihoods
    batch = {
        "input_ids": input_ids,
        "completion_ids": completion_ids,
        # Optional fields not used by this stripped-down flow
    }
    with torch.no_grad():
        lls = _evaluate_and_save_variants_v10(model, batch, start_tokens=[47, 63])

    # Print results to stdout: accession,mean_log_likelihood
    for name, ll in zip(cand_names, lls.tolist()):
        print(f"{name},{ll}")


if __name__ == "__main__":
    main()

