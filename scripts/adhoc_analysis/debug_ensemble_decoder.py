import argparse
import os
import torch
import torch.nn.functional as F

from src.models.base import load_checkpoint
from src.models.inference import (
    EnsemblePromptBuilder,
    ProFamEnsembleSampler,
)
from src.data.objects import ProteinDocument
from src.data.processors.preprocessing import PreprocessingConfig, ProteinDocumentPreprocessor, AlignedProteinPreprocessingConfig
from src.sequence.fasta import read_fasta
from src.models.llama import LlamaLitModule


def _pick_non_special_token_id(tokenizer) -> int:
    # Prefer an amino-acid-like token if available
    for tok in ["A", "L", "G", "S", "T"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid not in getattr(tokenizer, "all_special_ids", []):
            return int(tid)
    # Fallback: first non-special id
    vocab_size = getattr(tokenizer, "vocab_size", None)
    special = set(getattr(tokenizer, "all_special_ids", []))
    if vocab_size is None:
        vocab_size = max([i for i in range(4096)])  # conservative fallback
    for tid in range(int(vocab_size)):
        if tid not in special:
            return int(tid)
    # As a last resort, return 0
    return 0


@torch.no_grad()
def assert_kv_cache_equivalence(
    model: LlamaLitModule,
    sampler: ProFamEnsembleSampler,
    variants,
    force_float32_eval: bool = False,
):
    pass


def build_pool_from_fasta(path: str, is_msa: bool) -> ProteinDocument:
    if is_msa:
        names, seqs = read_fasta(path, keep_insertions=False, to_upper=True)
    else:
        names, seqs = read_fasta(path)
    # representative is first by default if present
    rep = names[0] if len(names) > 0 else None
    return ProteinDocument(
        sequences=seqs,
        accessions=names,
        identifier=os.path.basename(path),
        representative_accession=rep,
    )


def main():
    parser = argparse.ArgumentParser(description="Debug ensemble decoder sampling")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-23_22-18/2025-06-10_22-48-14-455325/", 
        help="Checkpoint run directory (contains .hydra)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="../data/ProteinGym/filtered_msas_poet/PABP_YEAST_Melamed_2013_filtered.fasta", 
        help="Path to FASTA/MSA file"
        )
    parser.add_argument("--msa", default=True, action="store_true", help="Treat input as MSA (a3m/a2m)")
    parser.add_argument("--num_variants", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_generated_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reduction", type=str, default="mean_probs", choices=["mean_probs", "sum_log_probs"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="torch.bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--test_kv_cache", action="store_true", help="Run KV-cache equivalence test and exit")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    # Load model (and tokenizer) from checkpoint dir
    model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(args.device, dtype=eval(args.dtype))

    # Build full pool from input (no subsampling here)
    pool = build_pool_from_fasta(args.input, args.msa)
    doc_token = "[RAW]"

    # Preprocessor with deferred sampling (keeps all sequences)
    cfg = AlignedProteinPreprocessingConfig(
        document_token=doc_token,
        defer_sampling=True,
        padding="do_not_pad",
        shuffle_proteins_in_document=True,
        keep_insertions=False,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    preprocessor = ProteinDocumentPreprocessor(cfg=cfg)

    # Ensemble sampler
    builder = EnsemblePromptBuilder(preprocessor=preprocessor, shuffle=True)
    sampler = ProFamEnsembleSampler(
        name="ensemble_sampler",
        model=model,
        prompt_builder=builder,
        document_token=doc_token,
        reduction=args.reduction,
        temperature=args.temperature,
    )
    sampler.to(args.device)


    sequences, variants = sampler.sample_seqs_ensemble(
        protein_document=pool,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        num_variants=args.num_variants,
        max_generated_length=args.max_generated_length,
    )

    print("Generated sequences:")
    for i, s in enumerate(sequences):
        print(f"[{i}] {s}")


if __name__ == "__main__":
    main()

