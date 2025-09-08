import argparse
import os
import glob
import torch
import torch.nn.functional as F

from src.models.base import load_checkpoint
from src.models.inference import (
    EnsemblePromptBuilder,
    ProFamEnsembleSampler,
    PromptBuilder,
    ProFamSampler
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


def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


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
        "--glob",
        type=str,
        required=True,
        help="Glob pattern for input FASTA/MSA files (e.g. '../data/val/*.fasta')"
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated FASTA files")
    parser.add_argument("--msa", default=True, action="store_true", help="Treat inputs as aligned MSAs (a3m/a2m)")
    parser.add_argument("--sampler", type=str, default="ensemble", choices=["ensemble", "single"], help="Sampler type: ensemble or single")
    parser.add_argument("--num_variants", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_generated_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability mass (0<p<=1)")
    parser.add_argument("--reduction", type=str, default="mean_probs", choices=["mean_probs", "sum_log_probs"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    # Load model (and tokenizer) from checkpoint dir
    model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model.to(args.device, dtype=dtype_map[args.dtype])

    # Prepare save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Collect input files
    input_files = sorted(glob.glob(args.glob))
    if len(input_files) == 0:
        raise FileNotFoundError(f"No input files matched pattern: {args.glob}")

    doc_token = "[RAW]"

    # Preprocessor with deferred sampling (keeps all sequences)
    cfg = AlignedProteinPreprocessingConfig(
        document_token=doc_token,
        defer_sampling=True,
        padding="do_not_pad",
        shuffle_proteins_in_document=True,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
    )
    preprocessor = ProteinDocumentPreprocessor(cfg=cfg)

    # Build sampler according to selection
    if args.sampler == "ensemble":
        builder = EnsemblePromptBuilder(preprocessor=preprocessor, shuffle=True)
        sampler = ProFamEnsembleSampler(
            name="ensemble_sampler",
            model=model,
            prompt_builder=builder,
            document_token=doc_token,
            reduction=args.reduction,
            temperature=args.temperature,
            top_p=args.top_p,
            add_final_sep=True,
        )
    else:
        builder = PromptBuilder(preprocessor=preprocessor, prompt_is_aligned=True)
        sampling_kwargs = {}
        if args.top_p is not None:
            sampling_kwargs["top_p"] = args.top_p
        if args.temperature is not None:
            sampling_kwargs["temperature"] = args.temperature
        sampler = ProFamSampler(
            name="single_sampler",
            model=model,
            prompt_builder=builder,
            document_token=doc_token,
            sampling_kwargs=sampling_kwargs if len(sampling_kwargs) > 0 else None,
            add_final_sep=True,
        )
    sampler.to(args.device)

    # Process each file
    for fasta_path in input_files:
        base = os.path.splitext(os.path.basename(fasta_path))[0]
        out_path = os.path.join(args.save_dir, f"{base}_generated.fasta")
        if os.path.exists(out_path):
            print(f"Skipping {fasta_path} because {out_path} already exists")
            continue
        try:
            pool = build_pool_from_fasta(fasta_path, args.msa)
            if args.max_generated_length is None:
                max_gen_len = int(max(pool.sequence_lengths) * 1.5)
            else:
                max_gen_len = min(args.max_generated_length, int(max(pool.sequence_lengths) * 1.5))
            if args.sampler == "ensemble":
                sequences, _ = sampler.sample_seqs_ensemble(
                    protein_document=pool,
                    num_samples=args.num_samples,
                    max_tokens=args.max_tokens,
                    num_variants=min(args.num_variants, len(pool.sequences)),
                    max_generated_length=max_gen_len,
                )
            else:
                sequences, _ = sampler.sample_seqs(
                    protein_document=pool,
                    num_samples=args.num_samples,
                    max_tokens=args.max_tokens,
                    max_generated_length=max_gen_len,
                )


            accessions = [f"{base}_sample_{i}" for i in range(len(sequences))]
            write_fasta(sequences, accessions, out_path)
            print(f"Wrote {len(sequences)} sequences -> {out_path}")
        except Exception as e:
            print(f"Error processing {fasta_path}: {e}")


if __name__ == "__main__":
    main()

