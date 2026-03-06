import argparse
import glob
import os
from pathlib import Path
from typing import Sequence

import torch

from profam.constants import resolve_runtime_path
from profam.data.objects import ProteinDocument
from profam.data.processors.preprocessing import (
    AlignedProteinPreprocessingConfig,
    ProteinDocumentPreprocessor,
)
from profam.models.inference import (
    EnsemblePromptBuilder,
    ProFamEnsembleSampler,
    ProFamSampler,
    PromptBuilder,
)
from profam.models.llama import LlamaLitModule
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
    parser = argparse.ArgumentParser(description="Generate sequences from family prompts")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="model_checkpoints/profam-1",
        help="Checkpoint run directory (contains checkpoints/last.ckpt)",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/generate_sequences_example/4_1_1_39_cluster.filtered.fasta",
        help="Filepath or glob for input FASTA/MSA files",
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
        "--continuous_sampling",
        action="store_true",
        default=False,
        help="Ignore [SEP] EOS and generate until token budget; drop final partial segment",
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
    parser.add_argument("--task_index", type=int, default=None, help="Task index")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repeat_guard = not args.disable_repeat_guard
    if args.max_tokens > 8192:
        raise ValueError(
            "Max tokens must be less than or equal to 8192: model was only trained up to 8192 tokens."
        )

    seed_all(args.seed)

    checkpoint_dir = resolve_runtime_path(args.checkpoint_dir)
    ckpt_path = checkpoint_dir / "checkpoints" / "last.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run `profam-download-checkpoint` to download the checkpoint."
        )

    attn_impl = args.attn_implementation
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if attn_impl == "flash_attention_2":
            raise ImportError(
                "Flash attention is not installed. Select an alternative attention implementation "
                "such as `--attn_implementation sdpa`, or install it with "
                "`pip install flash-attn --no-build-isolation`."
            )

    try:
        ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hyper_params = ckpt_blob.get("hyper_parameters", {})
        cfg_obj = hyper_params.get("config", None)
        if cfg_obj is None:
            raise RuntimeError(
                "Could not find 'config' in checkpoint hyper_parameters to override attention implementation"
            )
        setattr(cfg_obj, "attn_implementation", attn_impl)
        setattr(cfg_obj, "_attn_implementation", attn_impl)
        model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(
            str(ckpt_path), config=cfg_obj, strict=False
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

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    file_pattern = os.path.expanduser(args.file_path)
    input_files = sorted(glob.glob(file_pattern))
    if len(input_files) == 0 and not glob.has_magic(file_pattern):
        resolved = resolve_runtime_path(file_pattern)
        if resolved.exists():
            input_files = [str(resolved)]

    if args.task_index is not None and args.num_tasks is not None:
        batch_size = len(input_files) // args.num_tasks
        start_idx = args.task_index * batch_size
        end_idx = len(input_files) if args.task_index == args.num_tasks - 1 else start_idx + batch_size
        input_files = input_files[start_idx:end_idx]
        print(f"Processing {len(input_files)} files in task {args.task_index} of {args.num_tasks}")
        for fpath in input_files:
            print(fpath)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No input files matched pattern: {args.file_path}")

    doc_token = "[RAW]"
    cfg = AlignedProteinPreprocessingConfig(
        document_token=doc_token,
        defer_sampling=True if args.sampler == "ensemble" else False,
        padding="do_not_pad",
        shuffle_proteins_in_document=True,
        keep_insertions=True,
        to_upper=True,
        keep_gaps=False,
        use_msa_pos=False,
        max_tokens_per_example=None if args.sampler == "ensemble" else args.max_tokens,
    )
    preprocessor = ProteinDocumentPreprocessor(cfg=cfg)

    if args.sampler == "ensemble":
        builder = EnsemblePromptBuilder(preprocessor=preprocessor, shuffle=True, seed=args.seed)
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
        builder = PromptBuilder(preprocessor=preprocessor, prompt_is_aligned=True, seed=args.seed)
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

    for fasta_path in input_files:
        base = os.path.splitext(os.path.basename(fasta_path))[0]
        out_path = save_dir / f"{base}_generated_{args.sampler}.fasta"
        if out_path.exists():
            print(f"Skipping {fasta_path} because {out_path} already exists")
            continue
        try:
            pool = build_pool_from_fasta(fasta_path)
            longest_prompt_len = int(max(pool.sequence_lengths))
            default_cap = int(longest_prompt_len * float(args.max_sequence_length_multiplier))
            if args.max_generated_length is None:
                max_gen_len = default_cap
            else:
                max_gen_len = min(int(args.max_generated_length), default_cap)
            if args.continuous_sampling:
                max_gen_len = None
            if args.sampler == "ensemble":
                sequences, scores, _ = sampler.sample_seqs_ensemble(
                    protein_document=pool,
                    num_samples=args.num_samples,
                    max_tokens=args.max_tokens,
                    num_prompts_in_ensemble=min(args.num_prompts_in_ensemble, len(pool.sequences)),
                    max_generated_length=max_gen_len,
                    continuous_sampling=args.continuous_sampling,
                    minimum_sequence_length_proportion=args.minimum_sequence_length_proportion,
                    minimum_sequence_identity=args.minimum_sequence_identity,
                    maximum_retries=args.maximum_retries,
                    repeat_guard=repeat_guard,
                )
            else:
                preprocessor.cfg.max_tokens_per_example = args.max_tokens - max_gen_len
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
                sequences, scores, _ = sampler.sample_seqs(
                    protein_document=pool,
                    num_samples=args.num_samples,
                    max_tokens=args.max_tokens,
                    max_generated_length=max_gen_len,
                    continuous_sampling=args.continuous_sampling,
                    minimum_sequence_length_proportion=args.minimum_sequence_length_proportion,
                    minimum_sequence_identity=args.minimum_sequence_identity,
                    maximum_retries=args.maximum_retries,
                    repeat_guard=repeat_guard,
                )

            accessions = [
                f"{base}_sample_{i}_log_likelihood_{score:.3f}"
                for i, score in enumerate(scores)
            ]
            write_fasta(sequences, accessions, str(out_path))
            print(f"Wrote {len(sequences)} sequences -> {out_path}")
        except Exception as e:
            print(f"Error processing {fasta_path}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
