import argparse
import os
import glob
import torch
import torch.nn.functional as F
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

"""
Script for scoring sequences from a checkpoint model
input: conditioning_sequences.fasta, candidate_sequences.fasta
optional_input: ground_truth_scores.csv # if provided, will compute Spearman rank correlation between ground truth scores and model scores
outputs are saved to output_path as a csv file
"""

from src.models.base import load_checkpoint
from src.models.inference import (
    EnsemblePromptBuilder,
    PromptBuilder,
)
from src.data.objects import ProteinDocument
from src.data.processors.preprocessing import PreprocessingConfig, ProteinDocumentPreprocessor, AlignedProteinPreprocessingConfig
from src.sequence.fasta import read_fasta
from src.models.llama import LlamaLitModule
from src.utils.utils import seed_all



def write_fasta(sequences, accessions, fasta_path):
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def build_pool_from_fasta(path: str) -> ProteinDocument:
    names, seqs = read_fasta(path, keep_insertions=False, to_upper=True)
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
        default="model_checkpoints/abyoeovl", 
        help="Checkpoint run directory (contains .hydra)"
    )
    parser.add_argument(
        "--glob",
        type=str,
        required=True,
        help="Glob pattern for input FASTA/MSA files (e.g. '../data/val/*.fasta')"
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated FASTA files")
    parser.add_argument("--sampler", type=str, default="single", choices=["ensemble", "single"], help="Sampler type: ensemble or single")
    parser.add_argument("--num_prompts_in_ensemble", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability mass (0<p<=1)")
    parser.add_argument("--reduction", type=str, default="mean_probs", choices=["mean_probs", "sum_log_probs"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--task_index", type=int, default=None, help="Task index")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Override attention implementation before model init (e.g. flash_attention_2)",
    )
    args = parser.parse_args()

    # Seed RNGs for reproducibility
    seed_all(args.seed)

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    # Load model (and tokenizer) from checkpoint dir, optionally overriding attention implementation
    override_attn_impl = args.attn_implementation or None
    if override_attn_impl is not None:
        try:
            ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            hyper_params = ckpt_blob.get("hyper_parameters", {})
            cfg_obj = hyper_params.get("config", None)
            if cfg_obj is None:
                raise RuntimeError("Could not find 'config' in checkpoint hyper_parameters to override attn implementation")
            # Set both public and internal fields to be safe across HF versions
            setattr(cfg_obj, "attn_implementation", override_attn_impl)
            setattr(cfg_obj, "_attn_implementation", override_attn_impl)
            model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(ckpt_path, config=cfg_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to override attention implementation: {e}")
    else:
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
    if args.task_index is not None and args.num_tasks is not None:
        batch_size = len(input_files) // args.num_tasks
        start_idx = args.task_index * batch_size
        if args.task_index == args.num_tasks - 1:
            end_idx = len(input_files)
        else:
            end_idx = start_idx + batch_size
        input_files = input_files[start_idx:end_idx]
        print(f"Processing {len(input_files)} files in task {args.task_index} of {args.num_tasks}")
        for fpath in input_files:
            print(fpath)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No input files matched pattern: {args.glob}")

    doc_token = "[RAW]"

    # Preprocessor with deferred sampling (keeps all sequences)
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

    # Build sampler according to selection
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


if __name__ == "__main__":
    main()

