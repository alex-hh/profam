#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Dict, Optional

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.constants import BASEDIR
from src.data.objects import ProteinDocument
from src.data.processors.preprocessing import (
    PreprocessingConfig,
    ProteinDocumentPreprocessor,
)
from src.models.inference import ProFamSampler, PromptBuilder
from src.models.utils import load_named_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample sequences from a trained model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-23_22-18/2025-06-10_22-48-14-455325/checkpoints/last.ckpt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="llama_32_1b",
        help="Name of the model config to use (e.g. llama_32_1b)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature for sampling (higher = more random)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/samples",
        help="Directory to save generated sequences",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type to use for inference (float32, float16, or bfloat16)",
    )
    return parser.parse_args()


def get_config_from_checkpoint(checkpoint_path):
    run_dir = checkpoint_path.split("/checkpoints")[0]
    config_path = glob.glob(f"{run_dir}/.hydra/config.yaml")
    if len(config_path) == 0:
        raise ValueError(f"No config file found in {run_dir}")
    config = OmegaConf.load(config_path[0])

    # Ensure the config has the necessary structure
    if "model" not in config:
        raise ValueError(
            f"Config in {config_path[0]} does not contain a 'model' section"
        )

    return config


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    config = get_config_from_checkpoint(args.checkpoint_path)
    tokenizer = instantiate(config.tokenizer)

    # Set the data type based on the command-line argument
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load the model directly from checkpoint
    model_class = hydra.utils.get_class(config.model._target_)
    model = model_class.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")
    model = model.to(dtype)
    # Create a simple preprocessing config for unconditional sampling
    preprocessing_config = PreprocessingConfig(
        document_token="[RAW]",
        drop_first_protein=False,
        keep_first_protein=False,
        allow_unk=False,
        max_tokens_per_example=None,
        shuffle_proteins_in_document=False,
        padding="do_not_pad",
    )

    # Create a simple prompt builder for unconditional sampling
    preprocessor = ProteinDocumentPreprocessor(cfg=preprocessing_config)
    prompt_builder = PromptBuilder(preprocessor=preprocessor)

    # Set up sampling parameters
    sampling_kwargs = {
        "temperature": args.temperature,
        "batch_size": args.batch_size,
    }

    # Create a ProFamSampler instance
    sampler = ProFamSampler(
        name="unconditional_sampler",
        model=model,
        prompt_builder=prompt_builder,
        document_token="[RAW]",
        sampling_kwargs=sampling_kwargs,
        dtype=dtype,
    )

    # Create a minimal prompt with just the document token for unconditional sampling
    prompt = ProteinDocument(sequences=["M"])

    # Generate sequences
    sequences, _ = sampler.sample_seqs(
        protein_document=prompt,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
    )

    # Save generated sequences
    output_file = os.path.join(args.output_dir, "generated_sequences.txt")
    with open(output_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">sample_{i}\n{seq}\n")

    print(f"Generated {len(sequences)} sequences and saved to {output_file}")


if __name__ == "__main__":
    main()
