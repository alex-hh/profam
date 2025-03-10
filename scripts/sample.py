#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.constants import BASEDIR
from src.data.objects import ProteinDocument
from src.models.utils import load_named_model


def parse_args():
    parser = argparse.ArgumentParser(description="Sample sequences from a trained model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="logs/train_single_seq/runs/2025-03-06_21-30-48-576411/checkpoints/last.ckpt",
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
        default=0.8,
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
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = load_named_model(args.model_config)
    
    # Load checkpoint
    model.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    # Create a minimal prompt with just the document token
    prompt = ProteinDocument(sequences=["[RAW]"])
    
    # Set up sampling parameters
    sampling_kwargs = {
        "temperature": args.temperature,
        "batch_size": args.batch_size,
    }
    
    # Generate sequences
    with torch.no_grad():
        sequences, _ = model.sample_seqs(
            prompt,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            **sampling_kwargs
        )
    
    # Save generated sequences
    output_file = os.path.join(args.output_dir, "generated_sequences.txt")
    with open(output_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">sample_{i}\n{seq}\n")
    
    print(f"Generated {len(sequences)} sequences and saved to {output_file}")


if __name__ == "__main__":
    main() 