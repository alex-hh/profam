#!/usr/bin/env python3
"""
Script to analyze sequence statistics from processed MSA parquet files and generate logos
for a subset of the data.
"""

import glob
import os
import random
import subprocess
import uuid
from typing import List, Tuple

import logomaker
import numpy as np
import pandas as pd
from Bio import AlignIO
from tqdm import tqdm


def write_fasta(sequences: List[str], accessions: List[str], fasta_path: str) -> None:
    """Write sequences to FASTA format."""
    with open(fasta_path, "w") as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def create_logo_from_sequences(sequences: List[str], output_logo: str) -> None:
    counts_matrix = logomaker.alignment_to_matrix(sequences)
    logo = logomaker.Logo(
        counts_matrix, color_scheme="weblogo_protein", width=0.8, figsize=(60, 2.5)
    )
    logo.fig.savefig(output_logo)


def compute_seq_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two aligned sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != "-" and b != "-")
    total = sum(1 for a, b in zip(seq1, seq2) if a != "-" and b != "-")

    return matches / total if total > 0 else 0.0


def analyze_sequences(sequences: np.ndarray) -> dict:
    """Compute statistics for a set of sequences."""
    # Convert sequences to list if they're not already
    sequences = [str(seq) for seq in sequences]

    # Length statistics
    lengths_with_gaps = [len(seq) for seq in sequences]
    lengths_without_gaps = [len(seq.replace("-", "")) for seq in sequences]

    # Gap statistics
    gap_counts = [seq.count("-") for seq in sequences]

    # Sequence length ratio
    min_length = min(lengths_without_gaps)
    max_length = max(lengths_without_gaps)

    # Compute average sequence identity
    num_seq_pairs = min(100, (len(sequences) * (len(sequences) - 1)) // 2)
    if num_seq_pairs > 0:
        # Randomly sample sequence pairs
        seq_pairs = random.sample(
            [
                (i, j)
                for i in range(len(sequences))
                for j in range(i + 1, len(sequences))
            ],
            num_seq_pairs,
        )
        identities = [
            compute_seq_identity(sequences[i], sequences[j]) for i, j in seq_pairs
        ]
        avg_identity = np.mean(identities)
    else:
        avg_identity = 0.0

    return {
        "avg_length_with_gaps": np.mean(lengths_with_gaps),
        "avg_length_without_gaps": np.mean(lengths_without_gaps),
        "min_length_without_gaps": min(lengths_without_gaps),
        "max_length_without_gaps": max(lengths_without_gaps),
        "avg_gaps": np.mean(gap_counts),
        "length_ratio": max_length / min_length if min_length > 0 else 0,
        "avg_sequence_identity": avg_identity,
        "min_identity": min(identities),
        "max_identity": max(identities),
        "num_sequences": len(sequences),
    }


def main():
    # Parameters
    input_pattern = (
        "../data/openfold/uniclust30_filtered_parquet_fragments_debugging/*.parquet"
    )
    output_dir = "./outputs/openfold_sequence_analysis_output"
    num_files = 100
    rows_per_file = 20
    num_logos = 20

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    logo_dir = os.path.join(output_dir, "logos")
    os.makedirs(logo_dir, exist_ok=True)

    # Get list of parquet files and randomly sample
    parquet_files = glob.glob(input_pattern)
    if len(parquet_files) > num_files:
        parquet_files = random.sample(parquet_files, num_files)

    # Store all results
    all_results = []
    logo_candidates = []

    # Process each file
    for parquet_file in tqdm(parquet_files, desc="Processing files"):
        df = pd.read_parquet(parquet_file)

        # Randomly sample rows
        if len(df) > rows_per_file:
            sampled_rows = df.sample(n=rows_per_file)
        else:
            sampled_rows = df

        # Analyze each row
        for _, row in sampled_rows.iterrows():
            sequences = row["sequences"]
            stats = analyze_sequences(sequences)

            # Add file and row info
            stats["parquet_file"] = os.path.basename(parquet_file)
            if "fam_id" in row:
                stats["fam_id"] = row["fam_id"]

            all_results.append(stats)
            if len(sequences) > 20:
                logo_candidates.append((sequences, stats))

    # Save statistics to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "sequence_statistics.csv"), index=False)

    # Generate logos for random subset
    if len(logo_candidates) > num_logos:
        logo_samples = random.sample(logo_candidates, num_logos)
    else:
        logo_samples = logo_candidates

    # Create logos
    for i, (sequences, stats) in enumerate(tqdm(logo_samples, desc="Generating logos")):

        try:
            logo_file = os.path.join(logo_dir, f"logo_{i}.png")
            create_logo_from_sequences(list(sequences), logo_file)
        except Exception as e:
            print(f"Error generating logo for sequence {i}: {e}")

    print(f"Analysis complete. Results saved to {output_dir}")
    print(
        f"Sequence statistics saved to {os.path.join(output_dir, 'sequence_statistics.csv')}"
    )
    print(f"Logos saved to {logo_dir}")


if __name__ == "__main__":
    main()
