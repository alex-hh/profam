#!/usr/bin/env python3
"""
Created by Jude Wells, 2025-01-20

This script iterates over clustered parquet files (one family per row),
performs a multiple alignment of the entire family, generates a logo of that alignment,
and then for each identity threshold:
  1) Finds the largest cluster within that family.
  2) Logs its size in a CSV.
  3) If the cluster has more than 10 members, generate a multiple alignment & logo for that cluster as well.

It references:
  - sequence clustering at various identity thresholds (cluster_within_familes.py),
  - how sequence logos are created (logo_maker.py).

Usage Example:
  python generate_cluster_alignments_and_logos.py \
      --input_dir ../data/ted/s100_parquets_clustered \
      --output_dir ./analysis_outputs \
      --threads 4
"""

import os
import sys
import glob
import shutil
import uuid
import argparse
import numpy as np
import pandas as pd
import subprocess

from Bio import AlignIO
import logomaker


def write_fasta(sequences, accessions, fasta_path):
    """
    Write an array of sequences + accessions to a FASTA file.
    """
    with open(fasta_path, 'w') as f:
        for acc, seq in zip(accessions, sequences):
            f.write(f">{acc}\n{seq}\n")


def run_alignment_with_mafft(fasta_input, fasta_output, threads=1):
    """
    Demonstrates how you might run an alignment with MAFFT.

    Example usage:
      mafft --thread N --auto input.fasta > output.fasta
    """
    cmd = [
        "mafft",
        "--thread", str(threads),
        "--auto",
        fasta_input
    ]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    with open(fasta_output, 'w') as fout:
        subprocess.run(cmd, check=True, stdout=fout)


def create_logo_from_fasta(alignment_fasta, output_logo):
    """
    Read an alignment from alignment_fasta, convert to a list of sequences,
    feed into logomaker, and save the resulting PNG.
    """
    alignment = AlignIO.read(alignment_fasta, 'fasta')
    sequences = [str(record.seq) for record in alignment]

    # Build logomaker counts matrix
    counts_matrix = logomaker.alignment_to_matrix(sequences)
    logo = logomaker.Logo(counts_matrix,
                          color_scheme="weblogo_protein",
                          width=0.8,
                          figsize=(60, 2.5))
    logo.fig.savefig(output_logo)
    print(f"Sequence logo saved as {output_logo}")


def main():
    parser = argparse.ArgumentParser(
        description="""Iterate over clustered parquet files, align full families
                       and largest clusters at each identity threshold, generate logos."""
    )
    parser.add_argument("--input_dir", default="../data/ted/s100_parquets_clustered",
                        help="Path to a directory containing clustered parquet files.")
    parser.add_argument("--output_dir", default="../ted_cluster_msas",
                        help="Where to write alignments, logos, and CSV of largest cluster sizes.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for alignment tool.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    alignment_dir = os.path.join(args.output_dir, "alignments")
    os.makedirs(alignment_dir, exist_ok=True)
    logo_dir = os.path.join(args.output_dir, "logos")
    os.makedirs(logo_dir, exist_ok=True)

    # Gather any parquet files in input_dir
    parquet_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))

    # We will keep track of largest clusters in a list of dicts, then write to CSV.
    largest_cluster_log = []

    for parquet_path in parquet_files:
        print(f"Processing parquet: {parquet_path}", file=sys.stderr)
        df = pd.read_parquet(parquet_path)

        # Attempt to automatically infer identity threshold columns from patterns like "cluster_ids_..."
        cluster_cols = [c for c in df.columns if c.startswith("cluster_ids_")]

        # For each row (family) in the parquet file
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            sequences = row['sequences']
            if not isinstance(sequences, np.ndarray):
                sequences = np.array(sequences, dtype=str)

            if 'accessions' in row and row['accessions'] is not None:
                accessions = row['accessions']
                if not isinstance(accessions, np.ndarray):
                    accessions = np.array(accessions, dtype=str)
            else:
                # Fall back on enumerated accessions if not present
                accessions = np.array([f"seq_{i}" for i in range(len(sequences))], dtype=str)

            if len(sequences) == 0:
                # Skip empty families
                continue

            # Attempt to capture a fam_id if it exists
            fam_id = row.get("fam_id", None)
            # Generate a unique family identifier for output files
            unique_family_id = str(fam_id) if fam_id else f"{os.path.splitext(os.path.basename(parquet_path))[0]}_row{row_idx}"

            # 1) Create alignment for the entire family
            family_fasta = os.path.join(alignment_dir, f"{unique_family_id}.full.fasta")
            family_aln = os.path.join(alignment_dir, f"{unique_family_id}.full_aln.fasta")
            family_logo = os.path.join(logo_dir, f"{unique_family_id}.full_logo.png")

            write_fasta(sequences, accessions, family_fasta)
            try:
                run_alignment_with_mafft(family_fasta, family_aln, threads=args.threads)
                create_logo_from_fasta(family_aln, family_logo)
            except Exception as e:
                print(f"ERROR: Could not align family {unique_family_id}: {e}", file=sys.stderr)

            # 2) For each identity threshold, find the largest cluster:
            for col in cluster_cols:
                cluster_ids = row[col]  # np.array of labels or None
                if cluster_ids is None or not isinstance(cluster_ids, np.ndarray) or len(cluster_ids) == 0:
                    continue

                # Find the label of the most frequent cluster
                vals, counts = np.unique(cluster_ids, return_counts=True)
                largest_cluster_label = vals[np.argmax(counts)]
                largest_cluster_size = counts.max()

                # Record result for CSV logging
                entry = {
                    "parquet_file": os.path.basename(parquet_path),
                    "row_index": row_idx,
                    "cluster_column": col,
                    "largest_cluster_label": largest_cluster_label,
                    "largest_cluster_size": largest_cluster_size
                }
                if fam_id:
                    entry["fam_id"] = fam_id
                largest_cluster_log.append(entry)

                # If largest cluster has more than 10 members, align & generate logo
                if largest_cluster_size > 10:
                    subsel = (cluster_ids == largest_cluster_label)
                    sub_sequences = sequences[subsel]
                    sub_accessions = accessions[subsel]

                    cluster_id_str = str(largest_cluster_label)
                    cluster_out_prefix = f"{unique_family_id}.{col}.cluster_{cluster_id_str}"
                    cluster_fasta = os.path.join(alignment_dir, f"{cluster_out_prefix}.fasta")
                    cluster_aln = os.path.join(alignment_dir, f"{cluster_out_prefix}_aln.fasta")
                    cluster_logo_path = os.path.join(logo_dir, f"{cluster_out_prefix}_logo.png")

                    write_fasta(sub_sequences, sub_accessions, cluster_fasta)
                    try:
                        run_alignment_with_mafft(cluster_fasta, cluster_aln, threads=args.threads)
                        create_logo_from_fasta(cluster_aln, cluster_logo_path)
                    except Exception as e:
                        print(f"ERROR: Could not align largest cluster for {cluster_out_prefix}: {e}",
                              file=sys.stderr)

    # Write out the largest cluster log to CSV
    csv_path = os.path.join(args.output_dir, "largest_clusters.csv")
    pd.DataFrame(largest_cluster_log).to_csv(csv_path, index=False)
    print(f"Finished. Wrote largest cluster log to {csv_path}")


if __name__ == "__main__":
    main()