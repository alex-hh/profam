#!/usr/bin/env python3

"""
Created by Jude Wells 2025-01-16

This script does mmseqs clustering within each family at different levels of sequence identity,
replacing the use of foldseek with mmseqs.

Instead of using temporary directories for intermediate files, it will create and store them
directly in the output directory, then clean them up. Each row in the parquet must at least have two columns:
    - 'sequences': an np.array of sequences (strings)
    - 'accessions': an np.array of accessions (strings)

Following clustering, it assigns each sequence to a cluster for the desired identity thresholds, then writes
those cluster assignments to new columns in the same parquet file.
"""

import os
import sys
import argparse
import subprocess
import uuid
import glob
import numpy as np
import pandas as pd
import shutil
import time

ERROR_LOGS = []
TIMINGS = []


def run_mmseqs_easy_cluster(fasta_file: str,
                            out_prefix: str,
                            min_seq_id: float,
                            threads: int):
    """
    Runs mmseqs easy-cluster on the given FASTA file, storing results into
    out_prefix* files in the same directory as out_prefix.
    Returns the path to the resulting 'cluster.tsv' file for further parsing.

    We use --min-seq-id to set the cluster identity threshold,
    --cluster-mode 1 to produce an adjacency list suitable for connected-component style parsing,
    and --remove-tmp-files 1 to remove intermediate files.
    """
    cmd = [
        "mmseqs", "easy-cluster",
        fasta_file,
        out_prefix,
        out_prefix,
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads),
        "--remove-tmp-files", "1",
        "--cluster-mode", "1"
    ]
    print(f"Running mmseqs: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)

    # The cluster adjacency output is in out_prefix + "_cluster.tsv"
    cluster_tsv = f"{out_prefix}_cluster.tsv"
    return cluster_tsv


def parse_mmseqs_cluster_results(cluster_tsv: str, sequence_ids: list) -> np.ndarray:
    """
    mmseqs outputs a cluster file (in adjacency-list format), typically lines of the form:
        seqID1 seqID2
    Indicating seqID1 is in the same cluster as seqID2 (connectivity).
    We load these lines and figure out which cluster each sequence belongs to.
    If any sequence is missing from the cluster file, treat it as a singleton.

    Returns: an np.array of cluster IDs (same length as sequence_ids).
    """
    if not os.path.isfile(cluster_tsv):
        # Return array of empty strings matching input length
        return np.array([''] * len(sequence_ids), dtype=str)
    
    # Handle missing sequences by joining with original IDs
    all_ids = pd.DataFrame({'accession': sequence_ids})
    cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=['cluster_rep', 'member'])
    merged = pd.merge(all_ids, cluster_df, how='left', left_on='accession', right_on='member')
    
    # Fill NA values with their own accession (singleton clusters)
    merged['cluster_id'] = merged['cluster_rep'].fillna(merged['accession'])
    return merged['cluster_id'].values.astype(str)


def cluster_family_sequences(sequences: np.ndarray,
                            accessions: np.ndarray,
                            min_seq_id: float,
                            threads: int,
                            output_dir: str,
                            ) -> np.ndarray:
    """
    Writes sequences to a FASTA file in output_dir, runs mmseqs easy-cluster
    with the given --min-seq-id, parses the results to produce an array
    of cluster IDs. The index order is the same as the input sequences.
    Deletes intermediate files afterwards.
    """
    if len(sequences) == 0:
        # Edge case: no sequences
        return np.array([], dtype=str)

    # Generate a unique directory for this clustering job
    unique_dir = os.path.join(output_dir, uuid.uuid4().hex)
    os.makedirs(unique_dir, exist_ok=True)
    out_prefix = os.path.join(unique_dir, "cluster")
    fasta_file = os.path.join(unique_dir, "input.fasta")

    try:
        # Write sequences to FASTA
        with open(fasta_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">{accessions[i]}\n{seq}\n")

        # Run mmseqs and parse results
        try:
            cluster_tsv = run_mmseqs_easy_cluster(
                fasta_file, out_prefix, min_seq_id, threads
            )
        except Exception as e:
            print(f"Error running mmseqs: {e}")
            ERROR_LOGS.append(f"Error running mmseqs: {e}")
            return np.array([], dtype=str)

        cluster_array = parse_mmseqs_cluster_results(cluster_tsv, accessions)

    finally:
        # Clean up the entire directory
        shutil.rmtree(unique_dir)

    return cluster_array


def process_parquet_file(parquet_path: str,
                         output_dir: str,
                         identity_thresholds: list,
                         threads: int):
    """
    Loads a parquet file, iterates over each row (family),
    runs clustering at each identity threshold, and adds new columns
    with cluster assignments (np.array of ints). Saves to a new parquet file
    in output_dir.
    """
    start_time = time.time()
    filename = os.path.basename(parquet_path)
    new_parquet_path = os.path.join(output_dir, filename)
    if os.path.exists(new_parquet_path):
        df = pd.read_parquet(new_parquet_path)
        has_missing = False
        for thr in identity_thresholds:
            col_name = f"cluster_ids_{str(thr).replace('.', '_')}"
            if col_name in df.columns and df[col_name].isnull().any():
                has_missing = True
                break
        if not has_missing:
            print(f"Skipping {parquet_path} as it is already processed ")
            return

    print(f"Processing {parquet_path} => {new_parquet_path}", file=sys.stderr)
    df = pd.read_parquet(parquet_path)
    drop_row_indices = []
    # For each row, run clustering for each threshold
    # We'll store the cluster arrays in new columns named "cluster_ids_{thr}".
    mmseqs_outputs_dir = os.path.join(output_dir, "mmseqs_outputs")
    for idx in range(len(df)):
        os.makedirs(mmseqs_outputs_dir, exist_ok=True)
        sequences = df.loc[idx, 'sequences']
        accessions = df.loc[idx, 'accessions'] if 'accessions' in df.columns else None
        if accessions is None or sequences is None or len(sequences) == 0 or len(accessions) == 0:
            drop_row_indices.append(idx)
            continue
        if not isinstance(sequences, np.ndarray):
            # Convert to np.array if needed (some parquet data can come back as lists)
            sequences = np.array(sequences, dtype=str)
        if accessions is not None and not isinstance(accessions, np.ndarray):
            accessions = np.array(accessions, dtype=str)

        # If there are no sequences or only empty strings, skip to avoid mmseqs error.
        if (len(sequences) == 0) or all(len(str(s)) == 0 for s in sequences):
            for thr in identity_thresholds:
                col_name = f"cluster_ids_{str(thr).replace('.', '_')}"
                df.at[idx, col_name] = np.array([], dtype=str)
            continue

        for thr in identity_thresholds:
            cluster_ids = cluster_family_sequences(
                sequences=sequences,
                accessions=accessions,
                min_seq_id=thr,
                threads=threads,
                output_dir=mmseqs_outputs_dir
            )
            # Ensure we always have a string dtype array
            cluster_ids = cluster_ids.astype(str)
            col_name = f"cluster_ids_{str(thr).replace('.', '_')}"
            df.at[idx, col_name] = cluster_ids

    # Convert back to Arrow Table and save
    if len(drop_row_indices) > 0:
        print(f"Dropping {len(drop_row_indices)} rows due to empty sequences from {parquet_path}")
        df = df.drop(drop_row_indices)
    df.to_parquet(new_parquet_path, index=False)
    print(f"Finished processing {parquet_path}")
    end_time = time.time()
    TIMINGS.append({
        "n_families": len(df),
        "time_taken": end_time - start_time
    })

def main():
    parser = argparse.ArgumentParser(
        description="mmseqs clustering within each family at different levels of sequence identity."
    )
    parser.add_argument("--input_pattern", required=True,
                        help="Pattern to match parquet files.")
    parser.add_argument("--task_index", type=int, required=True,
                        help="Index of the task to run.")
    parser.add_argument("--num_tasks", type=int, required=True,
                        help="Number of tasks to run.")
    parser.add_argument("--identity_thresholds", nargs="+", default=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                        type=float,
                        help="Sequence identity thresholds for clustering.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of CPU threads for mmseqs.")
    args = parser.parse_args()


    # Gather all parquet files in input_dir
    parquet_files = glob.glob(args.input_pattern)
    print(f"Found {len(parquet_files)} parquet files")
    batch_size = (len(parquet_files) // args.num_tasks) +1
    start_idx = args.task_index * batch_size
    end_idx = start_idx + batch_size
    parquet_files = parquet_files[start_idx:end_idx]
    print(f"Processing {len(parquet_files)} parquet files in batch {args.task_index} of {args.num_tasks}")

    # Process each parquet file
    for parquet_path in parquet_files:
        output_dir = os.path.join(os.path.dirname(parquet_path), "clustered")

        # Create output dir if doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        process_parquet_file(parquet_path,
                             output_dir,
                             args.identity_thresholds,
                             args.threads)

        if len(TIMINGS) > 0:
            timings_df = pd.DataFrame(TIMINGS)
            timings_df.to_csv(os.path.join(output_dir, "timings.csv"), index=False)
            total_fams_processed = timings_df['n_families'].sum()
            total_time_taken = timings_df['time_taken'].sum()
            mean_time_per_fam = total_time_taken / total_fams_processed
            print(f"Processed {total_fams_processed} families in {total_time_taken:.2f} seconds")
            print(f"Average time per family: {mean_time_per_fam:.2f} seconds")
            target_families = 5500
            print(f"Estimated time to process {target_families} families: {mean_time_per_fam * target_families:.2f} seconds")
            print(f"Error logs: {len(ERROR_LOGS)}")
            # write the error logs to a file
            with open(os.path.join(output_dir, f"error_logs_{args.task_index}.txt"), "w") as f:
                for error in ERROR_LOGS:
                    f.write(error + "\n")

if __name__ == "__main__":
    main()