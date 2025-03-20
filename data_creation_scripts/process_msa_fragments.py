#!/usr/bin/env python3

"""
Created to process MSA alignments from openfold parquet files.

This script:
1. Reads MSA data in a3m format from parquet files
2. Splits sequences at regions with >10 consecutive gaps
3. Processes subsequences (removes gaps, converts lowercase to uppercase, filters by length)
4. Clusters subsequences using MMSEQS at 30% identity
5. Further clusters within 30% clusters at higher identity thresholds
6. Formats results into new parquet files with specified structure
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
import re

ERROR_LOGS = []
TIMINGS = []


def parse_fasta(fasta_string):
    """Parse a FASTA string into a list of (header, sequence) tuples."""
    sequences = []
    lines = fasta_string.strip().split('\n')
    current_seq = []
    current_id = ""

    for line in lines:
        if line.startswith('>'):
            if current_id:
                sequences.append((current_id, ''.join(current_seq)))
                current_seq = []
            current_id = line[1:]
        else:
            current_seq.append(line)

    if current_id:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def extract_uniprot_id(header):
    """Extract UniProt ID from FASTA header."""
    match = re.search(r'\|([A-Z0-9]+)\|', header)
    if match:
        return match.group(1)
    return header


def split_sequence(sequence, max_allowed_gaps=10, min_sub_seq_len=20):
    """
    Split a sequence at long gap regions and process each subsequence.
    
    Args:
        sequence: Aligned sequence string
        max_allowed_gaps: Maximum number of consecutive gaps allowed before splitting
        min_sub_seq_len: Minimum length of subsequence to keep
        
    Returns:
        List of processed subsequences
    """
    split_string = "-" * max_allowed_gaps
    subsequences = sequence.split(split_string)
    subsequences = [s.replace("-", "") for s in subsequences]
    subsequences = [s.upper() for s in subsequences]
    subsequences = [s for s in subsequences if len(s) >= min_sub_seq_len]
    return subsequences


def process_subsequence(subsequence):
    """
    Process a subsequence by:
    1. Removing gaps (-)
    2. Converting lowercase (insertions) to uppercase
    
    Returns:
        Processed subsequence
    """
    # Remove gaps
    no_gaps = subsequence.replace('-', '')
    
    # Convert to uppercase (insertions in a3m are lowercase)
    return no_gaps.upper()


def run_mmseqs_cluster(fasta_file, out_prefix, min_seq_id, threads, generate_msa=False):
    """
    Runs mmseqs cluster and then generates MSAs for each cluster.
    
    The steps are:
      1. Create a database from the FASTA file.
      2. Run mmseqs cluster to cluster the sequences.
      3. Optionally, create a TSV file with cluster info.
      4. Generate MSAs from the clusters using mmseqs result2msa.
    """
    # Define database and temporary file names
    db = f"{out_prefix}_DB"
    db_clu = f"{out_prefix}_DB_clu"
    tmp = f"{out_prefix}_tmp"
    msa_out = f"{out_prefix}_DB_clu_msa"
    cluster_tsv = f"{out_prefix}_cluster.tsv"
    
    # Step 1: Create database from FASTA
    cmd_create_db = ["mmseqs", "createdb", fasta_file, db]
    print(f"Running mmseqs: {' '.join(cmd_create_db)}", file=sys.stderr)
    try:
        subprocess.run(cmd_create_db, check=True)
    except Exception as e:
        print(f"Error creating database: {e}", file=sys.stderr)
        ERROR_LOGS.append(f"Error creating database: {e}")
        return None
    
    # Step 2: Run clustering on the database
    cmd_cluster = [
        "mmseqs", "cluster", db, db_clu, tmp,
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads),
        "--cov-mode", "0",  # both sequences must have at least c coverage
        "-c", "0.7",        # coverage threshold
        "--cluster-mode", "1",
        "-v", "3",          # verbosity: 0=quiet, 3=info
        "-a", "1",          # save backtrace for later alignment
    ]
    print(f"Running mmseqs: {' '.join(cmd_cluster)}", file=sys.stderr)
    try:
        subprocess.run(cmd_cluster, check=True)
    except Exception as e:
        print(f"Error running mmseqs cluster: {e}", file=sys.stderr)
        ERROR_LOGS.append(f"Error running mmseqs cluster: {e}")
        return None

    # Step 3: (Optional) Create a TSV file describing the clusters
    cmd_createtsv = ["mmseqs", "createtsv", db, db, db_clu, cluster_tsv]
    print(f"Running mmseqs: {' '.join(cmd_createtsv)}", file=sys.stderr)
    try:
        subprocess.run(cmd_createtsv, check=True)
    except Exception as e:
        # You may choose to continue even if this fails.
        print(f"Error running mmseqs createtsv: {e}", file=sys.stderr)
        ERROR_LOGS.append(f"Error running mmseqs createtsv: {e}")
    if generate_msa:
        # Step 4: Generate multiple sequence alignments for each cluster
        cmd_result2msa = [
            "mmseqs", "result2msa",
            db, db, db_clu, msa_out,
            "--msa-format-mode", "3"
        ]
        print(f"Running mmseqs: {' '.join(cmd_result2msa)}", file=sys.stderr)
        try:
            subprocess.run(cmd_result2msa, check=True)
            return cluster_tsv, msa_out
        except Exception as e:
            print(f"Error running mmseqs result2msa: {e}", file=sys.stderr)
            ERROR_LOGS.append(f"Error running mmseqs result2msa: {e}")
            return cluster_tsv, None
    else:
        return cluster_tsv, None


def parse_mmseqs_cluster_results(cluster_tsv, sequence_ids):
    """
    Parse mmseqs cluster results to determine which cluster each sequence belongs to.
    
    Args:
        cluster_tsv: Path to the cluster.tsv file
        sequence_ids: List of sequence IDs to match with clusters
        
    Returns:
        np.array of cluster IDs
    """
    if not os.path.isfile(cluster_tsv):
        return np.array([''] * len(sequence_ids), dtype=str)
    
    # Handle missing sequences by joining with original IDs
    all_ids = pd.DataFrame({'accession': sequence_ids})
    cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=['cluster_rep', 'member'])
    merged = pd.merge(all_ids, cluster_df, how='left', left_on='accession', right_on='member')
    
    # Fill NA values with their own accession (singleton clusters)
    merged['cluster_id'] = merged['cluster_rep']
    merged = merged.drop_duplicates()
    if merged['cluster_id'].isnull().sum() > 0:
        for i, row in merged.iterrows():
            if pd.isnull(row.cluster_rep):
                merged.at[i, 'cluster_id'] = row.accession
    
    return dict(zip(merged.accession, merged.cluster_id))


def cluster_sequences(sequences, accessions, min_seq_id, threads, output_dir, generate_msa=False):
    """
    Cluster sequences using mmseqs at the specified identity threshold.
    
    Args:
        sequences: List of sequences to cluster
        accessions: List of sequence accessions
        min_seq_id: Minimum sequence identity threshold
        threads: Number of CPU threads for mmseqs
        output_dir: Directory to store temporary files
        
    Returns:
        np.array of cluster IDs
    """
    if len(sequences) == 0:
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
        cluster_tsv, msa_out = run_mmseqs_cluster(
            fasta_file, out_prefix, min_seq_id, threads, generate_msa
        )
        
        if cluster_tsv:
            accession_to_cluster_id = parse_mmseqs_cluster_results(cluster_tsv, accessions)
        else:
            accession_to_cluster_id = dict(zip(accessions, accessions))

    finally:
        # Clean up the directory
        shutil.rmtree(unique_dir)

    return accession_to_cluster_id


def process_msa_file(parquet_path, output_dir, threads, max_allowed_gaps=10, min_sub_seq_len=20):
    """
    Process an MSA file to extract and cluster subsequences.
    
    Args:
        parquet_path: Path to the input parquet file
        output_dir: Directory to save output files
        threads: Number of CPU threads for mmseqs
        max_allowed_gaps: Maximum number of consecutive gaps allowed before splitting
        min_sub_seq_len: Minimum length of subsequence to keep
    """
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    df = df.sample(frac=1).reset_index(drop=True)
    # Lists to store the fragmented sequences
    all_fragments = []
    parquet_index = 0
    # Process each MSA in the file
    for _, row in df.iterrows():
        msa_text = row['text']
        msa = parse_fasta(msa_text)
        
        # Process each sequence in the MSA
        all_subsequences = []
        all_accessions = []
        
        for header, sequence in msa:
            uniprot_id = extract_uniprot_id(header)
            subsequences = split_sequence(sequence, max_allowed_gaps, min_sub_seq_len)
            
            # Create unique IDs for each subsequence
            for i, subseq in enumerate(subsequences):
                if subseq not in all_subsequences:  
                    all_subsequences.append(subseq)
                    all_accessions.append(f"{uniprot_id}_{i}")
        
        # Skip if no valid subsequences were found
        if len(all_subsequences) == 0:
            continue
            
        # Initial clustering at 30% identity
        accession_to_cluster_id = cluster_sequences(
            all_subsequences, 
            all_accessions, 
            0.3,  # 30% identity
            threads, 
            os.path.join(output_dir, "clustering_tmp"),
            generate_msa=True
        )
        
        # Group sequences by 30% cluster ID
        clusters = {}
        for accession, subseq in zip(all_accessions, all_subsequences):
            cluster_id = accession_to_cluster_id[accession]
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'sequences': [], 
                    'accessions': []
                }
            clusters[cluster_id]['sequences'].append(subseq)
            clusters[cluster_id]['accessions'].append(accession)
        
        # Filter out singleton clusters
        clusters = {k: v for k, v in clusters.items() if len(v['sequences']) > 1}
        
        # For each 30% cluster, perform further clustering at higher identity thresholds
        for cluster_index, (cluster_id, cluster_data) in enumerate(clusters.items()):
            sequences = cluster_data['sequences']
            accessions = cluster_data['accessions']
            
            # Extract parent UniProt ID for family naming
            parent_uniprot = accessions[0].split('_')[0]
            fam_id = f"{parent_uniprot}_clust_{cluster_index}"
            
            # Create data structure for this cluster
            cluster_result = {
                'fam_id': fam_id,
                'sequences': np.array(sequences),
                'accessions': np.array(accessions)
            }
            assert len(sequences) == len(accessions)
            # Perform further clustering at higher identity thresholds
            for identity in [0.45, 0.65, 0.9, 0.95]:
                cluster_col = f"cluster_ids_{str(identity).replace('.', '_')}"
                accession_to_cluster_id = cluster_sequences(
                    sequences, 
                    accessions, 
                    identity,
                    threads, 
                    os.path.join(output_dir, "clustering_tmp")
                    generate_msa=False
                )
                cluster_result[cluster_col] = np.array([accession_to_cluster_id[acc] for acc in accessions], dtype=str)
            all_fragments.append(cluster_result)
            if len(all_fragments) == 10000:
                result_df = pd.DataFrame(all_fragments)
                result_df = result_df.sample(frac=1).reset_index(drop=True)
                filename = os.path.basename(parquet_path) + f"_fragments_{parquet_index}.parquet"
                output_path = os.path.join(output_dir, filename)
                result_df.to_parquet(output_path, index=False)
                print(f"Saved processed clusters to {output_path}")
                all_fragments = []
                parquet_index += 1
    
    # Create DataFrame and save to parquet
    if all_fragments:
        result_df = pd.DataFrame(all_fragments)
        result_df = result_df.sample(frac=1).reset_index(drop=True)
        filename = os.path.basename(parquet_path) + f"_fragments_{parquet_index}.parquet"
        output_path = os.path.join(output_dir, filename)
        result_df.to_parquet(output_path, index=False)
        print(f"Saved processed clusters to {output_path}")
    else:
        print(f"No valid clusters found in {parquet_path}")
    
    end_time = time.time()
    TIMINGS.append({
        "file": os.path.basename(parquet_path),
        "time_taken": end_time - start_time
    })


def main():
    parser = argparse.ArgumentParser(
        description="Process MSA files, split sequences, and cluster with MMSEQS."
    )
    parser.add_argument("--input_pattern", default="/mnt/disk2/cath_plm/data/openfold/uniclust30_filtered_parquet/*.parquet",
                        help="Pattern to match parquet input files.")
    parser.add_argument("--output_dir", default="/mnt/disk2/cath_plm/data/openfold/uniclust30_filtered_parquet_fragments",
                        help="Directory to save output files.")
    parser.add_argument("--max_allowed_gaps", type=int, default=10,
                        help="Maximum number of consecutive gaps allowed before splitting.")
    parser.add_argument("--min_sub_seq_len", type=int, default=20,
                        help="Minimum length of subsequence to keep.")
    parser.add_argument("--threads", type=int, default=20,
                        help="Number of CPU threads for mmseqs.")
    parser.add_argument("--task_index", type=int, default=None,
                        help="Index of the task to run (for parallel processing).")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Number of tasks to run (for parallel processing).")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "clustering_tmp"), exist_ok=True)
    
    # Gather all parquet files
    parquet_files = glob.glob(args.input_pattern)
    print(f"Found {len(parquet_files)} parquet files")
    
    # Handle task-based parallelism if specified
    if args.task_index is not None and args.num_tasks is not None:
        batch_size = (len(parquet_files) // args.num_tasks) + 1
        start_idx = args.task_index * batch_size
        end_idx = min(start_idx + batch_size, len(parquet_files))
        parquet_files = parquet_files[start_idx:end_idx]
        print(f"Processing {len(parquet_files)} parquet files in batch {args.task_index} of {args.num_tasks}")
    
    # Process each parquet file
    for parquet_path in parquet_files:
        process_msa_file(
            parquet_path,
            args.output_dir,
            args.threads,
            args.max_allowed_gaps,
            args.min_sub_seq_len
        )
    
    # Write timing information
    if TIMINGS:
        timings_df = pd.DataFrame(TIMINGS)
        timings_path = os.path.join(args.output_dir, f"timings_{args.task_index if args.task_index is not None else 'all'}.csv")
        timings_df.to_csv(timings_path, index=False)
        
        total_time = sum(t['time_taken'] for t in TIMINGS)
        print(f"Total processing time: {total_time:.2f} seconds")
    
    # Write error logs
    if ERROR_LOGS:
        error_path = os.path.join(args.output_dir, f"errors_{args.task_index if args.task_index is not None else 'all'}.txt")
        with open(error_path, "w") as f:
            for error in ERROR_LOGS:
                f.write(f"{error}\n")


if __name__ == "__main__":
    main() 