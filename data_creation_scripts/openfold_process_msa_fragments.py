"""
Created to process MSA alignments from openfold parquet files.
This script:
1. Reads MSA data in a3m format from parquet files
2. Splits sequences at regions with >10 consecutive gaps
3. Processes subsequences (removes gaps, converts lowercase to uppercase, filters by length)
4. Clusters subsequences using MMSEQS at 30% identity
5. Generates alignmenst for each 30% SI cluster
6. Further clusters within 30% clusters at higher identity thresholds
7. Formats results into new parquet files
Note that you still need to do a global shuffle of the resultant parquet files
by running:
data_creation_scripts/shuffling/shuffle_openfold.sh
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
from tqdm import tqdm
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
    split_string = "-" * (max_allowed_gaps + 1)
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


def run_mmseqs_cluster(fasta_file, out_prefix, min_seq_id, threads, generate_msa=False, db_path=None, sensitivity=4.0):
    """
    Runs mmseqs cluster and then generates MSAs for each cluster.
    
    The steps are:
      1. Create a database from the FASTA file (if fasta_file is provided).
      2. Run mmseqs cluster to cluster the sequences.
      3. Optionally, create a TSV file with cluster info.
      4. Generate MSAs from the clusters using mmseqs result2msa.
    """
    # Define database and temporary file names
    db_clu = f"{out_prefix}_DB_clu"
    tmp = f"{out_prefix}_tmp"
    msa_out = f"{out_prefix}_DB_clu_msa"
    cluster_tsv = f"{out_prefix}_cluster.tsv"

    # Step 1: Create database from FASTA
    if fasta_file:
        db_path = f"{out_prefix}_DB"
        cmd_create_db = [
            "mmseqs", "createdb", 
            fasta_file, 
            db_path,
            "--shuffle", "0",
            "-v", "1"
        ]
        # print(f"Running mmseqs: {' '.join(cmd_create_db)}", file=sys.stderr)
        try:
            subprocess.run(cmd_create_db, check=True)
        except Exception as e:
            print(f"Error creating database: {e}", file=sys.stderr)
            ERROR_LOGS.append(f"Error creating database: {e}")
            return None, None

    # Step 2: Run clustering on the database
    cmd_cluster = [
        "mmseqs", "cluster", db_path, db_clu, tmp,
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads),
        "--cov-mode", "0",  # both sequences must have at least c coverage
        "-c", "0.7",        # coverage threshold
        "--cluster-mode", "1",
        "-v", "1",          # verbosity: 0=quiet, 3=info
        "-a", "1",          # save backtrace for later alignment
        "-s", str(sensitivity),
    ]
    # print(f"Running mmseqs: {' '.join(cmd_cluster)}", file=sys.stderr)
    try:
        subprocess.run(cmd_cluster, check=True)
    except Exception as e:
        print(f"Error running mmseqs cluster: {e}", file=sys.stderr)
        ERROR_LOGS.append(f"Error running mmseqs cluster: {e}")
        return None, None

    # Step 3: Create a TSV file describing the clusters
    cmd_createtsv = [
        "mmseqs", "createtsv", db_path, db_path, db_clu, cluster_tsv,
        "-v", "1"
    ]
    # print(f"Running mmseqs: {' '.join(cmd_createtsv)}", file=sys.stderr)
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
            db_path, db_path, db_clu, msa_out,
            "--msa-format-mode", "3",
            "-v", "1"
        ]
        # print(f"Running mmseqs: {' '.join(cmd_result2msa)}", file=sys.stderr)
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
    cluster_id_to_accessions = {}
    for cluster_id in merged.cluster_id.unique():
        accessions = list(merged[merged.cluster_id == cluster_id].accession.unique())
        if len(accessions) > 1:
            cluster_id_to_accessions[cluster_id] = accessions

    return cluster_id_to_accessions


def parse_mmseqs_msa_file(msa_file):
    """
    Parse the MMseqs2 MSA output file to extract aligned sequences.
    
    Args:
        msa_file: Path to the MMseqs2 MSA output file
        
    Returns:
        Dictionary mapping sequence accessions to their aligned sequences
    """
    if not os.path.isfile(msa_file):
        return {}

    aligned_sequences = {}
    current_cluster = None
    current_header = None
    new_cluster = True
    cluster_counter = 0
    with open(msa_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '#cl-' in line:
                # # New cluster header
                new_cluster = True
            elif line.startswith('>'):
                # Sequence header
                current_header = line[1:]
                if new_cluster:
                    cluster_representative = f"{current_header}_clust_{cluster_counter}"
                    if cluster_representative in aligned_sequences:
                        cluster_counter += 1
                        cluster_representative = f"{current_header}_clust_{cluster_counter}"
                    else:
                        cluster_counter = 0
                        cluster_representative = f"{current_header}_clust_{cluster_counter}"
                    aligned_sequences[cluster_representative] = {}
                    new_cluster = False
            elif line and current_header:
                # Sequence data
                aligned_sequences[cluster_representative][current_header] = line
                current_header = None

    return aligned_sequences


def cluster_sequences(sequences, accessions, min_seq_id, threads, temp_dir, generate_msa=False, reuse_db=None, sensitivity=4.0):
    """
    Cluster sequences using mmseqs at the specified identity threshold.
    
    Args:
        sequences: List of sequences to cluster
        accessions: List of sequence accessions
        min_seq_id: Minimum sequence identity threshold
        threads: Number of CPU threads for mmseqs
        temp_dir: Directory to store temporary files
        generate_msa: Whether to generate MSA files for clusters
        reuse_db: Optional path to reuse an existing MMseqs database
        
    Returns:
        Dictionary mapping accessions to cluster IDs and aligned sequences if generate_msa=True
    """
    if len(sequences) == 0:
        return {}

    # Generate a unique directory for this clustering job
    unique_dir = os.path.join(temp_dir, uuid.uuid4().hex)
    os.makedirs(unique_dir, exist_ok=True)
    out_prefix = os.path.join(unique_dir, "cluster")
    fasta_file = os.path.join(unique_dir, "input.fasta")
    db = f"{out_prefix}_DB"

    try:
        # If not reusing a database, create a new one
        if reuse_db is None:
            # Write sequences to FASTA
            with open(fasta_file, 'w') as f:
                for i, seq in enumerate(sequences):
                    f.write(f">{accessions[i]}\n{seq}\n")

            # Create database from FASTA
            cmd_create_db = [
                "mmseqs", "createdb", 
                fasta_file, 
                db, 
                "--shuffle", "0", 
                "-v", "1"
                ]
            # print(f"Running mmseqs: {' '.join(cmd_create_db)}", file=sys.stderr)
            try:
                subprocess.run(cmd_create_db, check=True)
                reuse_db = db
            except Exception as e:
                print(f"Error creating database: {e}", file=sys.stderr)
                ERROR_LOGS.append(f"Error creating database: {e}")
                return {}
        else:
            # Reuse existing database
            db = reuse_db

        # Run mmseqs and parse results
        cluster_tsv, msa_out = run_mmseqs_cluster(
            fasta_file if reuse_db is None else None, 
            out_prefix, 
            min_seq_id, 
            threads, 
            generate_msa,
            db_path=db,
            sensitivity=sensitivity
        )

        # Parse cluster assignments
        if cluster_tsv:
            cluster_id_to_accessions = parse_mmseqs_cluster_results(cluster_tsv, accessions)
        else:
            cluster_id_to_accessions = dict(zip(accessions, accessions))

        # Parse aligned sequences if MSA was generated
        aligned_sequences = {}
        if generate_msa and msa_out:
            aligned_sequences = parse_mmseqs_msa_file(msa_out)

        result = {
            'cluster_id_to_accessions': cluster_id_to_accessions,
            'aligned_sequences': aligned_sequences,
            'db_path': db
        }

    finally:
        shutil.rmtree(unique_dir)

    return result


def process_msa_file(
        parquet_path, 
        output_dir, 
        threads,
        temp_dir,
        max_allowed_gaps=10, 
        min_sub_seq_len=20, 
        generate_additional_clusters=True,
    ):
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

    # Check if the output file already exists
    filenames = os.path.basename(parquet_path) + f"_fragments_*.parquet"
    output_path =f"{output_dir}/{filenames}"
    if len(glob.glob(output_path)) > 0:
        print(f"Skipping {parquet_path} because it already exists")
        return

    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    # Lists to store the fragmented sequences
    all_fragments = []
    parquet_index = 0
    # Process each MSA in the file
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows in parquet file"):
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

        if len(all_subsequences) < 3:
            continue
        # Initial clustering at 30% identity
        clustering_result = cluster_sequences(
            all_subsequences, 
            all_accessions, 
            0.3,  # 30% identity
            threads, 
            temp_dir,
            generate_msa=True
        )
        # print(f"Time taken to cluster sequences: {end_time - start_time:.2f} seconds")

        cluster_id_to_accessions = clustering_result['cluster_id_to_accessions']
        aligned_sequences = clustering_result['aligned_sequences']
        db_path = clustering_result.get('db_path')


        parent_uniprot = all_accessions[0].split('_')[0]
        for cluster_index, (sub_seq_rep_accession, acc2seq) in enumerate(aligned_sequences.items()):
            acc2seq = {k:v for k,v in acc2seq.items() if len(v.replace("-", "")) >= min_sub_seq_len}
            if len(acc2seq) < 2:
                continue
            accessions = list(acc2seq.keys())
            sequences = list(acc2seq.values())
            if len(set([len(seq) for seq in sequences])) > 1:
                bp=1
                continue
            fam_id = f"{parent_uniprot}_clust_{cluster_index}"
            cluster_result = {
                'fam_id': fam_id,
                'sequences': np.array(sequences),
                'accessions': np.array(accessions),
            }

            if generate_additional_clusters:
                # Create a new database for this specific 30% cluster
                cluster_tmp_dir = os.path.join(temp_dir, "clustering_tmp", uuid.uuid4().hex)
                os.makedirs(cluster_tmp_dir, exist_ok=True)
                cluster_fasta = os.path.join(cluster_tmp_dir, "cluster_input.fasta")
                cluster_db_prefix = os.path.join(cluster_tmp_dir, "cluster_db")
                cluster_db = f"{cluster_db_prefix}"

                # Write sequences for this cluster to FASTA
                with open(cluster_fasta, 'w') as f:
                    for acc, seq in zip(accessions, sequences):
                        f.write(f">{acc}\n{seq}\n")

                # Create database for this cluster
                cmd_create_db = [
                    "mmseqs", "createdb", 
                    cluster_fasta, 
                    cluster_db, 
                    "--shuffle", "0", 
                    "-v", "1"
                    ]
                try:
                    subprocess.run(cmd_create_db, check=True)
                except Exception as e:
                    print(f"Error creating database for cluster: {e}", file=sys.stderr)
                    ERROR_LOGS.append(f"Error creating database for cluster: {e}")
                    # Clean up the temporary directory
                    shutil.rmtree(cluster_tmp_dir)
                    continue

                # Perform further clustering at higher identity thresholds
                for identity in [0.40, 0.50, 0.65, 0.80, 0.90, 0.95]:
                    print(f"Clustering at {identity} identity")
                    cluster_col = f"cluster_ids_{str(identity).replace('.', '_')}"
                    higher_clustering_result = cluster_sequences(
                        sequences, 
                        accessions, 
                        identity,
                        threads, 
                        cluster_tmp_dir,
                        generate_msa=False,
                        reuse_db=cluster_db,
                        sensitivity=1.0
                    )
                    accession_to_cluster_id = higher_clustering_result['cluster_assignments']
                    cluster_result[cluster_col] = np.array([accession_to_cluster_id[acc] for acc in accessions], dtype=str)

                # Clean up the temporary directory for this cluster
                shutil.rmtree(cluster_tmp_dir)

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

        # Clean up the original database after processing the entire row
        if db_path and os.path.exists(os.path.dirname(db_path)):
            try:
                shutil.rmtree(os.path.dirname(db_path))
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}", file=sys.stderr)
                ERROR_LOGS.append(f"Error cleaning up temporary directory: {e}")

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
    parser.add_argument("--input_pattern", default="../data/openfold/uniclust30_filtered_parquet/*.parquet",
                        help="Pattern to match parquet input files.")
    parser.add_argument("--output_dir", default="../data/openfold/uniclust30_filtered_parquet_fragments_ucl_cluster_v2",
                        help="Directory to save output files.")
    parser.add_argument("--max_allowed_gaps", type=int, default=10,
                        help="Maximum number of consecutive gaps allowed before splitting.")
    parser.add_argument("--min_sub_seq_len", type=int, default=90,
                        help="Minimum length of subsequence to keep.")
    parser.add_argument("--threads", type=int, default=20,
                        help="Number of CPU threads for mmseqs.")
    parser.add_argument("--task_index", type=int, default=None,
                        help="Index of the task to run (for parallel processing).")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Number of tasks to run (for parallel processing).")
    parser.add_argument("--generate_additional_clusters", type=bool, default=False,
                        help="Generate additional clusters at higher identity thresholds.")
    parser.add_argument("--scratch_dir", type=str, default=None,
                        help="Directory to store temporary files.")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.scratch_dir:
        temp_dir = os.path.join(args.scratch_dir, "clustering_tmp")
    else:
        temp_dir = os.path.join(args.output_dir, "clustering_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # Gather all parquet files
    parquet_files = sorted(glob.glob(args.input_pattern))
    # set seed for reproducibility
    np.random.seed(42)
    parquet_files = np.random.permutation(parquet_files)
    print(f"Found {len(parquet_files)} parquet files")

    # Handle task-based parallelism if specified
    if args.task_index is not None and args.num_tasks is not None:
        batch_size = (len(parquet_files) // args.num_tasks)
        start_idx = args.task_index * batch_size
        if args.task_index == args.num_tasks - 1:
            end_idx = len(parquet_files)
        else:
            end_idx = min(start_idx + batch_size, len(parquet_files))
        parquet_files = parquet_files[start_idx:end_idx]
        print(f"Processing {len(parquet_files)} parquet files in batch {args.task_index} of {args.num_tasks}")

    # Process each parquet file
    for parquet_path in parquet_files:
        process_msa_file(
            parquet_path,
            output_dir=args.output_dir,
            threads=args.threads,
            temp_dir=temp_dir,
            max_allowed_gaps=args.max_allowed_gaps,
            min_sub_seq_len=args.min_sub_seq_len,
            generate_additional_clusters=args.generate_additional_clusters,
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