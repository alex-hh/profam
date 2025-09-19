"""
Created by Jude Wells 2025-04-10
Creates a new train / val / test split using a combination of 
FunFams, Foldseek Clusters and Pfam

Pick 100 random foldseek clusters minimum size 5 maximum size 1000
Pick 50 FunFams minimum size 5 maximum size 1000
Use the selected Pfam families that were already selected
Hold out all experimentally confirmed PETase sequences

Note that at this stage we create intermediate train / val / test parquets 
which are located here:
data/{dataset_name}/train_test_split_v2/{train/val/test}/*.parquet
where val and test will consist of a single parquet file with the heldout families
and train will consist of multiple parquets with the remaining families.
These are not the final train parquets because we still need to do the MMSEQS filtering
of all the combined heldout sequences from all the datasets.

The filtering step generates new parquet files here:
data/{dataset_name}/train_test_split_v2/{train/val/test}_filtered/*.parquet


Compile all the sequences from all of the above familes
Creats a mmseqs database of all these sequences.
Create a mmseqs database of all other datasets.

remove everything that has 30% identity with 80% coverage of the held-out sequence.

For each dataset: 
1) choose families for the split 
2) create new train / val / test split parquets
3) gather all sequences from val and test
4) aggregate all sequences from all datasets
5) Do a 90% SI clustering of the aggregated sequences
6) take representative for each S90 cluster
7) create a mmseqs database of the representative sequences in val and test
8) remove everything from training set that has 30% identity with 80% coverage of the held-out sequences

Datasets which need to be searched against:
- Uniref90
- AFDB_s50_single
- FunFamsS100
- TEDS100
- Pfam
- Foldseek_s100
- OpenFold clustered


For each dataset that needs to be searched against:
Iterate over the parquet files
- create a fasta file of all the sequences in the parquet file include the filename, fam_id, accession as the header
- convert the fasta into an MMSeqs database
- run the mmseqs search command 
- make a csv of all the sequences that match (filename, fam_id, accession)
- remove the sequences from the training set that match (filter the sequence and the accession arrays in the relevant row)
- if the entire family is removed, remove the row from the parquet file and add it to the list of validation or testing rows.
- clean up the temporary files
"""
import shutil
import pandas as pd
import tqdm
import os
import glob
import random
import numpy as np
import subprocess
import uuid
import argparse
import math


np.random.seed(42)

datasets_to_filter = [


    {
        "name": "FunFamsS100",
        "parquet_pattern": "../data/funfams/s100_noali_parquets/train_test_split_v2/train/*.parquet",
        "output_dir": "../data/funfams/s100_noali_parquets/train_test_split_v2",
        "task_indices": list(range(0,11))
    },
    {
        "name": "Foldseek_s100",
        "parquet_pattern": "../data/foldseek/foldseek_s100_raw/train_test_split_v2/train/*.parquet",
        "output_dir": "../data/foldseek/foldseek_s100_raw/train_test_split_v2",
        "task_indices": list(range(11,112))
    },
    {
        "name": "TEDS100",
        "parquet_pattern": "../data/ted/s100_parquets/train_val_test_split_hq/train_val_test_split/*/clustered/*.parquet",
        "output_dir": "../data/ted/s100_parquets/train_test_split_v3_debug",
        "task_indices": list(range(112, 150))
    },
    {
        "name": "Pfam",
        "parquet_pattern": "../data/pfam/train_test_split_parquets/train/*.parquet",
        "output_dir": "../data/pfam/train_test_split_parquets/train_test_split_v2",
        "task_indices": list(range(150, 200))
    },
    {
        "name": "OpenFold_clustered",
        "parquet_pattern": "../data/openfold/uniclust30_clustered_shuffled_final/*.parquet",
        "output_dir": "../data/openfold/uniclust30_clustered_shuffled_final/train_test_split_v2",
        "task_indices": list(range(200, 260))
    },
    {
        "name": "Uniref90",
        "parquet_pattern": "../data/uniref/uniref90_parquets_shuffled/train/*.parquet",
        "output_dir": "../data/uniref/uniref90_parquets_shuffled/train_test_split_v2",
        "task_indices": list(range(260, 310))
    },
    {
        "name": "afdb_s50_single",
        "parquet_pattern": "../data/afdb_s50_single/*.parquet",
        "output_dir": "../data/afdb/afdb_s50_single_parquets/train_test_split_v2",
        "task_indices": list(range(310, 335))
    },
]

# Dictionary to track selected families and their details
selected_families_info = []

# Global variable for scratch directory
SCRATCH_DIR = None

def format_sequence(sequence):
    return sequence.replace(".", "").replace("-", "").upper()

def run_mmseqs_easy_cluster(fasta_file, out_prefix, min_seq_id=0.9, coverage=0.8, threads=20):
    """
    Run mmseqs easy-cluster on a fasta file with specified sequence identity and coverage thresholds.
    Returns the path to the representative sequences fasta file.
    """
    cmd = [
        "mmseqs", "easy-cluster",
        fasta_file,
        out_prefix,
        out_prefix,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--threads", str(threads),
        "--remove-tmp-files", "1",
        "--cluster-mode", "1",
        "-v", "1",  # 0=silent, 1=errors, 2=warnings
    ]
    print(f"Running mmseqs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    rep_seq_fasta = f"{out_prefix}_rep_seq.fasta"
    return rep_seq_fasta

def get_petase_heldout_sequences():
    petase_fasta_path = "../data/petase/combined_petase_sequences.fasta"
    petase_sequences = []
    current_seq = ""
    with open(petase_fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if len(current_seq) > 0:
                    petase_sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line.strip()
    if len(current_seq) > 0:
        petase_sequences.append(current_seq)
    selected_families_info.append({
        "fam_id": "petase",
        "dataset": "Petase",
        "num_sequences": len(petase_sequences),
        "split": "test"
    })
    return petase_sequences, ["test"] * len(petase_sequences)

def get_pfam_heldout_sequences():
    heldout_sequences = []
    heldout_splits = []
    pfam_val_parquet_path = "../data/pfam/train_test_split_parquets/val_clustered_split_combined.parquet"
    pfam_test_parquet_path = "../data/pfam/train_test_split_parquets/test_clustered_split_combined.parquet"

    pfam_val_df = pd.read_parquet(pfam_val_parquet_path)
    pfam_test_df = pd.read_parquet(pfam_test_parquet_path)

    for split, df in zip(["val", "test"], [pfam_val_df, pfam_test_df]):
        for i, row in df.iterrows():
            heldout_sequences.extend([format_sequence(seq) for seq in row["sequences"]])
            heldout_splits.extend([split] * len(row["sequences"]))
            selected_families_info.append({
                "fam_id": row.get("fam_id", f"pfam_{i}"),
                "dataset": "Pfam",
                "num_sequences": len(row["sequences"]),
                "split": split
            })

    return heldout_sequences, heldout_splits


def make_funfams_train_test_split():
    funfams_pure_path = "data/funfams_ec_pure.tsv"
    funfams_parquet_pattern = "../data/funfams/s100_noali_parquets/*.parquet"
    output_dir = "../data/funfams/s100_noali_parquets/train_test_split_v2"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    ec_pure_funfams = pd.read_csv(funfams_pure_path, sep="\t")
    random_family_selection = np.random.choice(ec_pure_funfams["FunFam_ID"], size=100, replace=False)
    np.random.shuffle(random_family_selection)
    val_families = random_family_selection[:40]
    test_families = random_family_selection[40:]
    all_heldout_families = set(val_families).union(set(test_families))
    val_df_rows = []
    test_df_rows = []
    heldout_sequences = []
    heldout_splits = []
    for parquet_path in glob.glob(funfams_parquet_pattern):
        train_drop_indices = []
        df = pd.read_parquet(parquet_path)
        df['fam_id'] = df.fam_id.apply(lambda x: x.replace(".fasta", ""))
        heldout_rows = df[df['fam_id'].isin(all_heldout_families)]
        if len(heldout_rows) > 0:
            for i, row in heldout_rows.iterrows():
                if row["fam_id"] in val_families:
                    val_df_rows.append(row)
                    train_drop_indices.append(i)
                    heldout_sequences.extend(row["sequences"])
                    heldout_splits.extend(["val"] * len(row["sequences"]))

                    selected_families_info.append({
                        "fam_id": row["fam_id"],
                        "dataset": "FunFams",
                        "num_sequences": len(row["sequences"]),
                        "split": "val"
                    })
                    
                elif row["fam_id"] in test_families:
                    test_df_rows.append(row)
                    train_drop_indices.append(i)
                    heldout_sequences.extend(row["sequences"])
                    heldout_splits.extend(["test"] * len(row["sequences"]))

                    selected_families_info.append({
                        "fam_id": row["fam_id"],
                        "dataset": "FunFams",
                        "num_sequences": len(row["sequences"]),
                        "split": "test"
                    })
                    
            train_df = df.drop(train_drop_indices)
        else:
            train_df = df
        train_save_path = os.path.join(output_dir, "train", f"train_{parquet_path.split('/')[-1]}")
        train_df.to_parquet(train_save_path, index=False)
    if len(val_df_rows) > 0:
        val_save_path = os.path.join(output_dir, "val", f"funfams_s100_noali_val.parquet")
        val_df = pd.DataFrame(val_df_rows)
        val_df.to_parquet(val_save_path, index=False)
    if len(test_df_rows) > 0:
        test_save_path = os.path.join(output_dir, "test", f"funfams_s100_noali_test.parquet")
        test_df = pd.DataFrame(test_df_rows)
        test_df.to_parquet(test_save_path, index=False)
    with open(os.path.join(output_dir, "heldout_sequences.txt"), "w") as f:
        for seq in heldout_sequences:
            f.write(seq + "\n")
    return heldout_sequences, heldout_splits


def make_foldseek_train_test_split():
    foldseek_parquet_pattern = "../data/foldseek/foldseek_s100_raw/*.parquet"
    output_dir = "../data/foldseek/foldseek_s100_raw/train_test_split_v2"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    n_holdout_clusters = 200
    foldseek_parquets = glob.glob(foldseek_parquet_pattern)
    n_foldseek_parquets = len(foldseek_parquets)
    assert n_holdout_clusters <= n_foldseek_parquets
    files_with_holdouts = np.random.choice(foldseek_parquets, size=n_holdout_clusters, replace=False)
    np.random.shuffle(files_with_holdouts)
    val_holdout_files = files_with_holdouts[:80]
    test_holdout_files = files_with_holdouts[80:]
    val_df_rows = []
    test_df_rows = []
    heldout_sequences = []
    heldout_splits = []
    for parquet_path in foldseek_parquets:
        df = pd.read_parquet(parquet_path)
        drop_indices = []
        if parquet_path in files_with_holdouts:
            holdout_index = np.random.choice(df.index, size=1, replace=False)[0]
            drop_indices.append(holdout_index)
            
            if parquet_path in val_holdout_files:
                val_df_rows.append(df.iloc[holdout_index])
                drop_indices.append(holdout_index)
                seqs = df.iloc[holdout_index]["sequences"]
                heldout_sequences.extend(seqs)
                heldout_splits.extend(["val"] * len(seqs))

                selected_families_info.append({
                    "fam_id": df.iloc[holdout_index]["fam_id"],
                    "dataset": "Foldseek",
                    "num_sequences": len(seqs),
                    "split": "val"
                })
                
            elif parquet_path in test_holdout_files:
                test_df_rows.append(df.iloc[holdout_index])
                drop_indices.append(holdout_index)
                seqs = df.iloc[holdout_index]["sequences"]
                heldout_sequences.extend(seqs)
                heldout_splits.extend(["test"] * len(seqs))

                selected_families_info.append({
                    "fam_id": df.iloc[holdout_index]["fam_id"],
                    "dataset": "Foldseek",
                    "num_sequences": len(seqs),
                    "split": "test"
                })
                
            df = df.drop(drop_indices)
        train_save_path = os.path.join(output_dir, "train", f"foldseek_s100_raw_train_{parquet_path.split('/')[-1]}")
        df.to_parquet(train_save_path, index=False)

    if len(val_df_rows) > 0:
        val_save_path = os.path.join(output_dir, "val", f"foldseek_s100_raw_val.parquet")
        val_df = pd.DataFrame(val_df_rows)
        val_df.to_parquet(val_save_path, index=False)
    if len(test_df_rows) > 0:
        test_save_path = os.path.join(output_dir, "test", f"foldseek_s100_raw_test.parquet")
        test_df = pd.DataFrame(test_df_rows)
        test_df.to_parquet(test_save_path, index=False)
    return heldout_sequences, heldout_splits


def save_sequences_to_fasta(sequences, splits, output_path):
    """
    Save a list of sequences to a FASTA file.
    Each sequence gets a unique identifier.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i, (seq, split) in enumerate(zip(sequences, splits)):
            f.write(f">{split}_{i}\n{seq}\n")
    return output_path


def cluster_sequences_with_mmseqs(sequences, splits, reps_output_path, min_seq_id=0.9, coverage=0.8, threads=20):
    """
    Cluster a list of sequences using mmseqs and return representatives.
    """
    # Create a temporary directory for mmseqs files
    global SCRATCH_DIR
    if SCRATCH_DIR:
        tmp_dir = os.path.join(SCRATCH_DIR, f"tmp_mmseqs_{uuid.uuid4().hex}")
    else:
        tmp_dir = f"tmp_mmseqs_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # Write sequences to a FASTA file
        input_fasta = os.path.join(tmp_dir, "input.fasta")
        fasta_path = save_sequences_to_fasta(sequences, splits, input_fasta)
        
        # Run mmseqs clustering
        out_prefix = os.path.join(tmp_dir, "cluster")
        rep_fasta = run_mmseqs_easy_cluster(
            fasta_path, 
            out_prefix, 
            min_seq_id=min_seq_id, 
            coverage=coverage, 
            threads=threads
        )
        shutil.copy(rep_fasta, reps_output_path)
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir)


def select_heldout_families(rep_fasta_path):
    heldout_info_path = "data/selected_heldout_families_info.csv"
    heldout_sequences_path = "data/all_heldout_sequences.csv"
    if not os.path.exists(heldout_info_path) or not os.path.exists(heldout_sequences_path):
        all_heldout_sequences = []
        all_heldout_splits = []

        pfam_heldout_sequences, pfam_heldout_splits = get_pfam_heldout_sequences()
        all_heldout_sequences.extend(pfam_heldout_sequences)
        all_heldout_splits.extend(pfam_heldout_splits)
        
        foldseek_heldout_sequences, foldseek_heldout_splits = make_foldseek_train_test_split()
        all_heldout_sequences.extend(foldseek_heldout_sequences)
        all_heldout_splits.extend(foldseek_heldout_splits)

        funfams_heldout_sequences, funfams_heldout_splits = make_funfams_train_test_split()
        all_heldout_sequences.extend(funfams_heldout_sequences)
        all_heldout_splits.extend(funfams_heldout_splits)

        petase_heldout_sequences, petase_heldout_splits = get_petase_heldout_sequences()
        all_heldout_sequences.extend(petase_heldout_sequences)
        all_heldout_splits.extend(petase_heldout_splits)
        
        print(f"FunFams held-out sequences: {len(funfams_heldout_sequences)}")
        print(f"Foldseek held-out sequences: {len(foldseek_heldout_sequences)}")
        print(f"Pfam held-out sequences: {len(pfam_heldout_sequences)}")
        print(f"Petase held-out sequences: {len(petase_heldout_sequences)}")
        

        all_heldout_sequences = list(set(all_heldout_sequences)) # remove duplicates
        print(f"Total unique held-out sequences: {len(all_heldout_sequences)}")
        
        with open(heldout_sequences_path, "w") as f:
            for seq, split in zip(all_heldout_sequences, all_heldout_splits):
                f.write(f"{split},{seq}\n")
        
        families_df = pd.DataFrame(selected_families_info)
        families_df.to_csv(heldout_info_path, index=False)
        print(f"Saved information about {len(families_df)} selected families to data/selected_heldout_families_info.csv")
    else:
        families_df = pd.read_csv(heldout_info_path)
        all_heldout_sequences = pd.read_csv(heldout_sequences_path, header=None)[1].tolist()
        all_heldout_splits = pd.read_csv(heldout_sequences_path, header=None)[0].tolist()

    # Cluster all held-out sequences and get representatives
    print(f"Clustering {len(all_heldout_sequences)} sequences at 90% identity, 80% coverage...")
    cluster_sequences_with_mmseqs(
        all_heldout_sequences,
        all_heldout_splits,
        rep_fasta_path,
        min_seq_id=0.9,
        coverage=0.8
    )
    assert os.path.exists(rep_fasta_path)
    n_reduced_sequences = len(pd.read_csv(rep_fasta_path, header=None)) // 2
    print(f"Reduced {len(all_heldout_sequences)} sequences to {n_reduced_sequences} representatives")
    print(f"Representatives saved to {rep_fasta_path}")


def remove_similar_sequences_from_train_set(rep_fasta_path, datasets_to_filter, task_index=None):
    global SCRATCH_DIR

    # Create worker‑local base tmp dir (isolated for this task / PID)
    unique_suffix = f"task_{task_index}" if task_index is not None else f"pid_{os.getpid()}"
    base_tmp_root = SCRATCH_DIR if SCRATCH_DIR else "data/tmp"
    tmp_base_dir = os.path.join(base_tmp_root, unique_suffix)
    os.makedirs(tmp_base_dir, exist_ok=True)

    # Build an MMSeqs DB for the representative held‑out sequences in this local tmp dir
    rep_db_path = os.path.join(tmp_base_dir, "heldout_rep_seqs")
    subprocess.run([
        "mmseqs", "createdb",
        rep_fasta_path,
        rep_db_path,
        "-v", "1",  # 0=silent, 1=errors, 2=warnings
    ], check=True)
    
    for dataset in datasets_to_filter:
        print(f"Processing dataset: {dataset['name']}")
        os.makedirs(dataset["output_dir"], exist_ok=True)
        os.makedirs(os.path.join(dataset["output_dir"], "train_filtered"), exist_ok=True)
        os.makedirs(os.path.join(dataset["output_dir"], "val_filtered"), exist_ok=True)
        os.makedirs(os.path.join(dataset["output_dir"], "test_filtered"), exist_ok=True)
        
        parquet_files = sorted(glob.glob(dataset["parquet_pattern"]))
        print(f"Found {len(parquet_files)} parquet files for {dataset['name']}")
        if task_index is not None:
            # select batch of files corresponding to the task index
            batch_index = dataset["task_indices"].index(task_index)
            batch_size = len(parquet_files) // len(dataset["task_indices"]) + 1
            parquet_files = parquet_files[batch_index * batch_size:(batch_index + 1) * batch_size]
            print(
                f"Processing batch {batch_index} of {len(dataset['task_indices'])} with {len(parquet_files)} files in this batch"
                )
            
        removed_from_train = []
        for parquet_file in tqdm.tqdm(parquet_files):
            print(f"Processing file: {os.path.basename(parquet_file)}")
            train_output_path = os.path.join(dataset["output_dir"], "train_filtered", f"train_{os.path.basename(parquet_file)}")
            if os.path.exists(train_output_path):
                try:
                    pd.read_parquet(train_output_path)
                    print(f"Skipping {os.path.basename(parquet_file)} because it already exists")
                    continue
                except:
                    print(f"Removing {train_output_path} because it is corrupted")
                    os.remove(train_output_path)
            df = pd.read_parquet(parquet_file)
            
            
            # Create a temporary directory for this file
            if SCRATCH_DIR:
                tmp_dir = os.path.join(SCRATCH_DIR, f"{dataset['name']}_{uuid.uuid4().hex}")
            else:
                tmp_dir = os.path.join(tmp_base_dir, f"{dataset['name']}_{uuid.uuid4().hex}")
            os.makedirs(tmp_dir, exist_ok=True)
            
            try:
                # Create a fasta file of all sequences in the parquet
                fasta_path = os.path.join(tmp_dir, "sequences.fasta")
                with open(fasta_path, 'w') as f:
                    for i, row in df.iterrows():
                        fam_id = row.get('fam_id', f"fam_{i}")
                        for j, seq in enumerate(row['sequences']):
                            formatted_seq = format_sequence(seq)
                            accession = row['accessions'][j]
                            # Use filename, fam_id, row index and sequence index as header
                            header = f"{os.path.basename(parquet_file)}|{fam_id}|{i}|{j}|{accession}"
                            f.write(f">{header}\n{formatted_seq}\n")
                
                # Create MMSeqs database for the sequences
                query_db_path = os.path.join(tmp_dir, "query_db")
                subprocess.run([
                    "mmseqs", "createdb", 
                    fasta_path, 
                    query_db_path,
                    "-v", "1",  # 0=silent, 1=errors, 2=warnings
                ], check=True)
                
                # Run MMSeqs search
                result_path = os.path.join(tmp_dir, "search_result")
                tmp_path = os.path.join(tmp_dir, "tmp")
                os.makedirs(tmp_path, exist_ok=True)
                
                subprocess.run([
                    "mmseqs", "search",
                    query_db_path,
                    rep_db_path,
                    result_path,
                    tmp_path,
                    "--min-seq-id", "0.3",     # 30% identity threshold
                    "-c", "0.8",               # 80% coverage threshold
                    "--threads", "20",
                    "-v", "1",  # 0=silent, 1=errors, 2=warnings
                ], check=True)
                
                # Convert results to readable format
                result_tsv = os.path.join(tmp_dir, "search_result.tsv")
                subprocess.run([
                    "mmseqs", "convertalis",
                    query_db_path,
                    rep_db_path,
                    result_path,
                    result_tsv,
                    "--format-output", "query,target,pident,alnlen,qlen,tlen,qcov,tcov",
                    "-v", "1",  # 0=silent, 1=errors, 2=warnings
                ], check=True)
                
                if os.path.exists(result_tsv) and os.path.getsize(result_tsv) > 0:
                    result_df = pd.read_csv(result_tsv, sep="\t", header=None)
                    result_df.columns = ["query_header", "target_header", "pident", "alnlen", "qlen", "tlen", "qcov", "tcov"]
                    matching_headers = result_df["query_header"].unique()
                else:
                    print(f"No matches found for {os.path.basename(parquet_file)}")
                    shutil.copy(parquet_file, train_output_path)
                    
                    continue
                    
                # Process each row to remove matching sequences
                rows_to_keep = []
                val_rows = []
                test_rows = []
                
                for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(parquet_file)}"):
                    if os.path.basename(parquet_file) == "train_096.parquet" and dataset['name'] == "TEDS100" and i ==31:
                        print("skipping bad row")
                        continue
                    if os.path.basename(parquet_file) == "train_funfams_s100_noali_data_00_009.parquet" and i ==820:
                        print("skipping bad row")
                        continue
                    fam_id = row.get('fam_id', f"fam_{i}")
                    keep_sequences = []
                    keep_accessions = []
                    
                    matched_splits = []
                    train_keep_mask = []
                    for j, seq in enumerate(row['sequences']):
                        header = f"{os.path.basename(parquet_file)}|{fam_id}|{i}|{j}|{row['accessions'][j]}"
                        
                        if header in matching_headers:
                            split_matches = result_df[result_df["query_header"] == header].target_header.apply(lambda x: x.split("_")[0]).unique()
                            if len(split_matches) > 1:
                                matched_split = "both"
                            else:
                                matched_split = split_matches[0]
                            matched_splits.append(matched_split)
                            removed_from_train.append((header, matched_split))
                            train_keep_mask.append(False)
                        else:
                            keep_sequences.append(seq)
                            keep_accessions.append(row['accessions'][j])
                            train_keep_mask.append(True)
                    # If there are sequences to keep, update the row
                    if len(keep_sequences) > 0:
                        updated_row = {}
                        for k,v in row.items():
                            if isinstance(v, list) or isinstance(v, np.ndarray):
                                assert len(v) == len(row['sequences']) or len(v) == 0
                                if len(v) == 0:
                                    updated_row[k] = v
                                else:
                                    updated_row[k] = np.array(v)[train_keep_mask]
                            else:
                                updated_row[k] = v
                        rows_to_keep.append(updated_row)
                    # If all sequences are filtered out, move the row to val or test
                    elif matched_splits and len(set(matched_splits)) == 1:
                        if matched_splits[0] == "val":
                            val_rows.append(row)
                        elif matched_splits[0] == "test":
                            test_rows.append(row)
                
                # Create new dataframe with filtered rows
                if rows_to_keep:
                    # Convert the list of Series into a DataFrame with the same structure as the original
                    filtered_df = pd.DataFrame(rows_to_keep)
                    print(f"kept {round(len(filtered_df) / len(df) * 100, 2)}% of families from {os.path.basename(parquet_file)}")
                    
                    for col in filtered_df.columns:
                        if col in df.columns:
                            filtered_df[col] = filtered_df[col].astype(df[col].dtype)
                    
                    
                    filtered_df.to_parquet(train_output_path, index=False)
                    print(f"Saved filtered training data to {train_output_path} ({len(filtered_df)} rows)")
                else:
                    print(f"All rows were filtered out from {os.path.basename(parquet_file)}")
                
                # Save val and test rows if any
                if val_rows:
                    val_df = pd.DataFrame(val_rows)
                    val_output_path = os.path.join(dataset["output_dir"], "val_filtered", f"val_{os.path.basename(parquet_file)}")
                    val_df.to_parquet(val_output_path, index=False)
                    print(f"Saved validation data to {val_output_path} ({len(val_df)} rows)")
                
                if test_rows:
                    test_df = pd.DataFrame(test_rows)
                    test_output_path = os.path.join(dataset["output_dir"], "test_filtered", f"test_{os.path.basename(parquet_file)}")
                    test_df.to_parquet(test_output_path, index=False)
                    print(f"Saved test data to {test_output_path} ({len(test_df)} rows)")
                
            finally:
                # Clean up worker-local temporary files
                shutil.rmtree(tmp_dir, ignore_errors=True)
        removed_csv_path = os.path.join(
            dataset["output_dir"],
            f"removed_from_train_{unique_suffix}.csv"
        )
        with open(removed_csv_path, "w") as f:
            for header, split in removed_from_train:
                f.write(f"{header},{split}\n")
        print(f"Completed processing dataset: {dataset['name']}")
        print(f"Removed {len(removed_from_train)} sequences from training set")
    
    # Clean up worker-local temporary files
    shutil.rmtree(tmp_base_dir, ignore_errors=True)
    print("Completed removing similar sequences from training sets")


def main():
    parser = argparse.ArgumentParser(description="Create train/test split using MMSEQS filtering")
    parser.add_argument("--task_index", type=int, help="Index of the dataset to process (from datasets_to_filter list)", required=False)
    parser.add_argument("--scratch_dir", type=str, help="Directory to use for temporary MMSeqs files and databases", required=False)
    args = parser.parse_args()
    
    global SCRATCH_DIR
    if args.scratch_dir:
        SCRATCH_DIR = args.scratch_dir
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        print(f"Using scratch directory: {SCRATCH_DIR}")

    os.makedirs("data", exist_ok=True)
    rep_fasta_path = "data/val_test_heldout_representative_sequences.fasta"
    if not os.path.exists(rep_fasta_path):
        select_heldout_families(rep_fasta_path)
    assert os.path.exists(rep_fasta_path)
    
    # Remove sequences similar to held-out sequences from training sets
    if args.task_index is not None:
        # Process only the specified dataset
        if 0 <= args.task_index <= max(max(ds['task_indices']) for ds in datasets_to_filter):
            for ds in datasets_to_filter:
                if args.task_index in ds['task_indices']:
                    dataset = ds
                    break
            print(f"Processing only dataset at index {args.task_index}: {dataset['name']}")
            remove_similar_sequences_from_train_set(rep_fasta_path, [dataset], task_index=args.task_index)
        else:
            print(f"Error: task_index {args.task_index} is out of range. Valid range: 0-{len(datasets_to_filter)-1}")
            print(f"Available datasets:")
            for i, dataset in enumerate(datasets_to_filter):
                print(f"  {i}: {dataset['name']}")
    else:
        # Process all datasets
        remove_similar_sequences_from_train_set(rep_fasta_path, datasets_to_filter)


if __name__ == "__main__":
    main()



