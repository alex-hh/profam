"""
Created by Jude Wells 2025-05-05

Applies the mmseqs train test split to the redundancy reduced datasets.

1) iterate over parquets in filtered_parquet_dir
2) create a dict where key is fam_id then has nested dict with keys train, val, test and each accession in fam assigned to one of the three keys
3) iterate over parquets in parquet_pattern
4) for row in each parquet filter it into train / val / test / test
5) use ParquetBufferWriter class for each of train, val, test and write new splits to disk
"""
import shutil
import os
import sys
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

from data_creation_scripts.val_test_split.parquet_buffer_writer import ParquetBufferWriter


np.random.seed(42)

datasets_to_filter = [
    {
        "name": "FunFamsS50",
        "parquet_pattern": "../data/funfams/s50_parquets/*.parquet",
        "filtered_parquet_dir": "../data/funfams/s100_noali_parquets/train_test_split_v2",
        "output_dir": "../data/funfams/s50_parquets/train_test_split_v2"
    },
    {
        "name": "foldseek_s50_seq_only",
        "parquet_pattern": "../data/foldseek/foldseek_s50_seq_only/train_val_test_split/*/*.parquet",
        "filtered_parquet_dir": "../data/foldseek/foldseek_s100_raw/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_s50_seq_only/train_test_split_v2"
    },
    {
        "name": "foldseek_s50_struct",
        "parquet_pattern": "../data/foldseek/foldseek_s50_struct/*.parquet",
        "filtered_parquet_dir": "../data/foldseek/foldseek_s100_raw/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_s50_struct/train_val_test_split_v2/"
    },
    {
        "name": "foldseek_s100_struct",
        "parquet_pattern": "../data/foldseek/foldseek_s100_struct/*.parquet",
        "filtered_parquet_dir": "../data/foldseek/foldseek_s100_raw/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_s100_struct/train_test_split_v2",
    },
    {
        "name": "foldseek_reps_single",
        "parquet_pattern": "../data/foldseek/foldseek_reps_single/*.parquet",
        "filtered_parquet_dir": "../data/foldseek/foldseek_s100_raw/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_reps_single/train_test_split_v2",
    },
    # {
    #     "name": "afdb_s50_single",
    #     "parquet_pattern": "../data/afdb_s50_single/*.parquet",
    #     "filtered_parquet_dir": "../data/afdb/afdb_s50_single_parquets/train_test_split_v2",
    #     "output_dir": "../data/afdb/afdb_s50_single_parquets/train_test_split_v2",
    # },
    {
        "name": "ted_s50_hq",
        "parquet_pattern": "../data/ted/s50_parquets/train_val_test_split_hq/train_val_test_split/*/*.parquet",
        "filtered_parquet_dir": "../data/ted/s100_parquets/train_test_split_v2",
        "output_dir": "../data/ted/s50_parquets/train_test_split_v2"
    }
]

def build_splits_json(filtered_parquet_dir):
    splits_json = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(filtered_parquet_dir, f"{split}_filtered")
        if not os.path.exists(split_dir):
            raise ValueError(f"Split {split} does not exist in {filtered_parquet_dir}")
            
        for parquet in glob.glob(os.path.join(split_dir, "*.parquet")):
            df = pd.read_parquet(parquet)
            for _, row in df.iterrows():
                fam_id = row["fam_id"].replace("_rep_seq", "").replace("_ted.fasta", "")
                accessions = row["accessions"] if "accessions" in row else row.get("accession", [])
                if fam_id not in splits_json:
                    splits_json[fam_id] = {
                        "train": [],
                        "val": [],
                        "test": []
                    }
                splits_json[fam_id][split].extend(accessions)
    return splits_json

def dataset_has_structure(dataset_info):
    path = glob.glob(os.path.join(dataset_info["filtered_parquet_dir"], "*/*.parquet"))[0]
    df = pd.read_parquet(path)
    if "C" in df.columns or "CA" in df.columns:
        return True
    return False

def filter_dataset(dataset_info):
    split_json = build_splits_json(dataset_info["filtered_parquet_dir"])
    output_dir = dataset_info["output_dir"]
    # Prepare output directories and writers
    os.makedirs(os.path.join(output_dir, "train_filtered"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val_filtered"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test_filtered"), exist_ok=True)
    mem_limit = 50 if dataset_has_structure(dataset_info) else 250
    parquet_writers = {
        "train": ParquetBufferWriter(os.path.join(output_dir, "train_filtered"), name="train", mem_limit=mem_limit),
        "val": ParquetBufferWriter(os.path.join(output_dir, "val_filtered"), name="val", mem_limit=mem_limit),
        "test": ParquetBufferWriter(os.path.join(output_dir, "test_filtered"), name="test", mem_limit=mem_limit),
    }

    parquet_files = glob.glob(dataset_info["parquet_pattern"])
    print(f"Processing {dataset_info['name']} – found {len(parquet_files)} parquet files")
    for parquet_path in tqdm.tqdm(parquet_files, desc=f"{dataset_info['name']}"):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            process_row(row, split_json, parquet_writers)
    # Flush remaining buffers
    for writer in parquet_writers.values():
        writer.write_dfs()


def process_row(row, split_json, parquet_writers):
    """Process a single parquet row, routing sub-rows to the correct writer."""
    fam_id = row["fam_id"].replace("_rep_seq", "").replace("_ted.fasta", "")
    accessions_list = row["accessions"]

    if fam_id not in split_json:
        print(f"Fam {fam_id} not in split_json, keys in split_json: {len(split_json.keys())}")
        print(f"Sample of keys in split_json: {random.sample(list(split_json.keys()), 10)}")
        return

    # Otherwise split by accession membership
    accession_to_split = {}
    for split in ["train", "val", "test"]:
        for acc in split_json[fam_id][split]:
            accession_to_split[acc] = split

    # Build masks per split
    splits_masks = {s: [] for s in ["train", "val", "test"]}
    for acc in accessions_list:
        s = accession_to_split.get(acc, "train")  # default to train
        for split in splits_masks:
            splits_masks[split].append(s == split)
    for split, mask in splits_masks.items():
        mask_np = np.array(mask, dtype=bool)  # Use mask_np to avoid conflict
        if mask_np.any():
            new_row = {}
            for k, v_original in row.items(): # Renamed v to v_original for clarity
                if isinstance(v_original, (np.ndarray, list)):
                    # Ensure proper handling of lists vs ndarrays and apply mask
                    v_as_array = np.array(v_original, copy=False) # Use copy=False if already ndarray and type is fine
                    
                    masked_data = v_as_array[mask_np]

                    # Standardize to np.float32 for specific coordinate/numerical columns
                    # to prevent errors from mixed float types (e.g., float16 and float32)
                    # which can lead to object arrays that pyarrow cannot convert directly.
                    if k in ['N', 'CA', 'C', 'O'] and masked_data.size > 0:
                        if masked_data.dtype == np.object_ or np.issubdtype(masked_data.dtype, np.floating):
                            try:
                                masked_data = masked_data.astype(np.float16)
                            except ValueError:
                                # This might occur if an object array contains non-numeric data.
                                # The original error suggests a mix of floats, so this is a safeguard.
                                print(f"Warning: Could not cast column '{k}' to np.float16 for fam_id {fam_id} in split {split}. Original dtype: {masked_data.dtype}. Skipping row.")
                                return
                    
                    new_row[k] = masked_data
                else:
                    new_row[k] = v_original # Assign scalars or other non-array types directly
            
            try:
                # Create DataFrame from the processed row
                df_to_add_to_buffer = pd.DataFrame([new_row])
                parquet_writers[split].update_buffer(df_to_add_to_buffer)
            except Exception as e_buffer:
                print(f"Error creating DataFrame or updating buffer for fam_id {fam_id}, split {split}. Error: {e_buffer}")
                # Detailed logging for problematic row structure
                print(f"Problematic new_row details for fam_id {fam_id}, split {split}:")
                for item_k, item_v in new_row.items():
                    if isinstance(item_v, np.ndarray):
                        print(f"  Item '{item_k}': type={type(item_v)}, dtype={item_v.dtype}, shape={item_v.shape}")
                    else:
                        print(f"  Item '{item_k}': type={type(item_v)}, value (sample if long): {str(item_v)[:100]}")
                # Optionally, re-raise or implement more specific error handling if needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=-1)
    args = parser.parse_args()
    if args.task_index != -1:
        datasets_to_filter = [datasets_to_filter[args.task_index]]
    for dataset_info in datasets_to_filter:
        filter_dataset(dataset_info)