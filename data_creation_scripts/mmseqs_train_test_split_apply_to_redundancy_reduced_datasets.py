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
        for parquet in glob.glob(os.path.join(filtered_parquet_dir, f"{split}_filtered/*.parquet")):
            df = pd.read_parquet(parquet)
            for _, row in df.iterrows():
                fam_id = row["fam_id"]
                accessions = row["accessions"] if "accessions" in row else row.get("accession", [])
                if fam_id not in splits_json:
                    splits_json[fam_id] = {
                        "train": [],
                        "val": [],
                        "test": []
                    }
                splits_json[fam_id][split].extend(accessions)
    return splits_json


def filter_dataset(dataset_info):
    split_json = build_splits_json(dataset_info["filtered_parquet_dir"])
    output_dir = dataset_info["output_dir"]
    # Prepare output directories and writers
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    parquet_writers = {
        "train": ParquetBufferWriter(os.path.join(output_dir, "train"), name="train", mem_limit=250),
        "val": ParquetBufferWriter(os.path.join(output_dir, "val"), name="val", mem_limit=250),
        "test": ParquetBufferWriter(os.path.join(output_dir, "test"), name="test", mem_limit=250),
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
    fam_id = row["fam_id"].replace("_rep_seq", "")
    accessions_list = row["accessions"]

    if fam_id not in split_json:
        print(f"Fam {fam_id} not in split_json")
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
        mask = np.array(mask, dtype=bool)
        if mask.any():
            new_row = {}
            for k, v in row.items():
                if isinstance(v, (np.ndarray, list)):
                    new_row[k] = np.array(v)[mask]
                else:
                    new_row[k] = v
            parquet_writers[split].update_buffer(pd.DataFrame([new_row]))

    # # Optional assign residual families to train
    # # Handle any remaining accessions not assigned to val / test (default to train)
    # if fam_id not in split_json:
    #     parquet_writers["train"].update_buffer(pd.DataFrame([row]))
    # else:
    #     all_assigned = set(split_json[fam_id]["train"]) | set(split_json[fam_id]["val"]) | set(split_json[fam_id]["test"])
    #     residual_mask = np.array([acc not in all_assigned for acc in accessions_list])
    #     if residual_mask.any():
    #         residual_row = {}
    #         for k, v in row.items():
    #             if isinstance(v, (np.ndarray, list)):
    #                 residual_row[k] = np.array(v)[residual_mask]
    #             else:
    #                 residual_row[k] = v
    #         parquet_writers["train"].update_buffer(pd.DataFrame([residual_row]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=-1)
    args = parser.parse_args()
    if args.task_index != -1:
        datasets_to_filter = [datasets_to_filter[args.task_index]]
    for dataset_info in datasets_to_filter:
        filter_dataset(dataset_info)