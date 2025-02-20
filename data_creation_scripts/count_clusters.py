#!/usr/bin/env python3

import argparse
import glob
import os
import pandas as pd
from tqdm import tqdm

def process_parquet_file(parquet_path, output_dir, task_index):
    df = pd.read_parquet(parquet_path)
    
    # Extract identity thresholds from column names
    threshold_cols = [col for col in df.columns if col.startswith('cluster_ids_')]
    
    # Create DataFrame to store counts
    counts_df = pd.DataFrame({
        'parquet_file': [os.path.basename(parquet_path)] * len(df),
        'fam_id': df['fam_id']
    })
    
    # Calculate cluster counts for each threshold
    for col in threshold_cols:
        counts_df[col.replace('cluster_ids_', 'count_')] = df[col].apply(lambda x: len(set(x)) if x is not None else 0)
    
    # Save counts to CSV
    output_path = os.path.join(output_dir, f"cluster_counts_{task_index}.csv")
    print(f"Saving counts to {output_path}")    
    
    counts_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

def main():
    parser = argparse.ArgumentParser(description="Count clusters in clustered parquet files")
    parser.add_argument("--input_pattern", default="../data/ted/s100_parquets/train_val_test_split_hq/train_val_test_split/*/clustered/*.parquet",
                        help="Pattern to match parquet files")
    parser.add_argument("--task_index", type=int, required=True,
                        help="Index of the task to run")
    parser.add_argument("--num_tasks", type=int, required=True,
                        help="Number of parallel tasks")
    args = parser.parse_args()

    # Get all parquet files and split between tasks
    parquet_files = glob.glob(args.input_pattern)
    batch_size = len(parquet_files) // args.num_tasks + 1
    start_idx = args.task_index * batch_size
    end_idx = min(start_idx + batch_size, len(parquet_files))
    task_files = parquet_files[start_idx:end_idx]

    # Create output directory
    output_dir = os.path.join(os.path.dirname(parquet_files[0]), "../../cluster_counts")
    os.makedirs(output_dir, exist_ok=True)

    # Process files
    for parquet_path in task_files:
        process_parquet_file(parquet_path, output_dir, args.task_index)
    print(f"Finished task {args.task_index} of {args.num_tasks}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 