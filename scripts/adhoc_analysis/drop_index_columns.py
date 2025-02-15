import argparse
import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

# Datasets definition (same as original)
DATASETS = {
    1: {
        "name": "ted_s100",
        "parent_dir": "../data/ted/s100_parquets",
        "split_dir": "../data/ted/s100_parquets/train_val_test_split",
    },
    2: {
        "name": "ted_s50",
        "parent_dir": "../data/ted/s50_parquets",
        "split_dir": "../data/ted/s50_parquets/train_val_test_split",
    },
    3: {
        "name": "funfam_s100_noali",
        "parent_dir": "../data/funfams/s100_noali_parquets",
        "split_dir": "../data/funfams/s100_noali_parquets/train_val_test_split",
    },
    4: {
        "name": "funfam_s50",
        "parent_dir": "../data/funfams/s50_parquets",
        "split_dir": "../data/funfams/s50_parquets/train_val_test_split",
    },
    5: {
        "name": "foldseek_s100_raw",
        "parent_dir": "../data/foldseek/foldseek_s100_raw",
        "split_dir": "../data/foldseek/foldseek_s100_raw/train_val_test_split",
    },
    6: {
        "name": "afdb_s50_single",
        "parent_dir": "../data/afdb_s50_single",
        "split_dir": "../data/afdb_s50_single/train_val_test_split",
    },
    7: {
        "name": "foldseek_s100_struct",
        "parent_dir": "../data/foldseek/foldseek_s100_struct",
        "split_dir": "../data/foldseek/foldseek_s100_struct/train_val_test_split",
    },
    8: {
        "name": "foldseek_reps_single",
        "parent_dir": "../data/foldseek/foldseek_reps_single",
        "split_dir": "../data/foldseek/foldseek_reps_single/train_val_test_split",
    },
    9: {
        "name": "foldseek_s50_struct",
        "parent_dir": "../data/foldseek/foldseek_s50_struct",
        "split_dir": "../data/foldseek/foldseek_s50_struct/train_val_test_split",
    },
    10: {
        "name": "afdb_s50_single_seq_only",
        "parent_dir": "../data/afdb_s50_single",
        "split_dir": "../data/afdb_s50_single_seq_only/afdb_s50_single/train_val_test_split",
    },
}

def remove_index_level_columns(parquet_path):
    """Remove columns with 'index_level' in their name from a parquet file"""
    try:
        df = pd.read_parquet(parquet_path)
        index_level_cols = [col for col in df.columns if 'index_level' in col]
        
        if not index_level_cols:
            return {'file': parquet_path, 'columns_removed': 0, 'status': 'no_action'}
            
        df = df.drop(columns=index_level_cols)
        df.to_parquet(parquet_path, index=False)
        return {'file': parquet_path, 'columns_removed': len(index_level_cols), 'status': 'updated'}
        
    except Exception as e:
        logging.error(f"Error processing {parquet_path}: {str(e)}")
        return {'file': parquet_path, 'columns_removed': 0, 'status': 'error'}

def process_dataset(dataset_id, task_index=0, num_tasks=1):
    """Process a subset of parquet files in a dataset based on task index"""
    if dataset_id not in DATASETS:
        logging.error(f"Invalid dataset ID: {dataset_id}")
        return

    dataset = DATASETS[dataset_id]
    logging.info(f"Processing dataset {dataset_id}: {dataset['name']}")
    
    # Get and sort all parquet files
    parquet_files = []
    parquet_files += glob.glob(os.path.join(dataset['parent_dir'], '*.parquet'))
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset['split_dir'], split)
        parquet_files += glob.glob(os.path.join(split_path, '*.parquet'))
    
    # Sort files for consistent ordering
    parquet_files = sorted(parquet_files)
    
    # Split files into batches
    batch_size = len(parquet_files) // num_tasks
    start = task_index * batch_size
    end = start + batch_size if task_index < num_tasks - 1 else len(parquet_files)
    batch_files = parquet_files[start:end]

    # Process files sequentially
    results = []
    for pf in tqdm(batch_files, desc=f"Processing {dataset['name']} (task {task_index})"):
        results.append(remove_index_level_columns(pf))

    # Print summary
    updated_files = sum(1 for r in results if r['status'] == 'updated')
    total_columns_removed = sum(r['columns_removed'] for r in results)
    
    logging.info(f"""
    Processed {len(parquet_files)} files
    - Updated {updated_files} files
    - Removed {total_columns_removed} index_level columns
    - {sum(1 for r in results if r['status'] == 'error')} errors
    """)

def main():
    parser = argparse.ArgumentParser(description='Remove index_level columns from parquet files')
    parser.add_argument('--task_index', type=int, default=0,
                      help='Task index for parallel processing')
    parser.add_argument('--num_tasks', type=int, default=1,
                      help='Total number of parallel tasks')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.task_index == 0:
        for dataset_id in DATASETS:
            process_dataset(dataset_id, args.task_index, args.num_tasks)
    else:
        process_dataset(args.task_index, args.task_index, args.num_tasks)

if __name__ == '__main__':
    main()