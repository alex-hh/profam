import os
import sys
import glob
import random
import pandas as pd

def make_small_datasets(datasets: dict, n_rows: int = 5):
    save_dir = "../data/mini_datasets"
    os.makedirs(save_dir, exist_ok=True)
    for dataset_name, dataset_path in datasets.items():
        print(f"Processing {dataset_name}...")
        print(f"Dataset path: {dataset_path}")
        parquet_files = glob.glob(dataset_path)
        print(f"Found {len(parquet_files)} files")
        # select random file
        random_file = random.choice(parquet_files)
        df = pd.read_parquet(random_file)

        sample_df = df.sample(n=n_rows)
        sample_df.to_parquet(os.path.join(save_dir, f"{dataset_name}_mini.parquet"))


if __name__ == "__main__":
    datasets = {
        "ted_s50": "../data/ted/s50_parquets/train_val_test_split/train/*.parquet",
        "ted_s100": "../data/ted/s100_parquets/train_val_test_split/train/*.parquet",
        "funfams_s50": "../data/funfams/s50_parquets/train_val_test_split/train/*.parquet",
        "funfams_s100": "../data/funfams/s100_noali_parquets/train_val_test_split/train/*.parquet",
        "foldseek_s100_raw": "../data/foldseek/foldseek_s100_raw/train_val_test_split/train/*.parquet",
        "foldseek_s100_struct": "../data/foldseek/foldseek_s100_struct/*.parquet",
        "foldseek_reps_single": "../data/foldseek/foldseek_reps_single/train_val_test_split/train/*.parquet",
        "foldseek_s50_struct": "../data/foldseek/foldseek_s50_struct/train_val_test_split/train/*.parquet",
        "afdb_s50_single": "../data/afdb_s50_single/train_val_test_split/train/*.parquet"
    }
    make_small_datasets(datasets)
