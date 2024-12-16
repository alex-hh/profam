"""
Created by Jude Wells 16-12-2024
purpose is to create parquet files which are
designed to test if models can learn repeated
sequences etc.
"""

import pandas as pd
import numpy as np
import os
import glob

def create_repeated_sequence_dataset(parquet_dir, output_dir):
    """

    """
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        parquet_path = glob.glob(os.path.join(parquet_dir, split, "*.parquet"))[0]
        df = pd.read_parquet(parquet_path)
        new_rows = []


if __name__ == "__main__":
    ted_parquet_dir = "..data/ted/s50_parquets/train_val_test_split"
    output_dir = "../data/artificial_testing_data/ted_s50_repeated_seqs"
    create_repeated_sequence_dataset(ted_parquet_dir, output_dir)