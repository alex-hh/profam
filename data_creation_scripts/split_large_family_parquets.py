"""
Iterates through all parquet files in input_directory
Splits large sequence families into multiple files with ~200k sequences each
"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def split_large_families(input_pattern, threshold=500000):
    for parquet_file in glob.glob(input_pattern):
        print(f"Processing {parquet_file}")
        process_file(parquet_file, threshold)

def process_file(file_path, threshold):
    file_path = Path(file_path)
    df = pd.read_parquet(file_path)
    rows_to_keep = []
    rows_to_split = []

    # Separate rows that need splitting
    for _, row in df.iterrows():
        if len(row['sequences']) > threshold:
            rows_to_split.append(row)
        else:
            rows_to_keep.append(row)

    # Process large families
    for row in rows_to_split:
        n = len(row['sequences'])
        num_parts = (n // threshold) + 1
        indices = np.random.permutation(n)
        splits = np.array_split(indices, num_parts)

        base_name = file_path.stem
        output_dir = file_path.parent

        for i, split_indices in enumerate(splits):
            new_row = {
                col: (row[col][split_indices] if isinstance(row[col], (list, np.ndarray)) else row[col])
                for col in df.columns
            }
            part_df = pd.DataFrame([new_row])
            part_file = output_dir / f"{base_name}_part_{i}.parquet"
            part_df.to_parquet(part_file, index=False)

    # Only delete and resave if we actually split something
    if rows_to_split:
        file_path.unlink()
        if rows_to_keep:
            pd.DataFrame(rows_to_keep).to_parquet(file_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Split large sequence families in Parquet files')
    parser.add_argument('--input_pattern', help='Directory containing Parquet files to process',
                        default="../data/ted/s100_parquets/train_val_test_split_hq/train_val_test_split/*/*.parquet"
                        )
    parser.add_argument('--threshold', type=int, default=500000, 
                       help='Sequence count threshold for splitting (default: 500000)')
    args = parser.parse_args()
    split_large_families(args.input_pattern, args.threshold)

