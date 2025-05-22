import glob
import os
import pandas as pd
import random
import argparse
"""
Created by Jude Wells 2025-05-21
aggregates or splits parquet files so that
each file is a reasonable size.

"""

def combine_parquet_files_in_directory(parquet_path_list, max_residue_per_file=3_000_000):
    print(f"Combining {len(parquet_path_list)} parquet files")
    combined_rows = []
    residue_count = 0
    parquet_index = 0
    random.shuffle(parquet_path_list)
    save_dir = os.path.dirname(parquet_path_list[0])
    original_row_counter = 0
    combined_row_counter = 0
    for parquet_path in parquet_path_list:
        df = pd.read_parquet(parquet_path)
        assert os.path.dirname(parquet_path) == save_dir, "All parquet files must be in the same directory"
        for _, row in df.iterrows():
            original_row_counter += 1
            row_residue_count = sum([len(seq) for seq in row["sequences"]])
            if residue_count + row_residue_count > max_residue_per_file:
                combined_df = pd.DataFrame(combined_rows)
                combined_df = combined_df.sample(frac=1, replace=False).reset_index(drop=True)
                combined_df.to_parquet(f"{save_dir}/combined_parquet_{str(parquet_index).zfill(4)}.parquet", index=False)
                combined_rows = []
                residue_count = 0
                parquet_index += 1
                print(f"Combined {combined_row_counter} rows into {parquet_index} parquet files")
            combined_rows.append(row.to_dict())
            residue_count += row_residue_count
            combined_row_counter += 1
    if len(combined_rows) > 0:  
        combined_df = pd.DataFrame(combined_rows)
        combined_df = combined_df.sample(frac=1, replace=False).reset_index(drop=True)
        combined_df.to_parquet(f"{save_dir}/combined_parquet_{str(parquet_index).zfill(4)}.parquet", index=False)
        parquet_index += 1
        print(f"Combined {combined_row_counter} rows into {parquet_index} parquet files")
    if original_row_counter != combined_row_counter:
        raise ValueError(f"Original row counter {original_row_counter} does not match combined row counter {combined_row_counter}")
    else:
        # remove all original parquet files
        for parquet_path in parquet_path_list:
            os.remove(parquet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_dir", 
        type=str, 
        default="../data/uniref/uniref90_parquets_shuffled/train_test_split_v2/val_filtered/"
    )
    parser.add_argument(
        "--max_residue_per_file", 
        type=int, 
        default=3_000_000
    )
    args = parser.parse_args()
    print(f"Combining parquet files in {args.parquet_dir}")
    parquet_path_pattern = os.path.join(args.parquet_dir, "*.parquet")
    parquet_path_list = glob.glob(parquet_path_pattern)
    assert len(parquet_path_list) > 0, "No parquet files found"
    print(f"Found {len(parquet_path_list)} parquet files")
    combine_parquet_files_in_directory(parquet_path_list, args.max_residue_per_file)