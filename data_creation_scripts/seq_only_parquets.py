"""
Iterates over parquet files and creates new parquet files where there is none of the 
structural data. Columns to keep:
- "sequences"
- "fam_id"
- "accessions"
- "af50_cluster_id"

"""

import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="../data/afdb_s50_single/train_val_test_split",
        help="Path to the directory containing input parquet files."
    )
    parser.add_argument(
        "--new_parquet_dir",
        type=str,
        default="../data/afdb_s50_single_seq_only/train_val_test_split",
        help="Path to the directory where output parquet files will be saved."
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of the current parallel task."
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1,
        help="Total number of parallel tasks."
    )

    args = parser.parse_args()

    os.makedirs(args.new_parquet_dir, exist_ok=True)

    for split_dir in ["train", "val", "test"]:
        split_input_dir = os.path.join(args.parquet_dir, split_dir)
        split_output_dir = os.path.join(args.new_parquet_dir, split_dir)
        os.makedirs(split_output_dir, exist_ok=True)

        files = sorted(os.listdir(split_input_dir))

        # Divide the list of files among the parallel tasks
        total_files = len(files)
        chunk_size = ((total_files + args.num_tasks - 1) // args.num_tasks) + 1
        start_index = args.task_index * chunk_size
        end_index = min(start_index + chunk_size, total_files)

        for file in files[start_index:end_index]:
            in_path = os.path.join(split_input_dir, file)
            out_path = os.path.join(split_output_dir, file)

            df = pd.read_parquet(in_path)
            df = df[["sequences", "fam_id", "accessions", "af50_cluster_id"]]
            df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()


