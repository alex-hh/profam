import os
import argparse
import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.fasta import read_fasta
import sys


def create_parquet_file(data, output_path):
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)


def calculate_list_size(string_list):
    # Base memory usage for the list itself
    total_bytes = sys.getsizeof([])

    # Add memory for each string
    for s in string_list:
        # Add the size of the string object
        total_bytes += sys.getsizeof(s)

        # Add the size of the string's contents
        # (Each character in Python 3 typically uses 1-4 bytes depending on the character)
        total_bytes += len(s.encode('utf-8'))

    return total_bytes

def main(task_index, num_tasks, output_dir):
    parquet_compression_factor = 5
    target_parquet_size_mb = 100
    list_size_mb = target_parquet_size_mb * parquet_compression_factor
    base_dir = "/SAN/orengolab/cath_alphafold/funfams_scans_2024/funfams"
    os.makedirs(output_dir, exist_ok=True)

    all_families = sorted(glob.glob(f"{base_dir}/*/*.faa"))
    # shuffle families with seed so same across all subtasks
    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_families)
    families_per_task = (len(all_families) // num_tasks) + 1
    start_index = task_index * families_per_task
    end_index = start_index + families_per_task

    families_to_process = all_families[start_index:end_index]
    print(f"Task {task_index} processing {len(families_to_process)} families")
    for i in range(2):
        print(families_to_process[i])
    data = []

    parquet_size_mb = 0
    parquet_index = 0
    for family_path in families_to_process:
        fam_id = family_path.split('/')[-1].split('.faa')[0]
        print(f"Processing family {fam_id}")
        accessions, sequences = read_fasta(
                    family_path,
                    return_dict=False,
                    encoding=None,
                    keep_insertions=True,
                    keep_gaps=True,
                    to_upper=False,
                )

        data.append({
            'fam_id': fam_id,
            'sequences': sequences,
            'accessions': accessions
        })
        sequence_bytes = calculate_list_size(sequences)
        parquet_size_mb += sequence_bytes / 1024 / 1024
        if parquet_size_mb >= list_size_mb:
            output_file = os.path.join(output_dir, f'funfam_data_{str(task_index).zfill(2)}_{str(parquet_index).zfill(2)}.parquet')
            create_parquet_file(data, output_file)
            print(f"Created parquet file: {output_file}")
            data = []
            parquet_size_mb = 0
            parquet_index += 1
    if len(data):
        output_file = os.path.join(output_dir, f'funfam_data_{str(task_index).zfill(2)}_{str(parquet_index).zfill(2)}.parquet')
        create_parquet_file(data, output_file)
        print(f"Created parquet file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parquet files from FASTA files")
    parser.add_argument('--task_index', type=int, required=True, help="Index of the current task")
    parser.add_argument('--num_tasks', type=int, help="Total number of tasks")
    parser.add_argument(
        '--save_dir',
        type=str,
        help="Output directory for parquet files"
    )
    args = parser.parse_args()

    main(args.task_index, args.num_tasks, args.save_dir)
