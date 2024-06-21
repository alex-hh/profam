import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
import math


def process_document(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # Add any preprocessing steps here
        return {'content': content, 'file_name': os.path.basename(file_path)}
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def get_all_filepaths():
    with open("data/openfold_uniclust30_ids.txt", "r") as f:
        uniclust30_ids = f.read().split("\n")
    all_files = [f"../data/openfold/uniclust30_filtered/{file}/a3m/uniclust30.a3m" for file in uniclust30_ids]
    return all_files


def process_batch(args):
    file_batch, batch_id = args
    results = []
    for file_path in tqdm(file_batch, desc=f"Batch {batch_id}", leave=False):
        result = process_document(file_path)
        if result:
            results.append(result)

    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    output_file = f'{output_dir}/{batch_id}.parquet'
    pq.write_table(table, output_file)
    print(f"Saved batch {batch_id} to {output_file}")
    return output_file


def convert_to_arrow(batch_size=1000):
    all_files = get_all_filepaths()
    total_files = len(all_files)
    num_batches = math.ceil(total_files / batch_size)

    # Create batches of files
    file_batches = [all_files[i:i + batch_size] for i in range(0, total_files, batch_size)]

    # Process batches in parallel with progress bar
    with Pool(processes=6) as pool:
        batch_args = [(batch, i) for i, batch in enumerate(file_batches)]
        output_files = list(tqdm(pool.imap(process_batch, batch_args), total=num_batches, desc="Processing batches"))

    return output_files

output_dir = f"../data/openfold/uniclust30_filtered_parquet/"
os.makedirs(output_dir, exist_ok=True)
output_files = convert_to_arrow()
print(f"Created {len(output_files)} parquet files: {output_files}")