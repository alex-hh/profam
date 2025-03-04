# data_creation_scripts/recombine.py
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
import sys

def process_recombinant(allocation, output_path):
    # Create writer to accumulate chunks
    writer = None
    schema = None
    
    for chunk_path in allocation:
        table = pq.read_table(chunk_path)
        if not schema:
            schema = table.schema
            writer = pq.ParquetWriter(output_path, schema)
        
        writer.write_table(table)
    
    if writer:
        writer.close()

def final_shuffle(parquet_path):
    df = pd.read_parquet(parquet_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_parquet(parquet_path, index=False)

def main(allocation_file, output_dir, task_index, num_tasks):
    os.makedirs(output_dir, exist_ok=True)
    with open(allocation_file) as f:
        full_allocation = json.load(f)
    
    # Split recombinant files across tasks
    recombinant_ids = sorted(full_allocation.keys())
    batch_size = (len(recombinant_ids) // num_tasks) + 1
    start_index = task_index * batch_size
    our_ids = recombinant_ids[start_index:start_index + batch_size]
    
    for r_id in our_ids:
        output_path = Path(output_dir) / f"recombinant_{r_id}.parquet"
        process_recombinant(full_allocation[str(r_id)], output_path)
        final_shuffle(output_path)

if __name__ == "__main__":
    # Usage: python recombine.py allocation.json output_dir --task_index 0 --num_tasks 4
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("allocation_file")
    parser.add_argument("output_dir")
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--num_tasks", type=int, required=True)
    args = parser.parse_args()
    
    main(args.allocation_file, args.output_dir, args.task_index, args.num_tasks)