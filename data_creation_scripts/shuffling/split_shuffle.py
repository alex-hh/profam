# data_creation_scripts/split_shuffle.py
import os
import sys
import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def process_file(input_path, output_dir, chunks_per_file=10):
    """Splits one input file into shuffled chunks"""
    # Read entire file (consider pyarrow for chunked reading if memory constrained)
    df = pd.read_parquet(input_path)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    # Split into chunks
    chunk_size = len(df) // chunks_per_file + 1
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i+chunk_size]
        chunk_path = Path(output_dir) / f"{input_path.stem}_chunk{i}.parquet"
        chunk.to_parquet(chunk_path, index=False)
        chunks.append({
            "path": str(chunk_path),
            "num_rows": len(chunk),
            "source_file": input_path.name
        })
    return chunks

def main(input_dir, output_dir, task_index, num_tasks, chunks_per_file):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_files = sorted([f for f in input_dir.glob("*.parquet")])
    # Split files across tasks
    batch_size = (len(all_files) // num_tasks) + 1
    start_index = task_index * batch_size
    our_files = all_files[start_index:start_index + batch_size]
    
    manifest = []
    for f in tqdm(our_files):
        manifest.extend(process_file(f, output_dir, chunks_per_file))
        
    # Write task-specific manifest
    with (output_dir / f"manifest_{task_index}.json").open("w") as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    # Usage: python split_shuffle.py input_dir output_dir --task_index 0 --num_tasks 4
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--num_tasks", type=int, required=True)
    parser.add_argument("--chunks_per_file", type=int, default=10)
    args = parser.parse_args()
    
    main(
        args.input_dir, 
        args.output_dir, 
        args.task_index, 
        args.num_tasks, 
        args.chunks_per_file
        )