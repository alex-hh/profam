# data_creation_scripts/create_allocation.py
import argparse
import json
from pathlib import Path
import random

def create_allocation(chunk_dir, output_path, num_recombinant_files):
    chunk_dir = Path(chunk_dir)
    manifests = list(chunk_dir.glob("manifest_*.json"))
    all_chunks = []
    for m in manifests:
        with m.open() as f:
            all_chunks.extend(json.load(f))
    
    random.shuffle(all_chunks)

    allocation = {i: [] for i in range(num_recombinant_files)}
    for i, chunk in enumerate(all_chunks):
        allocation[i % num_recombinant_files].append(chunk["path"])
    
    with open(output_path, "w") as f:
        json.dump(allocation, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_dir")
    parser.add_argument("output_file")
    parser.add_argument("num_recombinant_files", type=int)
    args = parser.parse_args()
    
    create_allocation(args.chunk_dir, args.output_file, args.num_recombinant_files)