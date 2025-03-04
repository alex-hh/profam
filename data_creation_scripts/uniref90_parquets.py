#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import math
from tqdm import tqdm
"""
Number of lines in uniref90.fasta:
1479892305
1_479_892_305
"""

def get_file_size(file_path):
    """Get the size of a file in bytes."""
    return os.path.getsize(file_path)


def count_sequences(file_path):
    """Count the number of sequences in a FASTA file by counting '>' characters."""
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def create_parquet_file(data, output_path):
    """Create a parquet file from a list of dictionaries."""
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)


def process_batch(fasta_file, start_pos, end_pos, batch_index, output_dir, rows_per_file=5000):
    """Process a batch of the FASTA file and create parquet files."""
    print(f"Processing batch {batch_index} from position {start_pos} to {end_pos}")
    
    # Seek to the start position
    with open(fasta_file, 'r') as f:
        f.seek(start_pos)
        
        # If we're not at the beginning of the file, read until we find the next header
        if start_pos > 0:
            line = f.readline()
            while not line.startswith('>') and f.tell() < end_pos:
                line = f.readline()
        
        # Process sequences until we reach the end position
        sequences = []
        accessions = []
        fam_ids = []
        
        current_header = None
        current_sequence = ""
        
        parquet_index = 0
        sequence_count = 0
        
        # Read until we reach the end position or EOF
        for line in f:
            if not line:  # Skip empty lines
                continue
                
            if line.startswith('>'):
                # Save the previous sequence if there was one
                if current_header is not None:
                    sequences.append(np.array([current_sequence]))
                    
                    # Extract accession from header (remove 'UniRef90_' prefix)
                    accession = current_header.split()[0].replace('UniRef90_', '')
                    accessions.append(np.array([accession]))
                    fam_ids.append(accession)
                    
                    sequence_count += 1
                    
                    # If we've reached the target number of sequences per file, save a parquet
                    if sequence_count % rows_per_file == 0:
                        data = {
                            'sequences': sequences,
                            'accessions': accessions,
                            'fam_id': fam_ids
                        }
                        
                        output_file = os.path.join(
                            output_dir, 
                            f'uniref90_batch_{batch_index:03d}_part_{parquet_index:05d}.parquet'
                        )
                        create_parquet_file(data, output_file)
                        print(f"Created parquet file {parquet_index}: {output_file} with {len(sequences)} sequences")
                        
                        # Reset for next batch
                        sequences = []
                        accessions = []
                        fam_ids = []
                        parquet_index += 1
                
                # Start a new sequence
                current_header = line[1:]  # Remove the '>' character
                current_sequence = ""
            else:
                # Append to the current sequence
                current_sequence += line.strip()
        
        # Don't forget the last sequence
        if current_header is not None:
            sequences.append(np.array([current_sequence]))
            accession = current_header.split()[0].replace('UniRef90_', '')
            accessions.append(np.array([accession]))
            fam_ids.append(accession)
        
        # Save any remaining sequences
        if sequences:
            data = {
                'sequences': sequences,
                'accessions': accessions,
                'fam_id': fam_ids
            }
            
            output_file = os.path.join(
                output_dir, 
                f'uniref90_batch_{batch_index:03d}_part_{parquet_index:05d}.parquet'
            )
            create_parquet_file(data, output_file)
            print(f"Created final parquet file: {output_file} with {len(sequences)} sequences")
    
    return sequence_count


def main(args):
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get file size
    file_size = get_file_size(args.fasta_file)
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")
    
    # Calculate batch size
    batch_size = file_size // args.num_tasks
    
    # Calculate start and end positions for this task
    start_pos = args.task_index * batch_size
    end_pos = file_size if args.task_index == args.num_tasks - 1 else (args.task_index + 1) * batch_size
    
    # Process the batch
    sequences_processed = process_batch(
        args.fasta_file, 
        start_pos, 
        end_pos, 
        args.task_index,
        args.output_dir,
        args.rows_per_file
    )
    
    print(f"Task {args.task_index} processed {sequences_processed} sequences")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parquet files from UniRef90 FASTA file")
    parser.add_argument('--task_index', type=int, default=0, help="Index of the current task")
    parser.add_argument('--num_tasks', type=int, default=1, help="Total number of tasks")
    parser.add_argument('--fasta_file', type=str, default="/mnt/disk2/cath_plm/data/uniref/uniref90.fasta", help="Path to the UniRef90 FASTA file")
    parser.add_argument('--output_dir', type=str, default="/mnt/disk2/cath_plm/data/uniref/uniref90_parquets", help="Output directory for parquet files")
    parser.add_argument('--rows_per_file', type=int, default=5000, help="Number of sequences per parquet file")
    
    args = parser.parse_args()
    main(args)