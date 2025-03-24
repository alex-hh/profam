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

Number of sequences in uniref90.fasta:
204_806_909 sequence

TAKES 30 MINUTES AS A SINGLE TASK
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


def process_fasta(fasta_file, output_dir, rows_per_file=5000):
    """Process the FASTA file and create parquet files."""
    
    # Seek to the start position
    with open(fasta_file, 'r') as f:

        sequences = []
        accessions = []
        fam_ids = []
        
        current_header = None
        current_sequence = ""
        
        parquet_index = 0
        sequence_count = 0
        for line in f:
            if not line:  # Skip empty lines
                continue
                
            if line.startswith('>'):
                if current_header is not None:
                    clean_sequence = current_sequence.replace('\n', '').replace('\r', '')
                    sequences.append(np.array([clean_sequence]))
                    
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
                            f'uniref90_part_{parquet_index:05d}.parquet'
                        )
                        create_parquet_file(data, output_file)
                        print(f"Created parquet file {parquet_index}: {output_file} with {len(sequences)} sequences")
            
                        sequences = []
                        accessions = []
                        fam_ids = []
                        parquet_index += 1

                current_header = line[1:]  # Remove the '>' character
                current_sequence = ""
            else:
                current_sequence += line.strip()
        
        # Don't forget the last sequence
        if current_header is not None:
            clean_sequence = current_sequence.replace('\n', '').replace('\r', '')
            sequences.append(np.array([clean_sequence]))
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
                f'uniref90_part_{parquet_index:05d}.parquet'
            )
            create_parquet_file(data, output_file)
            print(f"Created final parquet file: {output_file} with {len(sequences)} sequences")
    
    return sequence_count


def main(args):
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    sequences_processed = process_fasta(
        args.fasta_file, 
        args.output_dir,
        args.rows_per_file
    )
    
    print(f"processed {sequences_processed} sequences")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parquet files from UniRef90 FASTA file")
    parser.add_argument('--fasta_file', type=str, default="../data/uniref/uniref90.fasta", help="Path to the UniRef90 FASTA file")
    parser.add_argument('--output_dir', type=str, default="../data/uniref/uniref90_parquet_to_delete", help="Output directory for parquet files")
    parser.add_argument('--rows_per_file', type=int, default=100000, help="Number of sequences per parquet file")
    
    args = parser.parse_args()
    main(args)