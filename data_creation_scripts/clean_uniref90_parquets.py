#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Standard amino acids (20 standard + X for unknown)
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")

def clean_parquet_file(input_path):
    """
    Clean a parquet file by:
    1. Removing newline characters from sequences
    2. Replacing selenocysteine (U) with cysteine (C)
    3. Replacing pyrrolysine (O) with lysine (K)
    4. Replacing ambiguous amino acids: B (D/N) with D, Z (E/Q) with E
    5. Removing any other non-standard amino acids
    6. Checking for empty arrays
    7. Verifying that the length of sequences matches the length of accessions
    
    Args:
        input_path: Path to the input parquet file
    
    Returns:
        tuple: (cleaned_count, non_standard_count, empty_arrays, length_mismatch)
    """
    input_dir = os.path.dirname(input_path)
    output_dir = f"{input_dir}_cleaned"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{os.path.basename(input_path)}"
    
    # Set up logging for this file
    log_dir = f"{input_dir}_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{os.path.basename(input_path)}.log"
    file_logger = logging.getLogger(os.path.basename(input_path))
    
    if not file_logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_logger.addHandler(file_handler)
    
    # Read the parquet file
    df = pd.read_parquet(input_path)
    
    # Initialize counters
    cleaned_count = 0
    non_standard_count = 0
    empty_arrays = 0
    length_mismatch = 0
    
    # Check if lengths match
    if len(df['sequences']) != len(df['accessions']):
        length_mismatch += 1
        file_logger.warning(f"Length mismatch: sequences ({len(df['sequences'])}) != accessions ({len(df['accessions'])})")
    
    # Clean sequences and check for issues
    cleaned_sequences = []
    valid_indices = []
    
    for i, seq_array in enumerate(df['sequences']):
        # Check for empty arrays
        if len(seq_array) == 0:
            empty_arrays += 1
            file_logger.warning(f"Empty sequence array at index {i}")
            continue
        
        # Get the sequence (it's stored as a numpy array with a single element)
        seq = seq_array[0]
        
        # Check if it contains newlines
        if '\n' in seq or '\r' in seq:
            cleaned_count += 1
            # Clean the sequence
            seq = seq.replace('\n', '').replace('\r', '')
        
        # Replace selenocysteine (U) with cysteine (C) and pyrrolysine (O) with lysine (K)
        if 'U' in seq or 'O' in seq:
            old_seq = seq
            seq = seq.replace('U', 'C').replace('O', 'K')
            if old_seq != seq:
                non_standard_count += 1
                file_logger.info(f"Replaced U/O in sequence at index {i}: {old_seq[:20]}... -> {seq[:20]}...")
        
        # Replace ambiguous amino acids: B (D/N) with D, Z (E/Q) with E
        if 'B' in seq or 'Z' in seq:
            old_seq = seq
            seq = seq.replace('B', 'D').replace('Z', 'E')
            if old_seq != seq:
                non_standard_count += 1
                file_logger.info(f"Replaced ambiguous amino acids B/Z in sequence at index {i}: {old_seq[:20]}... -> {seq[:20]}...")
        
        # Check for other non-standard amino acids
        non_standard_chars = set(seq) - STANDARD_AMINO_ACIDS
        if non_standard_chars:
            old_seq = seq
            # Replace non-standard amino acids with 'X'
            for char in non_standard_chars:
                seq = seq.replace(char, 'X')
            non_standard_count += 1
            file_logger.warning(f"Replaced non-standard amino acids {non_standard_chars} with X at index {i}: {old_seq[:20]}... -> {seq[:20]}...")
        
        # Add the cleaned sequence to the list
        cleaned_sequences.append(np.array([seq]))
        valid_indices.append(i)
    
    # If we found issues, create a new DataFrame with only valid rows
    if empty_arrays > 0 or length_mismatch > 0:
        # Create a new DataFrame with only valid rows
        new_df = pd.DataFrame({
            'sequences': cleaned_sequences,
            'accessions': [df['accessions'][i] for i in valid_indices],
            'fam_id': [df['fam_id'][i] for i in valid_indices]
        })
        df = new_df
    else:
        # Just update the sequences
        df['sequences'] = cleaned_sequences
    
    # Save the cleaned DataFrame
    if cleaned_count > 0 or non_standard_count > 0 or empty_arrays > 0 or length_mismatch > 0:
        df.to_parquet(output_path, index=False)
        file_logger.info(f"Saved cleaned file to {output_path}")
    
    return cleaned_count, non_standard_count, empty_arrays, length_mismatch

def main(args):
    parquet_files = glob.glob(args.input_pattern)
    
    # Process each file
    total_files_cleaned = 0
    total_sequences_cleaned = 0
    total_non_standard = 0
    total_empty_arrays = 0
    total_length_mismatch = 0
    
    for file_path in tqdm(parquet_files, desc="Cleaning parquet files"):
        # Clean the file
        cleaned_count, non_standard_count, empty_arrays, length_mismatch = clean_parquet_file(file_path)
        
        if cleaned_count > 0 or non_standard_count > 0 or empty_arrays > 0 or length_mismatch > 0:
            total_files_cleaned += 1
            total_sequences_cleaned += cleaned_count
            total_non_standard += non_standard_count
            total_empty_arrays += empty_arrays
            total_length_mismatch += length_mismatch
            
            if args.verbose:
                print(f"File: {file_path}")
                print(f"  - Cleaned newlines: {cleaned_count} sequences")
                print(f"  - Non-standard amino acids: {non_standard_count} sequences")
                print(f"  - Empty arrays: {empty_arrays}")
                print(f"  - Length mismatches: {length_mismatch}")
    
    print(f"Cleaning complete!")
    print(f"Total files with issues: {total_files_cleaned}")
    print(f"Total sequences with newlines cleaned: {total_sequences_cleaned}")
    print(f"Total sequences with non-standard amino acids: {total_non_standard}")
    print(f"Total empty arrays removed: {total_empty_arrays}")
    print(f"Total files with length mismatches: {total_length_mismatch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean UniRef90 parquet files by removing newline characters, replacing non-standard amino acids, and checking for data integrity")
    
    # Input arguments
    parser.add_argument(
        '--input_pattern', 
        default="../data/uniref/uniref90_parquets_shuffled/*/*.parquet",
        help="Glob pattern for input parquet files"
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help="Print detailed information for each file"
    )
    
    args = parser.parse_args()
    main(args) 