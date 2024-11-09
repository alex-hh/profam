import pandas as pd
import numpy as np
from Bio import SeqIO
import gzip
import os
from tqdm import tqdm
import random
from itertools import islice

def chunk_iterator(fasta_file, chunk_size=10000):
    """
    Memory-efficient iterator for FASTA files that yields chunks of records
    """
    accessions = []
    sequences = []
    
    with gzip.open(fasta_file, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            uniprot_id = record.id.split('|')[1] if '|' in record.id else record.id
            accessions.append(np.array([uniprot_id]))
            sequences.append(np.array([str(record.seq)]))
            
            if len(accessions) >= chunk_size:
                yield accessions, sequences
                accessions = []
                sequences = []
    
    if accessions:  # Return any remaining records
        yield accessions, sequences

def count_records(fasta_file):
    """Count total records in a FASTA file"""
    count = 0
    with gzip.open(fasta_file, 'rt') as handle:
        for _ in SeqIO.parse(handle, 'fasta'):
            count += 1
    return count

def create_ordered_parquet_files(input1, input2, output_dir, num_files=512):
    """
    Create parquet files from FASTA files maintaining the original sequence order,
    processing in chunks to minimize memory usage
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total records to determine chunk size
    print("Counting records...")
    total_records = count_records(input1) + count_records(input2)
    records_per_file = (total_records + num_files - 1) // num_files
    chunk_size = min(records_per_file, 1000000)  # Process in chunks of 10k or one file worth
    
    current_file = 0
    records_in_current_file = 0
    current_chunk_acc = []
    current_chunk_seq = []
    
    def save_chunk(chunk_acc, chunk_seq, file_num):
        if not chunk_acc:
            return
        df = pd.DataFrame({
            'accessions': chunk_acc,
            'sequences': chunk_seq
        })
        output_file = os.path.join(output_dir, f'ordered_chunk_{file_num:03d}.parquet')
        df.to_parquet(output_file, index=False)
    
    # Process files sequentially
    for fasta_file in [input1, input2]:
        print(f"Processing {os.path.basename(fasta_file)}...")
        for chunk_acc, chunk_seq in chunk_iterator(fasta_file, chunk_size):
            for acc, seq in zip(chunk_acc, chunk_seq):
                current_chunk_acc.append(acc)
                current_chunk_seq.append(seq)
                records_in_current_file += 1
                
                if records_in_current_file >= records_per_file:
                    save_chunk(current_chunk_acc, current_chunk_seq, current_file)
                    current_file += 1
                    records_in_current_file = 0
                    current_chunk_acc = []
                    current_chunk_seq = []
    
    # Save any remaining records
    if current_chunk_acc:
        save_chunk(current_chunk_acc, current_chunk_seq, current_file)

def create_shuffled_parquet_files(input_dir, output_dir, num_files=512, chunk_size=10000):
    """
    Create shuffled parquet files without loading all data into memory at once
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First, create a list of all record locations
    print("Creating record mapping...")
    record_locations = []
    for file_idx, file in enumerate(sorted(os.listdir(input_dir))):
        if file.endswith('.parquet'):
            file_path = os.path.join(input_dir, file)
            num_records = pd.read_parquet(file_path, columns=[]).shape[0]
            record_locations.extend([(file_idx, i) for i in range(num_records)])
    
    # Shuffle the record locations
    print("Shuffling record locations...")
    random.seed(42)
    random.shuffle(record_locations)
    
    # Calculate records per output file
    total_records = len(record_locations)
    records_per_file = (total_records + num_files - 1) // num_files
    
    # Create a cache for file handles to avoid repeatedly opening/closing files
    file_cache = {}
    max_cache_size = 10  # Maximum number of files to keep open
    
    print("Creating shuffled files...")
    for output_idx in tqdm(range(num_files)):
        start_idx = output_idx * records_per_file
        end_idx = min((output_idx + 1) * records_per_file, total_records)
        
        # Process records for this output file
        output_records_acc = []
        output_records_seq = []
        
        # Group records by input file to minimize file operations
        file_groups = {}
        for i in range(start_idx, end_idx):
            file_idx, record_idx = record_locations[i]
            if file_idx not in file_groups:
                file_groups[file_idx] = []
            file_groups[file_idx].append(record_idx)
        
        # Process each input file's records
        for file_idx, record_indices in file_groups.items():
            input_file = os.path.join(input_dir, f'ordered_chunk_{file_idx:03d}.parquet')
            
            # Read only the required records from this file
            df = pd.read_parquet(input_file).iloc[record_indices]
            output_records_acc.extend(df['accessions'].tolist())
            output_records_seq.extend(df['sequences'].tolist())
        
        # Save the shuffled chunk
        output_df = pd.DataFrame({
            'accessions': output_records_acc,
            'sequences': output_records_seq
        })
        output_file = os.path.join(output_dir, f'shuffled_chunk_{output_idx:03d}.parquet')
        output_df.to_parquet(output_file, index=False)
        
        # Clear memory
        del output_df
        del output_records_acc
        del output_records_seq

if __name__ == "__main__":
    base_dir = "/SAN/orengolab/cath_plm/ProFam/data/uniprot"
    input1 = os.path.join(base_dir, "uniprot_sprot.fasta.gz")
    input2 = os.path.join(base_dir, "uniprot_trembl.fasta.gz")
    assert os.path.exists(input1), f"File {input1} does not exist"
    assert os.path.exists(input2), f"File {input2} does not exist"
    ordered_output_dir = os.path.join(base_dir, "ordered_parquet")
    shuffled_output_dir = os.path.join(base_dir, "shuffled_parquet")
    
    print("Creating ordered parquet files...")
    create_ordered_parquet_files(input1, input2, ordered_output_dir)
    
    print("Creating shuffled parquet files...")
    create_shuffled_parquet_files(ordered_output_dir, shuffled_output_dir)