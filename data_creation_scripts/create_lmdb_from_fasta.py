#!/usr/bin/env python3

import os
import logging
import argparse
import lmdb
from tqdm import tqdm
import mmap
import time

"""
Creates or updates an LMDB from AFDB FASTA file (Total records: 214,683,829 | Total size: 93GB).

LMDB structure:
key: UniProt accession (e.g. "A0A1B2C3D4")
value: Protein sequence (e.g. "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF...")
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_chunk(chunk: list, txn: lmdb.Transaction) -> int:
    for key, value in chunk:
        txn.put(key, value)
    return len(chunk)

def extract_uniprot_accession(record_id: bytes) -> bytes:
    """Extract UniProt accession from the record ID."""
    return record_id.split(b' ')[0].split(b':')[1].split(b'-')[1]

def mmap_fasta_parser(mm, file_size):
    start = 0
    while start < file_size:
        end = mm.find(b'\n>', start)
        if end == -1:
            end = file_size
        header_end = mm.find(b'\n', start)
        header = mm[start+1:header_end]
        sequence = mm[header_end+1:end].replace(b'\n', b'')
        yield header, sequence
        start = end + 1

def get_shard_key(accession: bytes) -> str:
    """Determine the shard key based on the first character of the accession."""
    first_char = chr(accession[0])
    if first_char.isalpha():
        return first_char.upper()
    return '0'  # Use '0' for any non-alphabetic first character

def create_lmdb_from_fasta(fasta_file: str, lmdb_path: str, batch_size: int):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    os.makedirs(lmdb_path, exist_ok=True)

    start_time = time.time()

    # Create a dictionary to hold all shard environments
    shard_envs = {}

    with open(fasta_file, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        file_size = mm.size()
        
        with tqdm(total=214683829, desc="Processing records", unit=" records") as pbar:
            for header, sequence in mmap_fasta_parser(mm, file_size):
                accession = extract_uniprot_accession(header)
                shard_key = get_shard_key(accession)
                
                if shard_key not in shard_envs:
                    shard_path = os.path.join(lmdb_path, f"shard_{shard_key}")
                    shard_envs[shard_key] = lmdb.open(shard_path, map_size=1099511627776 // 27, writemap=True, max_dbs=1)
                
                env = shard_envs[shard_key]
                with env.begin(write=True) as txn:
                    txn.put(accession, sequence)
                
                pbar.update(1)

    # Close all shard environments
    for env in shard_envs.values():
        env.sync()
        env.close()

    end_time = time.time()
    logger.info(f"LMDB databases created successfully at {lmdb_path}")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create LMDB from FASTA file")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--lmdb", required=True, help="Path to output LMDB file")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    logger.info("Script started")
    create_lmdb_from_fasta(args.fasta, args.lmdb, args.batch_size)
    logger.info("Script completed")