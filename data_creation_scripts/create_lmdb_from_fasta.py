#!/usr/bin/env python3

import os
import logging
import argparse
from typing import List, Tuple
import lmdb
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import mmap

"""
Creates or updates an LMDB from AFDB FASTA file (Total records: 214,683,829 | Total size: 93GB).

LMDB structure:
key: UniProt accession (e.g. "A0A1B2C3D4")
value: Protein sequence (e.g. "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF...")

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_batch(batch: List[Tuple[bytes, bytes]], lmdb_path: str) -> int:
    env = lmdb.open(lmdb_path, map_size=1099511627776, writemap=True, max_dbs=1, sync=False)
    try:
        with env.begin(write=True) as txn:
            for key, value in batch:
                txn.put(key, value)
        return len(batch)
    except lmdb.MapFullError:
        logger.error("LMDB map full. Consider increasing map_size.")
        return 0
    finally:
        env.close()

def worker_function(batch: List[Tuple[bytes, bytes]], lmdb_path: str) -> int:
    return process_batch(batch, lmdb_path)

def extract_uniprot_accession(record_id: str) -> str:
    """Extract UniProt accession from the record ID."""
    return record_id.split()[0].split(':')[1].split('-')[1]

def mmap_fasta_parser(mm, file_size):
    start = 0
    while start < file_size:
        end = mm.find(b'\n>', start)
        if end == -1:
            end = file_size
        record = mm[start:end].decode('ascii')
        header, sequence = record.split('\n', 1)
        yield header[1:], sequence.replace('\n', '')
        start = end + 1

def create_lmdb_from_fasta(fasta_file: str, lmdb_path: str, batch_size: int, num_cpus: int):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    logger.info(f"Using {num_cpus} CPUs")

    with open(fasta_file, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        file_size = mm.size()
        with mp.Pool(num_cpus) as pool:
            process_func = partial(worker_function, lmdb_path=lmdb_path)

            with tqdm(total=214683829, desc="Processing records", unit=" records") as pbar:
                batch = []
                for header, sequence in mmap_fasta_parser(mm, file_size):
                    batch.append((
                        extract_uniprot_accession(header).encode(),
                        sequence.encode()
                    ))
                    
                    if len(batch) >= batch_size:
                        results = pool.apply_async(process_func, (batch,))
                        processed = results.get()
                        pbar.update(processed)
                        batch = []
                
                if batch:
                    results = pool.apply_async(process_func, (batch,))
                    processed = results.get()
                    pbar.update(processed)

    logger.info(f"LMDB database created successfully at {lmdb_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create LMDB from FASTA file")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--lmdb", required=True, help="Path to output LMDB file")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--num-cpus", type=int, default=int(os.environ.get('NPROC', 1)),
                        help="Number of CPUs to use (default: NPROC environment variable or 1)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    logger.info("Script started")
    create_lmdb_from_fasta(args.fasta, args.lmdb, args.batch_size, args.num_cpus)
    logger.info("Script completed")