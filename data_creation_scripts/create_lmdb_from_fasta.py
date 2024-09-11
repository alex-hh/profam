#!/usr/bin/env python3

import os
import logging
import argparse
from typing import List, Tuple
from Bio import SeqIO
import lmdb
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

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
    """Process a batch of records and add them to the LMDB."""
    env = lmdb.open(lmdb_path, map_size=1099511627776, writemap=True, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            with txn.cursor() as cursor:
                cursor.putmulti(batch)
        return len(batch)
    finally:
        env.close()

def worker_function(batch: List[Tuple[bytes, bytes]], lmdb_path: str) -> int:
    """Worker function for multiprocessing."""
    try:
        return process_batch(batch, lmdb_path)
    except lmdb.MapFullError:
        logger.error("LMDB map full. Consider increasing map_size.")
        return 0

def extract_uniprot_accession(record_id: str) -> str:
    """Extract UniProt accession from the record ID."""
    return record_id.split()[0].split(':')[1].split('-')[1]

def create_lmdb_from_fasta(fasta_file: str, lmdb_path: str, batch_size: int, num_cpus: int):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    logger.info(f"Using {num_cpus} CPUs")

    with mp.Pool(num_cpus) as pool:
        process_func = partial(worker_function, lmdb_path=lmdb_path)

        with tqdm(total=214683829, desc="Processing records", unit=" records") as pbar:
            record_iterator = SeqIO.parse(fasta_file, "fasta")
            
            while True:
                batch = []
                for _ in range(batch_size):
                    try:
                        record = next(record_iterator)
                        batch.append((
                            extract_uniprot_accession(record.id).encode(),
                            str(record.seq).encode()
                        ))
                    except StopIteration:
                        break
                
                if not batch:
                    break
                
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