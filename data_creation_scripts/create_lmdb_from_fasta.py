#!/usr/bin/env python3

import os
import logging
import argparse
from typing import List, Tuple, Optional
from Bio import SeqIO
import lmdb
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import itertools

"""
Creates or updates an LMDB from AFDB FASTA file (Total records: 214,683,829 | Total size: 93GB).

LMDB structure:
key: UniProt accession (e.g. "A0A1B2C3D4")
value: Protein sequence (e.g. "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF...")

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_last_processed_accession(lmdb_path: str) -> Optional[str]:
    """Get the last processed UniProt accession from the LMDB."""
    if not os.path.exists(lmdb_path):
        return None
    
    with lmdb.open(lmdb_path, readonly=True) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            if cursor.last():
                for key, value in cursor.iterprev():
                    if value:
                        return key.decode()
    return None

def process_batch(batch: List[Tuple[bytes, bytes]], env: lmdb.Environment) -> int:
    """Process a batch of records and add them to the LMDB."""
    with env.begin(write=True) as txn:
        with txn.cursor() as cursor:
            cursor.putmulti(batch)
    return len(batch)

def extract_uniprot_accession(record_id: str) -> str:
    """Extract UniProt accession from the record ID."""
    return record_id.split()[0].split(':')[1].split('-')[1]

def create_lmdb_from_fasta(fasta_file: str, lmdb_path: str, batch_size: int, num_cpus: int):
    logger.info(f"Starting LMDB creation/update from FASTA file: {fasta_file}")
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    last_processed = get_last_processed_accession(lmdb_path)
    if last_processed:
        logger.info(f"Resuming from last processed accession: {last_processed}")
    else:
        logger.info("Starting new LMDB creation")

    env = lmdb.open(lmdb_path, writemap=True, max_dbs=1)

    total_records = 0
    batch = []
    resume_flag = last_processed is not None

    logger.info(f"Using {num_cpus} CPUs")

    with mp.Pool(num_cpus) as pool:
        process_func = partial(process_batch, env=env)

        with tqdm(total=214683829, desc="Processing records", unit=" records") as pbar:
            record_iterator = SeqIO.parse(fasta_file, "fasta")
            if resume_flag:
                record_iterator = itertools.dropwhile(
                    lambda r: extract_uniprot_accession(r.id) <= last_processed,
                    record_iterator
                )
                logger.info(f"Skipped records up to accession: {last_processed}")

            for batch in iter(lambda: list(itertools.islice(record_iterator, batch_size)), []):
                try:
                    batches = [
                        (extract_uniprot_accession(record.id).encode(), str(record.seq).encode())
                        for record in batch
                    ]
                    results = pool.apply_async(process_func, (batches,))
                    total_records += results.get()
                    pbar.update(len(batches))
                except IndexError as e:
                    logger.warning(f"Skipping record: Unable to extract UniProt accession. Error: {e}")
                except lmdb.MapFullError:
                    logger.error("LMDB map full. Consider increasing map_size.")
                    break

    env.close()
    logger.info(f"LMDB database created/updated successfully at {lmdb_path}")
    logger.info(f"Total new records processed: {total_records}")

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