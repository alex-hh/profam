#!/usr/bin/env python3

import os
import logging
import argparse
import lmdb
import gzip
from tqdm import tqdm
from Bio import SeqIO

"""
This script creates an LMDB database from one or more FASTA files.
It supports both gzipped and uncompressed FASTA files and processes records in batches
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_lmdb_from_fasta(fasta_files, lmdb_path: str, batch_size: int = 100000, total_records: int = None):
    if isinstance(fasta_files, str):
        fasta_files = [fasta_files]
    
    logger.info(f"Starting LMDB creation from FASTA file(s): {', '.join(fasta_files)}")
    env = lmdb.open(lmdb_path, map_size=300 * 1024 * 1024 * 1024)  # 300GB map size

    batch = []

    try:
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            for fasta_file in fasta_files:
                open_func = gzip.open if fasta_file.endswith('.gz') else open
                with open_func(fasta_file, "rt") as handle:
                    for record in tqdm(SeqIO.parse(handle, "fasta"), 
                                       desc=f"Processing {fasta_file}", 
                                       total=total_records):
                        batch.append((record.id.encode(), str(record.seq).encode()))
                        
                        if len(batch) >= batch_size:
                            cursor.putmulti(batch)
                            batch.clear()
            
            if batch:
                cursor.putmulti(batch)

    except lmdb.MapFullError:
        logger.error("LMDB map full. Consider increasing map_size.")
        return
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return
    finally:
        env.close()

    logger.info(f"LMDB database created/updated successfully at {lmdb_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create LMDB from FASTA file(s)")
    parser.add_argument("--fasta", required=True, nargs='+', help="Path to input FASTA file(s)")
    parser.add_argument("--lmdb", required=True, help="Path to output LMDB file")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for processing")
    parser.add_argument("--total-records", type=int, default = 245324902, help="Total number of records (optional, for progress bar)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    create_lmdb_from_fasta(args.fasta, args.lmdb, args.batch_size, args.total_records)