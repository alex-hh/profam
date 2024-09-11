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

def extract_uniprot_accession(record_id: str) -> str:
    return record_id.split(' ')[0].split(':')[1].split('-')[1]

def mmap_fasta_parser(file_path):
    with open(file_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header = b''
        sequence = []
        for line in iter(mm.readline, b''):
            line = line.strip()
            if line.startswith(b'>'):
                if header:
                    yield header.decode(), b''.join(sequence).decode()
                header = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if header:
            yield header.decode(), b''.join(sequence).decode()
        mm.close()

def create_lmdb_from_fasta(fasta_file: str, lmdb_path: str, batch_size: int):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=200 * 1024 * 1024 * 1024, writemap=True)  # 200GB map size

    try:
        total_records = 214683829
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            buffer = []
            start_time = time.time()
            with tqdm(total=total_records, desc="Processing records", unit=" records") as pbar:
                for header, sequence in mmap_fasta_parser(fasta_file):
                    accession = extract_uniprot_accession(header)
                    buffer.append((accession.encode(), sequence.encode()))
                    
                    if len(buffer) >= batch_size:
                        cursor.putmulti(buffer)
                        buffer = []
                        
                        # Commit and start a new transaction every 1 million records
                        if pbar.n % 1000000 == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                            cursor = txn.cursor()
                    
                    pbar.update(1)
                    
                    # Print progress every 5 million records
                    if pbar.n % 5000000 == 0:
                        elapsed_time = time.time() - start_time
                        records_per_second = pbar.n / elapsed_time
                        logger.info(f"Processed {pbar.n} records. Speed: {records_per_second:.2f} records/second")

                # Write any remaining records in the buffer
                if buffer:
                    cursor.putmulti(buffer)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        env.sync()
        env.close()

    logger.info(f"LMDB database created successfully at {lmdb_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create LMDB from FASTA file")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--lmdb", required=True, help="Path to output LMDB file")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for processing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    create_lmdb_from_fasta(args.fasta, args.lmdb, args.batch_size)