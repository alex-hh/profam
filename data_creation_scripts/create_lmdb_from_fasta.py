#!/usr/bin/env python3

import os
import logging
import argparse
import lmdb
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_uniprot_accession(record_id: str) -> str:
    try:
        return record_id.split()[0].split(':')[1].split('-')[1]
    except IndexError:
        return None

def create_lmdb_from_fasta(fasta_file: str, lmdb_path: str, batch_size: int = 100000):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    env = lmdb.open(lmdb_path, map_size=200 * 1024 * 1024 * 1024)  # 200GB map size

    total_records = 214683829  # Hard-coded total records
    batch = []

    try:
        with env.begin(write=True) as txn, open(fasta_file, 'r') as fasta:
            for _ in tqdm(range(total_records), desc="Processing records"):
                header = next(fasta, '').strip()
                if not header.startswith('>'):
                    break
                
                sequence = next(fasta, '').strip()
                accession = extract_uniprot_accession(header[1:])
                
                if accession:
                    batch.append((accession.encode(), sequence.encode()))
                
                if len(batch) >= batch_size:
                    txn.putmulti(batch)
                    batch.clear()
            
            if batch:
                txn.putmulti(batch)

    except lmdb.MapFullError:
        logger.error("LMDB map full. Consider increasing map_size.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        env.close()

    logger.info(f"LMDB database created/updated successfully at {lmdb_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create LMDB from FASTA file")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--lmdb", required=True, help="Path to output LMDB file")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for processing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    create_lmdb_from_fasta(args.fasta, args.lmdb, args.batch_size)