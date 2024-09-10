#!/usr/bin/env python3

import os
import logging
from Bio import SeqIO
import lmdb
from tqdm import tqdm
import multiprocessing as mp
import itertools

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_last_processed_accession(lmdb_path):
    """Get the last processed UniProt accession from the LMDB."""
    if not os.path.exists(lmdb_path):
        return None
    
    with lmdb.open(lmdb_path, readonly=True) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            if cursor.last():
                return cursor.key().decode()
    return None

def process_fasta_chunk(chunk, lmdb_path, last_processed):
    """Process a chunk of FASTA records and write to LMDB."""
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    processed_count = 0
    with env.begin(write=True) as txn:
        for record in chunk:
            try:
                uniprot_accession = record.id.split()[0].split(':')[1].split('-')[1]
                if last_processed and uniprot_accession <= last_processed:
                    continue
                txn.put(uniprot_accession.encode(), str(record.seq).encode())
                processed_count += 1
            except IndexError:
                logger.warning(f"Skipping record: Unable to extract UniProt accession from {record.id}")
    return processed_count

def create_lmdb_from_fasta(fasta_file, lmdb_path, batch_size=10000, num_processes=mp.cpu_count()):
    logger.info(f"Starting LMDB creation/update from FASTA file: {fasta_file}")
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    last_processed = get_last_processed_accession(lmdb_path)
    if last_processed:
        logger.info(f"Resuming from last processed accession: {last_processed}")
    else:
        logger.info("Starting new LMDB creation")

    with mp.Pool(processes=num_processes) as pool:
        fasta_parser = SeqIO.parse(fasta_file, "fasta")
        chunks = iter(lambda: list(itertools.islice(fasta_parser, batch_size)), [])
        
        results = pool.imap(
            lambda chunk: process_fasta_chunk(chunk, lmdb_path, last_processed),
            chunks
        )
        
        total_records = sum(tqdm(results, desc="Processing records", unit=" records"))

    logger.info(f"LMDB database created/updated successfully at {lmdb_path}")
    logger.info(f"Total new records processed: {total_records}")

if __name__ == "__main__":
    fasta_file = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.fasta"
    lmdb_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.lmdb"
    
    logger.info("Script started")
    create_lmdb_from_fasta(fasta_file, lmdb_path)
    logger.info("Script completed")