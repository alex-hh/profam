#!/usr/bin/env python3

import os
import logging
from Bio import SeqIO
import lmdb
from tqdm import tqdm

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

def create_lmdb_from_fasta(fasta_file, lmdb_path, batch_size=10000):
    logger.info(f"Starting LMDB creation/update from FASTA file: {fasta_file}")
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    last_processed = get_last_processed_accession(lmdb_path)
    if last_processed:
        logger.info(f"Resuming from last processed accession: {last_processed}")
    else:
        logger.info("Starting new LMDB creation")

    env = lmdb.open(lmdb_path, map_size=1099511627776, writemap=True)
    total_records = 0
    batch = []

    with env.begin(write=True) as txn:
        for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Processing records", unit=" records"):
            try:
                uniprot_accession = record.id.split()[0].split(':')[1].split('-')[1]
                if last_processed and uniprot_accession <= last_processed:
                    continue
                batch.append((uniprot_accession.encode(), str(record.seq).encode()))
                if len(batch) >= batch_size:
                    with txn.cursor() as cursor:
                        cursor.putmulti(batch)
                    total_records += len(batch)
                    batch = []
                    txn.commit()
                    txn = env.begin(write=True)
            except IndexError:
                logger.warning(f"Skipping record: Unable to extract UniProt accession from {record.id}")
            except lmdb.MapFullError:
                logger.error("LMDB map full. Consider increasing map_size.")
                break

        # Write any remaining records
        if batch:
            with txn.cursor() as cursor:
                cursor.putmulti(batch)
            total_records += len(batch)

    env.close()
    logger.info(f"LMDB database created/updated successfully at {lmdb_path}")
    logger.info(f"Total new records processed: {total_records}")

if __name__ == "__main__":
    fasta_file = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.fasta"
    lmdb_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.lmdb"
    
    logger.info("Script started")
    create_lmdb_from_fasta(fasta_file, lmdb_path)
    logger.info("Script completed")