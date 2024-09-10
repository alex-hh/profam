#!/usr/bin/env python3

import os
import logging
from Bio import SeqIO
import lmdb
from tqdm import tqdm
import itertools

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_uniprot_accession(record_id):
    """Extract UniProt accession from the FASTA header."""
    try:
        return record_id.split()[0].split(':')[1].split('-')[1]
    except IndexError:
        logger.error(f"Unable to extract UniProt accession from: {record_id}")
        raise ValueError(f"Unable to extract UniProt accession from: {record_id}")

def create_lmdb_from_fasta(fasta_file, lmdb_path, batch_size=100000):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    logger.info(f"LMDB will be created at: {lmdb_path}")

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    logger.info(f"Ensured directory exists: {os.path.dirname(lmdb_path)}")

    env = lmdb.open(lmdb_path, map_size=1099511627776, writemap=True)  # 1TB max size, use writemap for better performance
    logger.info("LMDB environment opened")

    record_count = 0
    batch = {}

    # Use a generator to parse the FASTA file
    fasta_parser = SeqIO.parse(fasta_file, "fasta")
    
    with env.begin(write=True) as txn:
        with tqdm(desc="Processing records", unit=" records") as pbar:
            while True:
                try:
                    chunk = list(itertools.islice(fasta_parser, batch_size))
                    if not chunk:
                        break

                    for record in chunk:
                        try:
                            uniprot_accession = extract_uniprot_accession(record.id)
                            sequence = str(record.seq)
                            
                            batch[uniprot_accession.encode()] = sequence.encode()
                            record_count += 1
                            pbar.update(1)
                            
                        except ValueError as e:
                            logger.warning(f"Skipping record due to error: {str(e)}")

                    # Write batch to LMDB
                    with txn.cursor() as curs:
                        curs.putmulti(batch.items())
                    batch.clear()

                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    # Optionally, implement a retry mechanism here

    env.close()
    logger.info(f"LMDB database created successfully at {lmdb_path}")
    logger.info(f"Total records processed: {record_count}")

if __name__ == "__main__":
    fasta_file = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.fasta"
    lmdb_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.lmdb"
    
    logger.info("Script started")
    create_lmdb_from_fasta(fasta_file, lmdb_path)
    logger.info("Script completed")