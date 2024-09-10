#!/usr/bin/env python3

import os
import logging
from Bio import SeqIO
import lmdb

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

def create_lmdb_from_fasta(fasta_file, lmdb_path):
    logger.info(f"Starting LMDB creation from FASTA file: {fasta_file}")
    logger.info(f"LMDB will be created at: {lmdb_path}")

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    logger.info(f"Ensured directory exists: {os.path.dirname(lmdb_path)}")

    env = lmdb.open(lmdb_path, map_size=1099511627776)  # 1TB max size
    logger.info("LMDB environment opened")

    record_count = 0
    with env.begin(write=True) as txn:
        for record in SeqIO.parse(fasta_file, "fasta"):
            try:
                uniprot_accession = extract_uniprot_accession(record.id)
                sequence = str(record.seq)
                
                # Write to LMDB
                txn.put(uniprot_accession.encode(), sequence.encode())
                record_count += 1
                
                if record_count % 1000000 == 0:
                    logger.info(f"Processed {record_count} records")
            except ValueError as e:
                logger.warning(f"Skipping record due to error: {str(e)}")

    env.close()
    logger.info(f"LMDB database created successfully at {lmdb_path}")
    logger.info(f"Total records processed: {record_count}")

if __name__ == "__main__":
    fasta_file = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.fasta"
    lmdb_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.lmdb"
    
    logger.info("Script started")
    create_lmdb_from_fasta(fasta_file, lmdb_path)
    logger.info("Script completed")