#!/usr/bin/env python3

import os
from Bio import SeqIO
import lmdb

def extract_uniprot_accession(record_id):
    """Extract UniProt accession from the FASTA header."""
    try:
        return record_id.split()[0].split(':')[1].split('-')[1]
    except IndexError:
        raise ValueError(f"Unable to extract UniProt accession from: {record_id}")

def create_lmdb_from_fasta(fasta_file, lmdb_path):

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=1099511627776)  # 1TB max size

    with env.begin(write=True) as txn:
        for record in SeqIO.parse(fasta_file, "fasta"):
            uniprot_accession = extract_uniprot_accession(record.id)
            sequence = str(record.seq)
            
            # Write to LMDB
            txn.put(uniprot_accession.encode(), sequence.encode())

    env.close()
    print(f"LMDB database created successfully at {lmdb_path}")

if __name__ == "__main__":
    fasta_file = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.fasta"
    lmdb_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.lmdb"
    
    create_lmdb_from_fasta(fasta_file, lmdb_path)