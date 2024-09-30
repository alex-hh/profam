import os
import lmdb
import gzip
import logging
import re
from Bio import SeqIO
from tqdm import tqdm

# Constants
INPUT_DIR = "/SAN/orengolab/cath_plm/ProFam/data/uniprot"
SPROT_FILE = "uniprot_sprot.fasta.gz"
TREMBL_FILE = "uniprot_trembl.fasta.gz"
OUTPUT_DIR = "/SAN/orengolab/cath_plm/ProFam/data/uniprot"
LMDB_DB_NAME = "uniprot_dict.lmdb"
MAP_SIZE = 1 << 38
BATCH_SIZE = 10000

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def parse_fasta(file_path):
    """
    Generator function to parse a gzipped FASTA file.

    Args:
        file_path (str): Path to the gzipped FASTA file.

    Yields:
        tuple: (header, sequence) for each record in the FASTA file.
    """
    try:
        with gzip.open(file_path, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                yield record.id, str(record.seq)
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        raise

def extract_uniprot_id(header):
    """
    Extract UniProt ID from the FASTA header.

    Args:
        header (str): FASTA header line.

    Returns:
        str: UniProt ID.
    """
    match = re.match(r"(\w+)", header)
    if match:
        return match.group(1)
    else:
        logging.warning(f"Could not extract UniProt ID from header: {header}")
        return header

def create_lmdb_database(input_files, output_path, map_size):
    """
    Create an LMDB database from given FASTA files, processing in batches using a cursor.

    Args:
        input_files (list): List of paths to gzipped FASTA files.
        output_path (str): Path to the output LMDB directory.
        map_size (int): Maximum size of the LMDB map.
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(
        output_path,
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=True,
        meminit=True
    )
    
    total_sequences = 0
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        for file in input_files:
            logging.info(f"Processing file: {file}")
            file_path = os.path.join(INPUT_DIR, file)
            for batch in tqdm(parse_fasta(file_path), desc=f"Processing {file}", unit="batch"):
                for uniprot_id, sequence in batch:
                    cursor.put(uniprot_id.encode('utf-8'), sequence.encode('utf-8'))
                total_sequences += len(batch)
                
                # Commit the transaction every 10 batches to save memory
                if total_sequences % (BATCH_SIZE * 10) == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    cursor = txn.cursor()
    
    env.close()
    logging.info(f"LMDB database created at {output_path}")
    logging.info(f"Total sequences processed: {total_sequences}")

def main():
    """
    Main function to orchestrate LMDB database creation from UniProt FASTA files.
    """
    input_files = [SPROT_FILE, TREMBL_FILE]
    output_path = os.path.join(OUTPUT_DIR, LMDB_DB_NAME)
    create_lmdb_database(input_files, output_path, MAP_SIZE)

if __name__ == "__main__":
    main()
