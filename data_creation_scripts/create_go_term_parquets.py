import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Iterator, Tuple, Optional
import time
import csv
import gzip
import sys
import logging
import lmdb
from tqdm import tqdm

# Constants
NUM_PARQUET_FILES = 300
BATCH_SIZE = 1000

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def stream_go_tsv(file_path: str) -> Iterator[Tuple[str, List[str]]]:
    """
    Stream GO terms and their associated UniProt accessions from a TSV file.

    Args:
        file_path (str): Path to the GO term TSV file.

    Yields:
        Tuple[str, List[str]]: A tuple containing the GO term and a list of UniProt accessions.
    """
    open_func = gzip.open if file_path.endswith('.gz') else open
    
    csv.field_size_limit(sys.maxsize)
    
    logging.info(f"Starting to stream GO TSV file: {file_path}")
    try:
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for row in tsv_reader:
                if len(row) == 2:
                    go_term, uniprot_accs = row
                    yield go_term, uniprot_accs.split(',')
                else:
                    logging.warning(f"Skipping malformed row: {row}")
    except Exception as e:
        logging.error(f"Error reading TSV file: {e}")
        raise
    logging.info(f"Finished streaming GO TSV file: {file_path}")

def setup_lmdb_env(lmdb_path: str) -> lmdb.Environment:
    """
    Set up and return an LMDB environment.

    Args:
        lmdb_path (str): Path to the LMDB directory.

    Returns:
        lmdb.Environment: An LMDB environment object.
    """
    logging.info(f"Setting up LMDB environment at {lmdb_path}")
    try:
        return lmdb.open(lmdb_path, readonly=True, lock=False)
    except lmdb.Error as e:
        logging.error(f"Error opening LMDB: {e}")
        raise

def batch_fetch_sequences(txn: lmdb.Transaction, accessions: List[str]) -> Dict[str, Optional[bytes]]:
    results = {}
    for acc in accessions:
        # Construct the key in the format used in the LMDB
        lmdb_key = f"AFDB:AF-{acc}-F1".encode()
        value = txn.get(lmdb_key, default=None)
        results[acc] = value
    return results

def process_go_terms(go_tsv_path: str, lmdb_env: lmdb.Environment, save_dir: str, file_prefix: str) -> None:
    logging.info(f"Starting GO term processing")
    os.makedirs(save_dir, exist_ok=True)
    fail_path = os.path.join(save_dir, "failed_sequences.txt")

    total_go_terms = 0
    failed_sequences = 0
    current_parquet_data = []
    current_file_index = 0
    index_data = []

    def write_parquet_file():
        nonlocal current_file_index, current_parquet_data
        if current_parquet_data:
            output_file = os.path.join(save_dir, f'{file_prefix}_{str(current_file_index).zfill(4)}.parquet')
            df = pd.DataFrame(current_parquet_data)
            df.to_parquet(output_file, index=False)
            logging.info(f"Saved {len(current_parquet_data)} GO terms to {output_file}")
            current_parquet_data = []  # Clear the data after writing
            current_file_index += 1

    with open(fail_path, "w") as fail_file, lmdb_env.begin(write=False) as txn:
        for go_term, uniprot_accs in tqdm(stream_go_tsv(go_tsv_path), desc="Processing GO terms"):
            sequences = []
            success_accs = []
            
            # Batch fetch sequences
            seq_dict = batch_fetch_sequences(txn, uniprot_accs)
            for acc, seq in seq_dict.items():
                if seq:
                    success_accs.append(acc)
                    sequences.append(seq)  # Keep as bytes
                else:
                    fail_file.write(f"{acc}\n")
                    failed_sequences += 1
            
            go_data = {
                'fam_id': go_term,
                'sequences': sequences,
                'accessions': success_accs
            }
            
            current_parquet_data.append(go_data)
            index_data.append({'fam_id': go_term, 'parquet_file': f'{file_prefix}_{str(current_file_index).zfill(4)}.parquet'})
            
            total_go_terms += 1
            
            # Write and start a new file if we've accumulated enough GO terms
            if len(current_parquet_data) >= BATCH_SIZE:
                write_parquet_file()

    # Write any remaining data
    write_parquet_file()

    # Save index file
    index_df = pd.DataFrame(index_data)
    index_df.to_csv(os.path.join(save_dir, "go_term_index.csv"), index=False)
    logging.info(f"Index file saved")

    logging.info(f"GO term processing completed")
    logging.info(f"Total GO terms processed: {total_go_terms}")
    logging.info(f"Failed sequences: {failed_sequences}")
    logging.info(f"Number of Parquet files created: {current_file_index}")

def main(go_tsv_path: str, save_dir: str, lmdb_path: str, file_prefix: str) -> None:
    t0 = time.time()
    
    try:
        with setup_lmdb_env(lmdb_path) as lmdb_env:
            process_go_terms(go_tsv_path, lmdb_env, save_dir, file_prefix)
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise

    t1 = time.time()
    logging.info(f"Total processing time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("--go_tsv_path", type=str, required=True, help="Path to the GO term TSV file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the parquet files")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to the LMDB directory containing sequence data")
    parser.add_argument("--file_prefix", type=str, default="GO", help="Prefix for output Parquet files")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir, args.lmdb_path, args.file_prefix)