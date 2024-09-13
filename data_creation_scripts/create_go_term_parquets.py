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
import psutil

# Constants
RECORDS_PER_FILE = 100

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    logger.info(f"Starting to stream GO TSV file: {file_path}")
    try:
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for row in tsv_reader:
                if len(row) == 2:
                    go_term, uniprot_accs = row
                    yield go_term, uniprot_accs.split(',')
    except Exception as e:
        logger.error(f"Error reading TSV file: {e}")
        raise
    logger.info(f"Finished streaming GO TSV file: {file_path}")

def setup_lmdb_env(lmdb_path: str) -> lmdb.Environment:
    """
    Set up and return an LMDB environment.

    Args:
        lmdb_path (str): Path to the LMDB directory.

    Returns:
        lmdb.Environment: An LMDB environment object.
    """
    logger.info(f"Setting up LMDB environment at {lmdb_path}")
    try:
        return lmdb.open(lmdb_path, readonly=True, lock=False, map_size=1099511627776)  # 1 TB
    except lmdb.Error as e:
        logger.error(f"Error opening LMDB: {e}")
        raise

def fetch_sequence(txn: lmdb.Transaction, accession: str) -> Optional[bytes]:
    lmdb_key = f"AFDB:AF-{accession}-F1".encode()
    return txn.get(lmdb_key, default=None)

def process_go_terms(go_tsv_path: str, lmdb_env: lmdb.Environment, save_dir: str, file_prefix: str) -> None:
    logger.info(f"Starting GO term processing")
    os.makedirs(save_dir, exist_ok=True)
    fail_path = os.path.join(save_dir, "failed_sequences.txt")

    total_go_terms = 0
    failed_sequences = 0
    successful_sequences = 0
    current_file_index = 0
    records_in_current_file = 0
    index_data = []

    schema = pa.schema([
        ('fam_id', pa.string()),
        ('sequences', pa.list_(pa.binary())),
        ('accessions', pa.list_(pa.string()))
    ])

    def get_parquet_writer():
        nonlocal current_file_index
        output_file = os.path.join(save_dir, f'{file_prefix}_{str(current_file_index).zfill(4)}.parquet')
        return pq.ParquetWriter(output_file, schema, use_dictionary=False)

    writer = get_parquet_writer()

    try:
        with open(fail_path, "w") as fail_file, lmdb_env.begin(write=False) as txn:
            pbar = tqdm(stream_go_tsv(go_tsv_path), desc="Processing GO terms", unit="term")
            for go_term, uniprot_accs in pbar:
                sequences = []
                success_accs = []
                
                for acc in uniprot_accs:
                    seq = fetch_sequence(txn, acc)
                    if seq:
                        success_accs.append(acc)
                        sequences.append(seq)
                        successful_sequences += 1
                    else:
                        fail_file.write(f"{acc}\n")
                        failed_sequences += 1
                
                go_data = pa.Table.from_pydict({
                    'fam_id': [go_term],
                    'sequences': [sequences],
                    'accessions': [success_accs]
                })
                
                writer.write_table(go_data)
                index_data.append({'fam_id': go_term, 'parquet_file': f'{file_prefix}_{str(current_file_index).zfill(4)}.parquet'})
                
                records_in_current_file += 1
                total_go_terms += 1
                
                if records_in_current_file >= RECORDS_PER_FILE:
                    writer.close()
                    current_file_index += 1
                    records_in_current_file = 0
                    writer = get_parquet_writer()

                # Update progress bar
                pbar.set_postfix({
                    "Total": total_go_terms, 
                    "Successful": successful_sequences,
                    "Failed": failed_sequences, 
                    "Files": current_file_index,
                    "Current File Records": records_in_current_file
                })

                # Log memory usage every 1000 terms
                if total_go_terms % 1000 == 0:
                    log_memory_usage()

    except Exception as e:
        logger.error(f"Fatal error in process_go_terms: {str(e)}")
        logger.exception("Exception details:")
    finally:
        # Ensure the last writer is closed
        if 'writer' in locals():
            writer.close()

        # Save index file
        try:
            index_df = pd.DataFrame(index_data)
            index_file = os.path.join(save_dir, "go_term_index.csv")
            index_df.to_csv(index_file, index=False)
            logger.info(f"Index file saved to {index_file}")
        except Exception as e:
            logger.error(f"Error saving index file: {str(e)}")

    logger.info(f"GO term processing completed. Total: {total_go_terms}, Successful: {successful_sequences}, Failed: {failed_sequences}, Files: {current_file_index + 1}")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def main(go_tsv_path: str, save_dir: str, lmdb_path: str, file_prefix: str) -> None:
    t0 = time.time()
    
    try:
        with setup_lmdb_env(lmdb_path) as lmdb_env:
            process_go_terms(go_tsv_path, lmdb_env, save_dir, file_prefix)
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        logger.exception("Exception details:")
    finally:
        t1 = time.time()
        logger.info(f"Total processing time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("--go_tsv_path", type=str, required=True, help="Path to the GO term TSV file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the parquet files")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to the LMDB directory containing sequence data")
    parser.add_argument("--file_prefix", type=str, default="GO", help="Prefix for output Parquet files")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir, args.lmdb_path, args.file_prefix)