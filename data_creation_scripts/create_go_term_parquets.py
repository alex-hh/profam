import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Iterator, Tuple
import time
import csv
import gzip
import sys
import logging
import pickle

# Constants
NUM_PARQUETS = 300

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
    with open_func(file_path, 'rt', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            if len(row) == 2:
                go_term, uniprot_accs = row
                yield go_term, uniprot_accs.split(',')
    logging.info(f"Finished streaming GO TSV file: {file_path}")

def load_sequence_dict(filepath: str) -> Dict[str, str]:
    logging.info(f"Starting to load sequence dictionary from {filepath}")
    start_time = time.time()
    seq_dict = {}
    try:
        with open(filepath, 'rb') as f:
            while True:
                try:
                    key = pickle.load(f)
                    value = pickle.load(f)
                    seq_dict[key] = value
                except EOFError:
                    break
                except Exception as e:
                    logging.warning(f"Error loading a key-value pair: {str(e)}")
                    continue
        end_time = time.time()
        logging.info(f"Loaded {len(seq_dict)} sequences in {end_time - start_time:.2f} seconds")
        return seq_dict
    except Exception as e:
        logging.error(f"Error loading sequence dictionary: {str(e)}")
        raise

def process_go_terms(go_tsv_path: str, seq_dict: Dict[str, str], save_dir: str) -> None:
    """
    Process GO terms from the TSV file, fetch sequences, and save to Parquet files.

    Args:
        go_tsv_path (str): Path to the GO term TSV file.
        seq_dict (Dict[str, str]): Dictionary of sequences.
        save_dir (str): Directory to save the Parquet files.
    """
    logging.info(f"Starting GO term processing")
    os.makedirs(save_dir, exist_ok=True)
    fail_path = os.path.join(save_dir, "failed_sequences.txt")
    index_data = []

    current_parquet = 0
    current_go_terms = 0
    data = []
    total_go_terms = 0
    failed_sequences = 0

    start_time = time.time()
    with open(fail_path, "w") as fail_file:
        for go_term, uniprot_accs in stream_go_tsv(go_tsv_path):
            sequences = []
            success_accs = []
            for acc in uniprot_accs:
                seq = seq_dict.get(acc)
                if seq:
                    success_accs.append(acc)
                    sequences.append(seq)
                else:
                    fail_file.write(f"{acc}\n")
                    failed_sequences += 1
            
            data.append({
                'fam_id': go_term,
                'sequences': sequences,
                'accessions': success_accs
            })
            current_go_terms += 1
            total_go_terms += 1

            if current_go_terms >= len(data) // NUM_PARQUETS + 1:
                df = pd.DataFrame(data)
                output_file = os.path.join(save_dir, f'GO_{str(current_parquet).zfill(4)}.parquet')
                df.to_parquet(output_file, index=False)
                logging.info(f"Saved {len(data)} GO terms to {output_file}")
                
                index_data.extend([{'fam_id': row['fam_id'], 'parquet_file': f'GO_{str(current_parquet).zfill(4)}.parquet'} for row in data])
                
                current_parquet += 1
                current_go_terms = 0
                data = []

    if data:
        df = pd.DataFrame(data)
        output_file = os.path.join(save_dir, f'GO_{str(current_parquet).zfill(4)}.parquet')
        df.to_parquet(output_file, index=False)
        logging.info(f"Saved {len(data)} GO terms to {output_file}")
        
        index_data.extend([{'fam_id': row['fam_id'], 'parquet_file': f'GO_{str(current_parquet).zfill(4)}.parquet'} for row in data])

    # Write index file
    index_df = pd.DataFrame(index_data)
    index_df.to_csv(os.path.join(save_dir, "go_term_index.csv"), index=False)
    logging.info(f"Index file saved")

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"GO term processing completed in {processing_time:.2f} seconds")
    logging.info(f"Total GO terms processed: {total_go_terms}")
    logging.info(f"Failed sequences: {failed_sequences}")
    logging.info(f"Average processing time per GO term: {processing_time / total_go_terms:.4f} seconds")

def main(go_tsv_path: str, save_dir: str, seq_dict_path: str) -> None:
    t0 = time.time()
    
    seq_dict = load_sequence_dict(seq_dict_path)
    process_go_terms(go_tsv_path, seq_dict, save_dir)

    t1 = time.time()
    logging.info(f"Total processing time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("go_tsv_path", type=str, help="Path to the GO term TSV file")
    parser.add_argument("save_dir", type=str, help="Directory to save the parquet files")
    parser.add_argument("seq_dict_path", type=str, help="Path to the sequence dictionary pickle file")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir, args.seq_dict_path)