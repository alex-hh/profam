import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List
import time
import csv
import gzip
import sys
import logging
import pickle

# Constants
NUM_PARQUETS = 300
BATCH_SIZE = 1000000

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_go_tsv(file_path: str) -> Dict[str, List[str]]:
    go_dict: Dict[str, List[str]] = {}
    open_func = gzip.open if file_path.endswith('.gz') else open
    
    csv.field_size_limit(sys.maxsize)
    
    logging.info(f"Reading GO TSV file: {file_path}")
    with open_func(file_path, 'rt', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            if len(row) == 2:
                go_term, uniprot_accs = row
                go_dict[go_term] = uniprot_accs.split(',')
    logging.info(f"Finished reading GO TSV file. Found {len(go_dict)} GO terms.")
    return go_dict

def load_sequence_dict(filepath: str) -> Dict[str, str]:
    logging.info(f"Loading sequence dictionary from {filepath}...")
    with open(filepath, 'rb') as f:
        seq_dict = pickle.load(f)
    logging.info(f"Loaded {len(seq_dict)} sequences.")
    return seq_dict

def process_go_terms(go_dict: Dict[str, List[str]], seq_dict: Dict[str, str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    fail_path = os.path.join(save_dir, "failed_sequences.txt")
    index_data = []

    total_go_terms = len(go_dict)
    go_terms_per_parquet = total_go_terms // NUM_PARQUETS + 1

    logging.info(f"Processing {total_go_terms} GO terms")
    
    current_parquet = 0
    current_go_terms = 0
    data = []

    with open(fail_path, "w") as fail_file:
        for go_term, uniprot_accs in go_dict.items():
            sequences = []
            success_accs = []
            for acc in uniprot_accs:
                seq = seq_dict.get(acc)
                if seq:
                    success_accs.append(acc)
                    sequences.append(seq)
                else:
                    fail_file.write(f"{acc}\n")
            
            data.append({
                'fam_id': go_term,
                'sequences': sequences,
                'accessions': success_accs
            })
            current_go_terms += 1

            if current_go_terms >= go_terms_per_parquet:
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

def main(go_tsv_path: str, save_dir: str, seq_dict_path: str) -> None:
    t0 = time.time()
    
    seq_dict = load_sequence_dict(seq_dict_path)
    go_dict = read_go_tsv(go_tsv_path)
    process_go_terms(go_dict, seq_dict, save_dir)

    t1 = time.time()
    logging.info(f"Total processing time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("go_tsv_path", type=str, help="Path to the GO term TSV file")
    parser.add_argument("save_dir", type=str, help="Directory to save the parquet files")
    parser.add_argument("seq_dict_path", type=str, help="Path to the sequence dictionary pickle file")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir, args.seq_dict_path)