import argparse
import pickle
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from typing import List, Dict, Optional
import time
import csv
import gzip
import sys
import logging
import mmap
import lmdb

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the logging level to DEBUG
logging.getLogger().setLevel(logging.DEBUG)

"""
@data_creation_scripts @create_foldseek_struct_with_af50.py

We want to create a new data creation script for GO documents.
The script should take as input a tsv file with GO terms (which will be the fam_id) and
a list of uniprot accessions that are annotated with this term.
The script will need to look up the sequence associated with the uniprot accession
store the results in a parquet file (one row for each document (GO term))
the columns will be a string for the go-term (column called fam_id)
a column with a list of sequences, and a column with a list of uniprot accession codes
(corresponding the the sequences).

Split the documents into 300 parquet files in total, the allocation of go-terms to
parquet files should be shuffled and random. The parquet files should aim to be roughly
equal in size (so parquets with large families should have fewer families in total)

See:
data_creation_scripts/create_foldseek_struct_with_af50.py
for reference
This should also contain paths to where you can look up from uniprot ID to sequence.

"""


def read_go_tsv(file_path: str) -> Dict[str, List[str]]:
    go_dict = {}
    open_func = gzip.open if file_path.endswith('.gz') else open
    
    # Increase the field size limit to maximum
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


def write_parquet_file(current_documents, current_parquet, save_dir):
    df = pd.DataFrame(current_documents)
    table = pa.Table.from_pandas(df, preserve_index=False)
    output_file = os.path.join(save_dir, f'GO_{str(current_parquet).zfill(4)}.parquet')
    pq.write_table(table, output_file)
    logging.info(f"Saved {len(current_documents)} GO terms with {len(df)} sequences to {output_file}")


def write_index_file(index_data: List[Dict[str, str]], save_dir: str):
    index_file_path = os.path.join(save_dir, "go_term_index.csv")
    with open(index_file_path, 'w', newline='') as csvfile:
        fieldnames = ['fam_id', 'parquet_file']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in index_data:
            writer.writerow(item)
    
    logging.info(f"Index file saved to {index_file_path}")

def create_and_save_go_documents(
    go_dict: Dict[str, List[str]],
    save_dir: str,
    num_parquets: int,
    seq_lookup: lmdb.Environment
):
    os.makedirs(save_dir, exist_ok=True)
    fail_path = os.path.join(save_dir, "failed_sequences.txt")
    total_sequences = sum(len(accs) for accs in go_dict.values())
    sequences_per_parquet = total_sequences // num_parquets

    current_parquet = 0
    current_sequences = 0
    fail_counter = 0
    success_counter = 0
    current_documents = []
    index_data = []

    logging.info(f"Starting to process {len(go_dict)} GO terms")
    with open(fail_path, "w") as f:
        f.write("Failed sequences:\n")
        for go_term, uniprot_accs in go_dict.items():
            sequences = []
            success_accs = []
            logging.debug(f"Processing GO term: {go_term} with {len(uniprot_accs)} accessions")
            for acc in uniprot_accs:
                seq = get_sequence(acc, seq_lookup)
                if seq is None:
                    f.write(f"{acc}\n")
                    fail_counter += 1
                else:
                    success_accs.append(acc)
                    sequences.append(seq)
                    success_counter += 1
            
            current_documents.append({
                'fam_id': go_term,
                'sequences': sequences,
                'accessions': success_accs
            })
            current_sequences += len(sequences)

            # Add to index data
            parquet_name = f'GO_{str(current_parquet).zfill(4)}.parquet'
            index_data.append({'fam_id': go_term, 'parquet_file': parquet_name})

            if current_sequences >= sequences_per_parquet:
                write_parquet_file(current_documents, current_parquet, save_dir)

                current_parquet += 1
                current_sequences = 0
                current_documents = []
                logging.info(f"Processed {success_counter} sequences, {fail_counter} failed")

    # Save any remaining documents
    if current_documents:
        write_parquet_file(current_documents, current_parquet, save_dir)

    # Write the index file
    write_index_file(index_data, save_dir)
    logging.info(f"Finished processing. Total successes: {success_counter}, Total failures: {fail_counter}")
    logging.info(f"Sample of failed accessions: {list(go_dict.values())[0][:5]}")

def load_sequence_dict(filepath):
    logging.info(f"Loading sequence dictionary from {filepath}...")
    lmdb_path = filepath + '_lmdb'
    
    if not os.path.exists(lmdb_path):
        logging.info(f"LMDB not found. Creating from pickle file: {filepath}")
        create_lmdb_from_pickle(filepath, lmdb_path)
    else:
        logging.info(f"LMDB found at {lmdb_path}")
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    # Check if the database is empty
    with env.begin() as txn:
        cursor = txn.cursor()
        first = cursor.first()
        if not first:
            logging.error("The LMDB database is empty!")
        else:
            logging.info(f"LMDB database contains data. First key: {first[0].decode()}")
    
    return env

def get_sequence(acc: str, seq_lookup: lmdb.Environment) -> Optional[str]:
    with seq_lookup.begin() as txn:
        seq = txn.get(acc.encode())
        return seq.decode() if seq else None

def create_lmdb_from_pickle(pickle_path, lmdb_path, map_size=1099511627776):  # 1TB
    logging.info(f"Creating LMDB at {lmdb_path} from {pickle_path}")
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    with open(pickle_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        offset = 0
        count = 0
        while offset < len(m):
            try:
                obj = pickle.loads(m[offset:])
                with env.begin(write=True) as txn:
                    for key, value in obj.items():
                        txn.put(key.encode(), value.encode())
                        count += 1
                        if count % 100000 == 0:
                            logging.info(f"Processed {count} entries")
                offset = m.tell()
            except Exception as e:
                logging.error(f"Error at offset {offset}: {str(e)}")
                offset += 1
    
    env.close()
    logging.info(f"Finished creating LMDB. Total entries: {count}")

def main(go_tsv_path: str, save_dir: str):
    t0 = time.time()
    sequence_dict_pickle_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/afdb_sequence_dict.pkl"
    
    # Load the sequence dictionary
    seq_lookup = load_sequence_dict(sequence_dict_pickle_path)
    
    # Print a sample of keys from the sequence dictionary
    with seq_lookup.begin() as txn:
        cursor = txn.cursor()
        sample_keys = [key.decode() for key, _ in cursor.iternext(keys=True, values=False)][:5]
    logging.info(f"Sample keys from sequence dictionary: {sample_keys}")

    logging.info("Reading GO TSV file...")
    go_dict = read_go_tsv(go_tsv_path)

    logging.info("Creating and saving GO documents...")
    create_and_save_go_documents(go_dict, save_dir, 300, seq_lookup)

    seq_lookup.close()  # Close the LMDB environment

    t1 = time.time()
    logging.info(f"Total processing time: {t1 - t0:.2f} seconds")

    # Print a sample of keys from the sequence dictionary
    with seq_lookup.begin() as txn:
        cursor = txn.cursor()
        sample_keys = [key.decode() for key, _ in cursor.iternext(keys=True, values=False)][:5]
    logging.info(f"Sample keys from sequence dictionary: {sample_keys}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("go_tsv_path", type=str, help="Path to the GO term TSV file")
    parser.add_argument("save_dir", type=str, help="Directory to save the parquet files")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir)