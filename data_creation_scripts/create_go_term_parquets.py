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
    
    with open_func(file_path, 'rt', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            if len(row) == 2:
                go_term, uniprot_accs = row
                go_dict[go_term] = uniprot_accs.split(',')
    return go_dict


def write_parquet_file(current_documents, current_parquet, save_dir):
    df = pd.DataFrame(current_documents)
    table = pa.Table.from_pandas(df, preserve_index=False)
    output_file = os.path.join(save_dir, f'GO_{str(current_parquet).zfill(4)}.parquet')
    pq.write_table(table, output_file)
    print(f"Saved {len(current_documents)} GO terms with {len(df)} sequences to {output_file}")


def write_index_file(index_data: List[Dict[str, str]], save_dir: str):
    index_file_path = os.path.join(save_dir, "go_term_index.csv")
    with open(index_file_path, 'w', newline='') as csvfile:
        fieldnames = ['fam_id', 'parquet_file']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in index_data:
            writer.writerow(item)
    
    print(f"Index file saved to {index_file_path}")

def create_and_save_go_documents(
    go_dict: Dict[str, List[str]],
    save_dir: str,
    num_parquets: int,
    seq_dict: Dict[str, str]
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

    with open(fail_path, "w") as f:
        f.write("Failed sequences:\n")
        for go_term, uniprot_accs in go_dict.items():
            sequences = []
            success_accs = []
            for acc in uniprot_accs:
                seq = get_sequence(acc, seq_dict)
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
                print(f"Processed {success_counter} sequences, {fail_counter} failed")

    # Save any remaining documents
    if current_documents:
        write_parquet_file(current_documents, current_parquet, save_dir)

    # Write the index file
    write_index_file(index_data, save_dir)


def load_sequence_dict(filepath):
    print(f"Loading sequence dictionary from {filepath}...")
    try:
        with open(filepath, "rb") as f:
            seq_lookup = pickle.load(f)
        print(f"Successfully loaded seq lookup with {len(seq_lookup)} entries")
        return seq_lookup
    except Exception as e:
        print(f"An error occurred while loading the sequence dictionary: {str(e)}")
        print("Continuing without sequence lookup. Sequences will be empty.")
        return {}

def get_sequence(acc: str, seq_lookup: Dict[str, str]) -> Optional[str]:
    return seq_lookup.get(acc)

def main(go_tsv_path: str, save_dir: str):
    t0 = time.time()
    sequence_dict_pickle_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/afdb_sequence_dict.pkl"
    
    # Load the sequence dictionary
    seq_lookup = load_sequence_dict(sequence_dict_pickle_path)

    print("Reading GO TSV file...")
    go_dict = read_go_tsv(go_tsv_path)

    print("Creating and saving GO documents...")
    create_and_save_go_documents(go_dict, save_dir, 300, seq_lookup)

    t1 = time.time()
    print(f"Total processing time: {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("go_tsv_path", type=str, help="Path to the GO term TSV file")
    parser.add_argument("save_dir", type=str, help="Directory to save the parquet files")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir)