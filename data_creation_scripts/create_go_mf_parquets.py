import argparse
import gzip
import lmdb
import logging
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import hashlib
import csv
from collections import defaultdict, Counter

def fetch_sequence(uniprot_id, db_env):
    """Fetches the protein sequence for a given UniProt ID from the LMDB database."""
    try:
        with db_env.begin() as txn:
            seq = txn.get(uniprot_id.encode('utf-8'))
            return seq.decode('utf-8') if seq else None
    except lmdb.Error as e:
        logging.error(f"LMDB error while fetching {uniprot_id}: {e}")
        return None

def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        filename='processing.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def assign_parquet_file(fam_id, num_parquet):
    """Assigns a fam_id to a parquet file based on its hash."""
    return int(hashlib.md5(fam_id.encode()).hexdigest(), 16) % num_parquet

def write_parquet(writers, output_dir):
    """Writes records to parquet files."""
    for i, records in writers.items():
        if records:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            parquet_path = os.path.join(output_dir, f'go_mf_{i}.parquet')
            try:
                pq.write_table(table, parquet_path, append=True)
                logging.info(f"Wrote {len(records)} records to {parquet_path}")
            except Exception as e:
                logging.error(f"Failed to write to {parquet_path}: {e}")

def create_go_term_mapping_csv(mapping, output_dir):
    """Creates a CSV file mapping GO terms to their corresponding parquet files."""
    csv_path = os.path.join(output_dir, 'go_parquet_mapping.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['GO_Term', 'Parquet_File'])
        writer.writeheader()
        for parquet_index, go_terms in mapping.items():
            for go_term in go_terms:
                writer.writerow({'GO_Term': go_term, 'Parquet_File': f'go_mf_{parquet_index}.parquet'})
    logging.info(f"Created GO term to parquet file mapping: {csv_path}")

def filter_go_terms(input_path, max_uniprot_ids=100000):
    """Filters GO terms with more than max_uniprot_ids UniProt IDs."""
    filtered_go_terms = set()
    total_uniprot_ids = 0
    with gzip.open(input_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                fam_id, _, uniprot_ids = parts
                uniprot_count = len(uniprot_ids.split(','))
                if uniprot_count > max_uniprot_ids:
                    filtered_go_terms.add(fam_id)
                else:
                    total_uniprot_ids += uniprot_count
    filtered_count = len(filtered_go_terms)
    logging.info(f"Filtered out {filtered_count} GO terms with more than {max_uniprot_ids} UniProt IDs.")
    return filtered_go_terms, total_uniprot_ids

def process_file(input_path, output_dir, db_env, num_parquet, filtered_go_terms, total_uniprot_ids, batch_size):
    """Processes the input TSV file and writes the data into parquet files."""
    writers = {i: [] for i in range(num_parquet)}
    go_term_mapping = defaultdict(set)

    success_count = 0
    failure_count = 0

    with gzip.open(input_path, 'rt') as f:
        pbar = tqdm(total=total_uniprot_ids, desc='Processing UniProt IDs', 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                               '[Successful: {postfix[0]}, Failed: {postfix[1]}]')
        pbar.postfix = [0, 0]

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                logging.warning(f"Skipping malformed line: {line.strip()}")
                continue
            fam_id, info_content, uniprot_ids = parts
            
            # Skip GO terms with too many UniProt IDs
            if fam_id in filtered_go_terms:
                continue

            uniprot_ids_list = [uid.strip() for uid in uniprot_ids.split(',') if uid.strip()]

            sequences = []
            accessions = []
            counts = Counter()
            for uid in uniprot_ids_list:
                seq = fetch_sequence(uid, db_env)
                if seq:
                    sequences.append(seq)
                    accessions.append(uid)
                    counts['success'] += 1
                    pbar.postfix[0] = counts['success']
                else:
                    logging.error(f"Failed to fetch sequence for UniProt ID: {uid}")
                    counts['failure'] += 1
                    pbar.postfix[1] = counts['failure']
                pbar.update(1)

            if sequences:
                parquet_index = assign_parquet_file(fam_id, num_parquet)
                writers[parquet_index].append({
                    'fam_id': fam_id,
                    'sequences': sequences,
                    'accessions': accessions
                })
                go_term_mapping[parquet_index].add(fam_id)

            if len(writers[parquet_index]) >= batch_size:
                write_parquet(writers, output_dir)
                writers = {i: [] for i in range(num_parquet)}

        pbar.close()

    # Write any remaining records
    write_parquet(writers, output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Successfully fetched {counts['success']} sequences.")
    logging.info(f"Failed to fetch {counts['failure']} sequences.")

    # Create the GO term to parquet file mapping CSV
    create_go_term_mapping_csv(go_term_mapping, output_dir)

    # Write filtered GO terms to a CSV file
    filtered_go_terms_path = os.path.join(output_dir, 'filtered_go_terms.csv')
    with open(filtered_go_terms_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filtered_GO_Term'])
        for go_term in filtered_go_terms:
            writer.writerow([go_term])
    logging.info(f"Created filtered GO terms list: {filtered_go_terms_path}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Process GO term data and generate parquet files.')
    parser.add_argument('--input', required=True, help='Path to input gzipped TSV file.')
    parser.add_argument('--output_dir', default='/SAN/orengolab/cath_plm/ProFam/data/GO_MF/mf_parquets', help='Directory to store output parquet files.')
    parser.add_argument('--lmdb_path', default='/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences_dict.lmdb', help='Path to LMDB database.')
    parser.add_argument('--num_parquet', type=int, default=100, help='Number of parquet files to generate.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (e.g., INFO, DEBUG).')
    parser.add_argument('--max_uniprot_ids', type=int, default=100000, help='Maximum number of UniProt IDs per GO term.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for writing records to parquet files.')
    return parser.parse_args()

def main():
    """Orchestrates the processing of the TSV file."""
    args = parse_arguments()
    setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)
    
    with lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False) as db_env:
        filtered_go_terms, total_uniprot_ids = filter_go_terms(args.input, args.max_uniprot_ids)
        process_file(args.input, args.output_dir, db_env, args.num_parquet, filtered_go_terms, total_uniprot_ids, args.batch_size)
    
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()