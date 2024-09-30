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
from collections import defaultdict

def fetch_sequence(uniprot_id, db_env):
    """
    Fetches the protein sequence for a given UniProt ID from the LMDB database.

    Args:
        uniprot_id (str): The UniProt ID.
        db_env (lmdb.Environment): The LMDB environment.

    Returns:
        str or None: The protein sequence if found, else None.
    """
    try:
        with db_env.begin() as txn:
            # Construct the key in the correct format
            key = f'AFDB:AF-{uniprot_id}-F1'.encode('utf-8')
            seq = txn.get(key)
            if seq is not None:
                # The value is already in bytes, so we just need to decode it
                return seq.decode('utf-8')
            else:
                return None
    except lmdb.Error as e:
        logging.error(f"LMDB error while fetching {uniprot_id}: {e}")
        return None

def setup_logging():
    """
    Sets up the logging configuration.
    """
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
    """
    Assigns a fam_id to a parquet file based on its hash.

    Args:
        fam_id (str): The GO MF functional annotation ID.
        num_parquet (int): Total number of parquet files.

    Returns:
        int: The index of the parquet file.
    """
    return int(hashlib.md5(fam_id.encode('utf-8')).hexdigest(), 16) % num_parquet

def write_parquet(writers, output_dir):
    for i, records in writers.items():
        if records:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            parquet_path = os.path.join(output_dir, f'output_{i}.parquet')
            try:
                pq.write_table(table, parquet_path, append=True)
                logging.info(f"Wrote {len(records)} records to {parquet_path}")
            except Exception as e:
                logging.error(f"Failed to write to {parquet_path}: {e}")

def create_go_term_mapping_csv(mapping, output_dir):
    """
    Creates a CSV file mapping GO terms to their corresponding parquet files.

    Args:
        mapping (dict): A dictionary mapping parquet file indices to lists of GO terms.
        output_dir (str): Directory to store the output CSV file.
    """
    csv_path = os.path.join(output_dir, 'go_term_parquet_mapping.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['GO_Term', 'Parquet_File'])
        for parquet_index, go_terms in mapping.items():
            for go_term in go_terms:
                writer.writerow([go_term, f'output_{parquet_index}.parquet'])
    logging.info(f"Created GO term to parquet file mapping: {csv_path}")

def process_file(input_path, output_dir, db_env, num_parquet):
    """
    Processes the input TSV file and writes the data into parquet files.

    Args:
        input_path (str): Path to the input gzipped TSV file.
        output_dir (str): Directory to store output parquet files.
        db_env (lmdb.Environment): The LMDB environment.
        num_parquet (int): Number of parquet files to generate.
    """
    # Initialize writers dictionary and GO term mapping
    BATCH_SIZE = 10000  # Adjust based on available memory
    writers = {i: [] for i in range(num_parquet)}
    go_term_mapping = defaultdict(set)

    success_count = 0
    failure_count = 0

    # Count total UniProt IDs
    total_uniprot_ids = 0
    with gzip.open(input_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                total_uniprot_ids += len(parts[2].split(','))

    # Process file with progress bar
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
            uniprot_ids_list = [uid.strip() for uid in uniprot_ids.split(',') if uid.strip()]

            sequences = []
            accessions = []
            for uid in uniprot_ids_list:
                seq = fetch_sequence(uid, db_env)
                if seq:
                    sequences.append(seq)
                    accessions.append(uid)
                    success_count += 1
                    pbar.postfix[0] += 1
                else:
                    logging.error(f"Failed to fetch sequence for UniProt ID: {uid}")
                    failure_count += 1
                    pbar.postfix[1] += 1
                pbar.update(1)

            if sequences:
                parquet_index = assign_parquet_file(fam_id, num_parquet)
                writers[parquet_index].append({
                    'fam_id': fam_id,
                    'sequences': sequences,
                    'accessions': accessions
                })
                go_term_mapping[parquet_index].add(fam_id)

            if len(writers[parquet_index]) >= BATCH_SIZE:
                write_parquet(writers, output_dir)
                writers = {i: [] for i in range(num_parquet)}

        pbar.close()

    # Write any remaining records
    write_parquet(writers, output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Successfully fetched {success_count} sequences.")
    logging.info(f"Failed to fetch {failure_count} sequences.")

    # Create the GO term to parquet file mapping CSV
    create_go_term_mapping_csv(go_term_mapping, output_dir)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process GO term data and generate parquet files.')
    parser.add_argument('--input', required=True, help='Path to input gzipped TSV file.')
    parser.add_argument('--output_dir', default='/SAN/orengolab/cath_plm/ProFam/data/GO_MF/mf_parquets', help='Directory to store output parquet files.')
    parser.add_argument('--lmdb_path', default='/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences_dict.lmdb', help='Path to LMDB database.')
    parser.add_argument('--num_parquet', type=int, default=100, help='Number of parquet files to generate.')
    return parser.parse_args()

def main():
    """
    The main function orchestrating the processing of the TSV file.
    """
    args = parse_arguments()
    setup_logging()

    db_env = None
    try:
        db_env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        process_file(args.input, args.output_dir, db_env, args.num_parquet)
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
    finally:
        if db_env:
            db_env.close()
        logging.info("Processing completed.")

if __name__ == "__main__":
    main()