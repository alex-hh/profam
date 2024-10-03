import argparse
import gzip
import lmdb
import logging
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os
import hashlib
import numpy as np

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_sequence(uniprot_id, txn):
    prefixes = [f'sp|{uniprot_id}|'.encode('utf-8'), f'tr|{uniprot_id}|'.encode('utf-8')]
    for prefix in prefixes:
        cursor = txn.cursor()
        cursor.set_range(prefix)
        for key, value in cursor:
            if key.startswith(prefix):
                return value.decode('utf-8')
    return None

def assign_parquet_file(fam_id, num_parquet):
    return int(hashlib.md5(fam_id.encode()).hexdigest(), 16) % num_parquet

def process_file(input_path, output_dir, lmdb_path, num_parquet, min_ic):
    writers = {i: {} for i in range(num_parquet)}
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    try:
        with env.begin(write=False) as txn, gzip.open(input_path, 'rt') as f:
            for line in tqdm(f, desc="Processing GO terms"):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                
                fam_id, info_content, uniprot_ids = parts
                if float(info_content) < min_ic:
                    continue

                sequences = []
                accessions = []
                for uid in uniprot_ids.split(','):
                    seq = fetch_sequence(uid.strip(), txn)
                    if seq:
                        sequences.append(seq)
                        accessions.append(uid.strip())

                if sequences:
                    parquet_index = assign_parquet_file(fam_id, num_parquet)
                    if fam_id not in writers[parquet_index]:
                        writers[parquet_index][fam_id] = {'sequences': [], 'accessions': []}
                    writers[parquet_index][fam_id]['sequences'].extend(sequences)
                    writers[parquet_index][fam_id]['accessions'].extend(accessions)
    finally:
        env.close()

    logging.info("Writing parquet files...")
    for i, fam_data in tqdm(writers.items(), desc="Writing parquet files", total=num_parquet):
        if fam_data:
            records = [
                {
                    'fam_id': fam_id,
                    'sequences': np.array(data['sequences'], dtype=object),
                    'accessions': np.array(data['accessions'])
                }
                for fam_id, data in fam_data.items()
            ]
            table = pa.Table.from_pylist(records)
            pq.write_table(table, os.path.join(output_dir, f'go_mf_{i}.parquet'), compression='snappy')

def main():
    parser = argparse.ArgumentParser(description='Process GO term data and generate parquet files.')
    parser.add_argument('--input', required=True, help='Path to input gzipped TSV file.')
    parser.add_argument('--output_dir', required=True, help='Directory to store output parquet files.')
    parser.add_argument('--lmdb_path', required=True, help='Path to LMDB database.')
    parser.add_argument('--num_parquet', type=int, default=500, help='Number of parquet files to generate.')
    parser.add_argument('--min_ic', type=float, default=11, help='Minimum Information Content (IC) threshold for GO terms.')
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Starting processing...")
    process_file(args.input, args.output_dir, args.lmdb_path, args.num_parquet, args.min_ic)
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()