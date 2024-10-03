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
    schema = pa.schema([
        ('fam_id', pa.string()),
        ('sequences', pa.list_(pa.string())),
        ('accessions', pa.list_(pa.string()))
    ])
    
    writers = {i: pq.ParquetWriter(os.path.join(output_dir, f'go_mf_{i}.parquet'), schema, compression='snappy')
               for i in range(num_parquet)}
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    try:
        with env.begin(write=False) as txn, gzip.open(input_path, 'rt') as f:
            # Count valid lines first
            valid_lines = sum(1 for line in f if len(line.strip().split('\t')) == 3 and float(line.strip().split('\t')[1]) >= min_ic)
            
            # Reset file pointer to the beginning
            f.seek(0)
            
            for line in tqdm(f, total=valid_lines, desc="Processing GO terms"):
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
                    record = pa.RecordBatch.from_arrays(
                        [pa.array([fam_id]), pa.array([sequences]), pa.array([accessions])],
                        schema=schema
                    )
                    writers[parquet_index].write_batch(record)
    finally:
        env.close()
        for writer in writers.values():
            writer.close()

    logging.info("Parquet files written successfully.")

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