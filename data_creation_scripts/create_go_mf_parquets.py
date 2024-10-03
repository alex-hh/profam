import argparse
import gzip
import lmdb
import logging
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from collections import defaultdict

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_sequence(uniprot_id, txn):
    prefixes = [f'sp|{uniprot_id}|', f'tr|{uniprot_id}|']
    for key, value in txn.cursor():
        if any(key.startswith(prefix.encode('utf-8')) for prefix in prefixes):
            return value.decode('utf-8')
    return None

def count_uniprot_ids(input_path, min_ic):
    total_ids = 0
    with gzip.open(input_path, 'rt') as f:
        for line in tqdm(f, desc="Counting UniProt IDs"):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            _, info_content, uniprot_ids = parts
            if float(info_content) < min_ic:
                continue
            
            total_ids += len(uniprot_ids.split(','))
    
    return total_ids

def create_batches(input_path, min_ic, batch_size):
    batches = defaultdict(list)
    current_batch = 0
    current_batch_size = 0
    
    with gzip.open(input_path, 'rt') as f:
        for line in tqdm(f, desc="Creating batches"):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            fam_id, info_content, uniprot_ids = parts
            if float(info_content) < min_ic:
                continue
            
            for uid in uniprot_ids.split(','):
                batches[current_batch].append((fam_id, uid.strip()))
                current_batch_size += 1
                
                if current_batch_size >= batch_size:
                    current_batch += 1
                    current_batch_size = 0
    
    return batches

def process_batch(batch, txn, writers, num_parquet, schema):
    fam_sequences = defaultdict(lambda: {'sequences': [], 'accessions': []})
    
    for fam_id, uid in batch:
        seq = fetch_sequence(uid, txn)
        if seq:
            fam_sequences[fam_id]['sequences'].append(seq)
            fam_sequences[fam_id]['accessions'].append(uid)
    
    for fam_id, data in fam_sequences.items():
        if data['sequences']:
            parquet_index = hash(fam_id) % num_parquet
            record = pa.RecordBatch.from_arrays(
                [pa.array([fam_id]), pa.array([data['sequences']]), pa.array([data['accessions']])],
                schema=schema
            )
            writers[parquet_index].write_batch(record)

def process_file(input_path, output_dir, lmdb_path, num_parquet, min_ic, batch_size):
    schema = pa.schema([
        ('fam_id', pa.string()),
        ('sequences', pa.list_(pa.string())),
        ('accessions', pa.list_(pa.string()))
    ])
    
    writers = {i: pq.ParquetWriter(os.path.join(output_dir, f'go_mf_{i}.parquet'), schema, compression='snappy')
               for i in range(num_parquet)}
    
    total_ids = count_uniprot_ids(input_path, min_ic)
    logging.info(f"Total UniProt IDs to process: {total_ids}")
    
    batches = create_batches(input_path, min_ic, batch_size)
    logging.info(f"Created {len(batches)} batches")
    
    with lmdb.open(lmdb_path, readonly=True, lock=False) as env, \
         env.begin(write=False) as txn:
        
        for batch_num, batch in tqdm(batches.items(), desc="Processing batches"):
            process_batch(batch, txn, writers, num_parquet, schema)
    
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
    parser.add_argument('--batch_size', type=int, default=10000, help='Number of UniProt IDs to process in each batch.')
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Starting processing...")
    process_file(args.input, args.output_dir, args.lmdb_path, args.num_parquet, args.min_ic, args.batch_size)
    logging.info("Processing completed.")

if __name__ == "__main__":
    main()