import argparse
import gzip
import lmdb
import logging
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os
from collections import defaultdict

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_sequence(uniprot_id, txn):
    prefixes = [b'sp|', b'tr|']
    for prefix in prefixes:
        partial_key = prefix + uniprot_id.encode('utf-8')
        cursor = txn.cursor()
        if cursor.set_range(partial_key):
            key, value = cursor.item()
            if key.startswith(partial_key):
                return value.decode('utf-8')
    return None

def create_batches_and_count(input_path, min_ic, batch_size):
    batches = defaultdict(list)
    current_batch = 0
    current_batch_size = 0
    total_ids = 0
    
    with gzip.open(input_path, 'rt') as f:
        for line in tqdm(f, desc="Creating batches and counting UniProt IDs"):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            fam_id, info_content, uniprot_ids = parts
            if float(info_content) < min_ic:
                continue
            
            uids = [uid.strip() for uid in uniprot_ids.split(',')]
            total_ids += len(uids)
            
            for uid in uids:
                batches[current_batch].append((fam_id, uid))
                current_batch_size += 1
                
                if current_batch_size >= batch_size:
                    current_batch += 1
                    current_batch_size = 0
    
    return batches, total_ids

def process_batch(batch, txn, writers, num_parquet, schema, seq_cache):
    fam_sequences = defaultdict(lambda: {'sequences': [], 'accessions': []})
    sequences_found = 0
    sequences_not_found = 0
    
    for fam_id, uid in batch:
        if uid in seq_cache:
            seq = seq_cache[uid]
        else:
            seq = fetch_sequence(uid, txn)
            seq_cache[uid] = seq
        
        if seq:
            fam_sequences[fam_id]['sequences'].append(seq)
            fam_sequences[fam_id]['accessions'].append(uid)
            sequences_found += 1
        else:
            sequences_not_found += 1

    for fam_id, data in fam_sequences.items():
        if data['sequences']:
            parquet_index = hash(fam_id) % num_parquet
            record = pa.RecordBatch.from_arrays(
                [pa.array([fam_id]), pa.array([data['sequences']]), pa.array([data['accessions']])],
                schema=schema
            )
            writers[parquet_index].write_batch(record)
    
    return sequences_found, sequences_not_found

def process_file(input_path, output_dir, lmdb_path, num_parquet, min_ic, batch_size):
    schema = pa.schema([
        ('fam_id', pa.string()),
        ('sequences', pa.list_(pa.string())),
        ('accessions', pa.list_(pa.string()))
    ])
    
    writers = {i: pq.ParquetWriter(os.path.join(output_dir, f'go_mf_{i}.parquet'), schema, compression='snappy')
               for i in range(num_parquet)}
    
    batches, total_ids = create_batches_and_count(input_path, min_ic, batch_size)
    logging.info(f"Total UniProt IDs to process: {total_ids}")
    logging.info(f"Created {len(batches)} batches")
    
    total_found = 0
    total_not_found = 0
    
    with lmdb.open(lmdb_path, readonly=True, lock=False) as env, \
         env.begin(write=False) as txn:
        
        for batch_num, batch in tqdm(batches.items(), desc="Processing batches"):
            found, not_found = process_batch(batch, txn, writers, num_parquet, schema, seq_cache={})
            total_found += found
            total_not_found += not_found
    
    for writer in writers.values():
        writer.close()

    logging.info("Parquet files written successfully.")
    logging.info(f"Total sequences found: {total_found}")
    logging.info(f"Total sequences not found: {total_not_found}")

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