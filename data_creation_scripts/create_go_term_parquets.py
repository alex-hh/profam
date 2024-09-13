import argparse
import csv
import gzip
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lmdb
from tqdm import tqdm
import logging
import psutil
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Constants
RECORDS_PER_FILE = 100
BATCH_SIZE = 100
BUFFER_SIZE = 10000

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def stream_go_tsv(file_path):
    open_func = gzip.open if file_path.endswith('.gz') else open
    with open_func(file_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                go_term, uniprot_accs = parts
                yield go_term, uniprot_accs.split(',')

def setup_lmdb(lmdb_path):
    return lmdb.open(lmdb_path, readonly=True, lock=False, map_size=int(1e12))

def fetch_sequence(txn, acc):
    return txn.get(f"AFDB:AF-{acc}-F1".encode())

def process_batch(batch, txn):
    results = []
    failed_seqs = []
    successful_seqs = []
    for go_term, uniprot_accs in batch:
        sequences = []
        success_accs = []
        for acc in uniprot_accs:
            seq = fetch_sequence(txn, acc)
            if seq:
                sequences.append(seq)
                success_accs.append(acc)
                successful_seqs.append(acc)
            else:
                failed_seqs.append(acc)
        if sequences:
            results.append({
                'fam_id': go_term,
                'sequences': sequences,
                'accessions': success_accs
            })
    return results, failed_seqs, successful_seqs

def process_go_terms(go_tsv_path, lmdb_path, save_dir, file_prefix):
    os.makedirs(save_dir, exist_ok=True)
    env = setup_lmdb(lmdb_path)
    
    schema = pa.schema([
        ('fam_id', pa.string()),
        ('sequences', pa.list_(pa.binary())),
        ('accessions', pa.list_(pa.string()))
    ])
    
    file_counter = 0
    record_counter = 0
    writer = None
    index_data = []
    failed_seqs = []
    successful_seqs = []

    batch = []
    with env.begin(write=False) as txn:
        for go_term, uniprot_accs in tqdm(stream_go_tsv(go_tsv_path), desc="Processing GO terms"):
            batch.append((go_term, uniprot_accs))
            
            if len(batch) >= BATCH_SIZE:
                results, new_failed_seqs, new_successful_seqs = process_batch(batch, txn)
                failed_seqs.extend(new_failed_seqs)
                successful_seqs.extend(new_successful_seqs)
                
                for result in results:
                    if writer is None or record_counter >= RECORDS_PER_FILE:
                        if writer:
                            writer.close()
                        file_counter += 1
                        file_name = f"{file_prefix}_{file_counter}.parquet"
                        writer = pq.ParquetWriter(os.path.join(save_dir, file_name), schema)
                        record_counter = 0
                    
                    writer.write_table(pa.Table.from_pydict(result))
                    index_data.append({'fam_id': result['fam_id'], 'parquet_file': file_name})
                    record_counter += 1
                
                batch = []
                
                if len(index_data) >= BUFFER_SIZE:
                    pd.DataFrame(index_data).to_csv(os.path.join(save_dir, "go_term_index.csv"), mode='a', header=not os.path.exists(os.path.join(save_dir, "go_term_index.csv")), index=False)
                    index_data = []
        
        # Process remaining batch
        if batch:
            results, new_failed_seqs, new_successful_seqs = process_batch(batch, txn)
            failed_seqs.extend(new_failed_seqs)
            successful_seqs.extend(new_successful_seqs)
            for result in results:
                if writer is None or record_counter >= RECORDS_PER_FILE:
                    if writer:
                        writer.close()
                    file_counter += 1
                    file_name = f"{file_prefix}_{file_counter}.parquet"
                    writer = pq.ParquetWriter(os.path.join(save_dir, file_name), schema)
                    record_counter = 0
                
                writer.write_table(pa.Table.from_pydict(result))
                index_data.append({'fam_id': result['fam_id'], 'parquet_file': file_name})
                record_counter += 1

    if writer:
        writer.close()

    # Write remaining index data
    if index_data:
        pd.DataFrame(index_data).to_csv(os.path.join(save_dir, "go_term_index.csv"), mode='a', header=not os.path.exists(os.path.join(save_dir, "go_term_index.csv")), index=False)

    # Log failed sequences
    with open(os.path.join(save_dir, "failed_sequences.txt"), "w") as f:
        for seq in failed_seqs:
            f.write(f"{seq}\n")

    # Log successful sequences
    with open(os.path.join(save_dir, "successful_sequences.txt"), "w") as f:
        for seq in successful_seqs:
            f.write(f"{seq}\n")

    logging.info(f"Total Parquet files created: {file_counter}")
    logging.info(f"Total failed sequences: {len(failed_seqs)}")
    logging.info(f"Total successful sequences: {len(successful_seqs)}")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Current memory usage: {mem_info.rss / 1024 / 1024:.2f} MiB")

def main(go_tsv_path, save_dir, lmdb_path, file_prefix):
    start_time = pd.Timestamp.now()
    try:
        process_go_terms(go_tsv_path, lmdb_path, save_dir, file_prefix)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        end_time = pd.Timestamp.now()
        logging.info(f"Total processing time: {end_time - start_time}")
        log_memory_usage()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO term documents and save them as parquet files.")
    parser.add_argument("--go_tsv_path", type=str, required=True, help="Path to the GO term TSV file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the parquet files")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to the LMDB directory containing sequence data")
    parser.add_argument("--file_prefix", type=str, default="GO", help="Prefix for output Parquet files")
    args = parser.parse_args()

    main(args.go_tsv_path, args.save_dir, args.lmdb_path, args.file_prefix)