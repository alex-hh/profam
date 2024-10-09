import os
import glob
import json
import requests
import sys
import time
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
"""
Iterates through all of the Pfam parquet files and
creates a mapping that maps from the sequence name 
(which is in the 'accession' column) to the UniProt accession.
"""

API_URL = "https://rest.uniprot.org"

def extract_uniprotkb_ids(sequence_names):
    uniprotkb_ids = []
    for name in sequence_names:
        id_part = name.split('/')[0]  # Extract before '/'
        uniprotkb_id = id_part.strip()
        uniprotkb_ids.append(uniprotkb_id)
    return uniprotkb_ids


def check_job_status(job_id):
    status_url = f"{API_URL}/idmapping/status/{job_id}"
    fail_counter = 0
    while True:
        response = requests.get(status_url, allow_redirects=False)
        response.raise_for_status()

        if response.status_code == 303:
            # Job is finished
            return True
        else:
            status = response.json()
            job_status = status.get('jobStatus')
            if job_status in ('RUNNING', 'NEW'):
                print(f"Job is {job_status}...")
                time.sleep(20)
                fail_counter = 0
            else:
                fail_counter += 1
                time.sleep(20)
            if fail_counter > 12:
                print(f"Job failed with status: {job_status}")
                return False

def get_uniprot_accessions_from_names(pfam_df, save_path):
    sequence_names = pfam_df['sequence_name'].tolist()
    uniprotkb_ids = extract_uniprotkb_ids(sequence_names)

    # Split IDs into chunks of up to 100,000 IDs
    chunk_size = 100000
    id_chunks = [uniprotkb_ids[i:i + chunk_size] for i in range(0, len(uniprotkb_ids), chunk_size)]

    all_mappings = []

    for idx, ids_chunk in enumerate(id_chunks):
        print(f"Processing chunk {idx+1}/{len(id_chunks)} with {len(ids_chunk)} IDs")
        job_id = submit_id_mapping('UniProtKB_AC-ID', 'UniProtKB', ids_chunk)
        print(f"Submitted job with ID: {job_id}")
        if check_job_status(job_id):
            print("Job finished. Retrieving results...")
            results_text = get_results(job_id)
            mapping_df = process_results(results_text)
            all_mappings.append(mapping_df)
        else:
            print(f"Job {job_id} failed.")

    # Combine all mappings
    mappings_df = pd.concat(all_mappings, ignore_index=True)
    pfam_df['join_id'] = pfam_df['sequence_name'].apply(lambda x: x.split("/")[0])

    merged_df = pfam_df.merge(mappings_df, left_on='join_id', right_on='From', how='left')
    merged_df = merged_df[['sequence_name', 'family_accession', 'aligned_sequence', 'sequence', 'Entry', 'Length']]
    # Save the merged data
    merged_df.to_csv(save_path, index=False)
    print(f"Mapping completed and results saved to {save_path}.")
    return merged_df


def submit_id_mapping(from_db, to_db, ids):
    url = f"{API_URL}/idmapping/run"
    params = {'from': from_db, 'to': to_db, 'ids': ','.join(ids)}
    response = requests.post(url, data=params)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Error: {response.status_code} - {response.text}")
        raise err
    return response.json()['jobId']

def get_results_url(job_id):
    status_url = f"{API_URL}/idmapping/status/{job_id}"
    response = requests.get(status_url, allow_redirects=False)
    if response.status_code == 303:
        redirect_url = response.headers['Location']
        # Modify the URL to point to the streaming endpoint
        if '/results/' in redirect_url:
            results_url = redirect_url.replace('/results/', '/results/stream/')
        else:
            results_url = redirect_url + '/stream'
        return results_url
    else:
        raise Exception("Results are not ready yet.")

def get_results(job_id):
    results_url = get_results_url(job_id)
    params = {'format': 'tsv'}
    response = requests.get(results_url, params=params, stream=True)
    response.raise_for_status()
    results_text = ''
    for chunk in response.iter_content(chunk_size=1024):
        results_text += chunk.decode('utf-8')
    return results_text


def process_results(results_text):
    return pd.read_csv(StringIO(results_text), sep='\t')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_sequence_name_to_uniprot_mapping(sequence_names):
    sequence_to_uniprot = {}
    uniprotkb_ids = extract_uniprotkb_ids(sequence_names)
    id_mapping = dict(zip(sequence_names, uniprotkb_ids))
    unique_ids = list(set(uniprotkb_ids))
    logging.info(f"Total unique sequence IDs to map: {len(unique_ids)}")

    # Split IDs into chunks of up to 100,000 IDs
    chunk_size = 100000
    id_chunks = [unique_ids[i:i + chunk_size] for i in range(0, len(unique_ids), chunk_size)]

    all_mappings = []
    failed_ids = []

    for idx, ids_chunk in enumerate(id_chunks):
        chunk_file = f'mapping_chunk_{idx+1}.csv'
        if os.path.exists(chunk_file):
            logging.info(f"Skipping chunk {idx+1}/{len(id_chunks)} because it already exists")
            continue
        logging.info(f"Processing chunk {idx+1}/{len(id_chunks)} with {len(ids_chunk)} IDs")
        try:
            job_id = submit_id_mapping('UniProtKB_AC-ID', 'UniProtKB', ids_chunk)
            logging.info(f"Submitted job with ID: {job_id}")
            if check_job_status(job_id):
                logging.info("Job finished. Retrieving results...")
                results_text = get_results(job_id)
                mapping_df = process_results(results_text)
                all_mappings.append(mapping_df)
                
                # Save mapping data to disk after each chunk
                mapping_df.to_csv(chunk_file, index=False)
                logging.info(f"Saved mapping data for chunk {idx+1} to {chunk_file}")
            else:
                logging.error(f"Job {job_id} failed.")
                failed_ids.extend(ids_chunk)
        except Exception as e:
            logging.error(f"An exception occurred while processing chunk {idx+1}: {e}")
            failed_ids.extend(ids_chunk)
            continue

    if all_mappings:
        mappings_df = pd.concat(all_mappings, ignore_index=True)
        mappings_df = mappings_df[['From', 'Entry']]
        from_to_entry = dict(zip(mappings_df['From'], mappings_df['Entry']))
        # Build sequence_name to uniprot accession mapping
        for seq_name, id_part in id_mapping.items():
            uniprot_acc = from_to_entry.get(id_part)
            sequence_to_uniprot[seq_name] = uniprot_acc
    else:
        logging.warning("No mappings were retrieved from UniProt.")

    # Save failed IDs to disk
    if failed_ids:
        failed_ids_path = 'failed_ids.txt'
        logging.info(f"Saving failed IDs to {failed_ids_path}")
        with open(failed_ids_path, 'w') as f:
            for fid in failed_ids:
                f.write(f'{fid}\n')

    return sequence_to_uniprot

def process_parquet_files(parq_paths, sequence_to_uniprot_mapping, output_dir):
    total_sequences = 0
    unmatched_sequences = 0

    for parq_path in parq_paths:
        logging.info(f"Processing parquet file: {parq_path}")
        df = pd.read_parquet(parq_path)
        # Assuming 'accessions' column contains list of sequence names in the family
        if 'accessions' in df.columns:
            df['family_uniprot_accessions'] = df['accessions'].apply(
                lambda seq_list: np.array([
                    sequence_to_uniprot_mapping.get(seq_name)
                    for seq_name in seq_list
                    if sequence_to_uniprot_mapping.get(seq_name) is not None
                ])
            )
            total = df['accessions'].apply(len).sum()
            unmatched = df['accessions'].apply(
                lambda seq_list: sum(1 for seq_name in seq_list if sequence_to_uniprot_mapping.get(seq_name) is None)
            ).sum()
        else:
            df['family_uniprot_accessions'] = None
            total = 0
            unmatched = 0

        unmatched_sequences += unmatched
        total_sequences += total
        logging.info(f"Unmatched sequences in this file: {unmatched}/{total}")

        # Save the updated parquet file
        output_path = os.path.join(output_dir, os.path.relpath(parq_path, start=pfam_parquet_dir))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(parq_path, index=False)
        logging.info(f"Overwritten parquet file: {parq_path}")

    # Log the proportion of names which cannot be matched
    if total_sequences > 0:
        proportion_unmatched = unmatched_sequences / total_sequences
        logging.info(f"Total unmatched sequences: {unmatched_sequences}/{total_sequences}")
        logging.info(f"Proportion of unmatched sequences: {proportion_unmatched}")
    else:
        logging.warning("No sequences were processed.")

if __name__=="__main__":
    setup_logging()
    all_sequence_names = set()
    pfam_seq_mapping_path = "../data/pfam/sequence_name_to_uniprot_mapping.csv"
    pfam_parquet_dir = "../data/pfam/train_test_split_parquets"
    output_parquet_dir = "../data/pfam/train_test_split_parquets_with_uniprot"

    parq_paths = glob.glob(os.path.join(pfam_parquet_dir, "**/*.parquet"), recursive=True)

    if not os.path.exists(pfam_seq_mapping_path):
        logging.info("Collecting all sequence names from parquet files...")
        for parq_path in parq_paths:
            logging.info(f"Reading parquet file: {parq_path}")
            df = pd.read_parquet(parq_path)
            sequence_names = df['accessions'].explode().explode().unique().tolist()
            all_sequence_names.update(sequence_names)
        logging.info(f"Total unique sequence names collected: {len(all_sequence_names)}")

        logging.info("Mapping sequence names to UniProt accessions...")
        sequence_to_uniprot_mapping = get_sequence_name_to_uniprot_mapping(list(all_sequence_names))

        # Save mapping to CSV
        logging.info(f"Saving sequence name to UniProt accession mapping to {pfam_seq_mapping_path}")
        mapping_df = pd.DataFrame([
            {'sequence_name': k, 'uniprot_accession': v}
            for k, v in sequence_to_uniprot_mapping.items()
        ])
        mapping_df.to_csv(pfam_seq_mapping_path, index=False)
    else:
        logging.info(f"Loading existing sequence name to UniProt accession mapping from {pfam_seq_mapping_path}")
        mapping_df = pd.read_csv(pfam_seq_mapping_path)
        sequence_to_uniprot_mapping = dict(zip(mapping_df['sequence_name'], mapping_df['uniprot_accession']))

    # Process parquet files to add uniprot accessions
    logging.info("Adding UniProt accessions to parquet files...")
    process_parquet_files(parq_paths, sequence_to_uniprot_mapping, output_parquet_dir)