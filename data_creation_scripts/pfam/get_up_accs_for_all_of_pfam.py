import logging
import os
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

"""
Iterates through all of the Pfam parquet files and
creates a mapping that maps from the sequence name 
(which is in the 'accession' column) to the UniProt accession.
"""

API_URL = "https://rest.uniprot.org"
IS_DEBUGGING = "judewells" in os.getcwd()

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
        time.sleep(20)
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
                fail_counter = 0
            else:
                fail_counter += 1
            if fail_counter > 12:
                print(f"Job failed with status: {job_status}")
                return False

def map_val_test_names_to_accessions(pfam_df, save_path):
    """
    Kept as a separate function from get_name_to_accession_mapping
    because it is used in the val/test family ID selection pipeline
    """
    sequence_names = pfam_df['sequence_name'].tolist()
    uniprotkb_ids = extract_uniprotkb_ids(sequence_names)

    # Split IDs into chunks of up to 100,000 IDs
    chunk_size = 99000
    id_chunks = [uniprotkb_ids[i:i + chunk_size] for i in range(0, len(uniprotkb_ids), chunk_size)]

    all_mappings = []

    for idx, ids_chunk in enumerate(id_chunks):
        if idx > 2 and IS_DEBUGGING:
            break
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

    merged_df = pfam_df.merge(mappings_df, left_on='join_id', right_on='sequence_name', how='left')
    print(merged_df.columns)
    print(merged_df.head())
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
    time.sleep(5)
    params = {'format': 'tsv'}
    response = requests.get(results_url, params=params, stream=True)
    response.raise_for_status()
    results_text = ''
    for chunk in response.iter_content(chunk_size=1024):
        results_text += chunk.decode('utf-8')
    time.sleep(5)
    return results_text
    
def process_results(results_text):
    mapping_df = pd.read_csv(StringIO(results_text), sep='\t')
    mapping_df = mapping_df[['From', 'Entry']]
    mapping_df.columns = ['sequence_name', 'matched_accession']
    return mapping_df

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_name_to_accession_mapping(sequence_names, map_save_dir):
    """
    Maps sequence names to UniProt accession numbers.

    This function takes a list of sequence names and attempts to map them to their corresponding
    UniProt accession numbers. It processes the sequence names in chunks, submits them to the
    UniProt ID mapping service, and retrieves the results. The function handles potential errors
    and saves intermediate results to disk.
    

    Args:
        sequence_names (list): A list of sequence names to be mapped.
        map_save_dir (str): The directory path where mapping results will be saved.

    Returns:
        dict: A dictionary mapping sequence names to their corresponding UniProt accession numbers.

    The function performs the following steps:
    1. Extracts UniProtKB IDs from the sequence names.
    2. Splits the IDs into chunks for processing.
    3. For each chunk:
       - Submits an ID mapping job to UniProt.
       - Retrieves and processes the results.
       - Saves the chunk results to a CSV file.
    4. Combines all chunk results.
    5. Builds the final mapping of sequence names to UniProt accessions.
    6. Saves any failed IDs to a separate file.
    """
    n_unique_ids = len(sequence_names)
    logging.info(f"Total unique sequence IDs to map: {n_unique_ids}")
    failed_id_path = f"{map_save_dir}/failed_ids.csv"
    # Split IDs into chunks of up to 100,000 IDs
    chunk_size = 99000
    id_chunks = [sequence_names[i:i + chunk_size] for i in range(0, n_unique_ids, chunk_size)]
    failed_ids = []
    name_to_accession_mapping = {}
    for idx, ids_chunk in enumerate(id_chunks):
        if idx > 2 and IS_DEBUGGING:
            break
        chunk_file = f'{map_save_dir}/mapping_chunk_{idx+1}.csv'
        if os.path.exists(chunk_file):
            logging.info(f"Chunk {idx+1}/{len(id_chunks)} already exists")
            mapping_df = pd.read_csv(chunk_file)
            name_to_accession_mapping.update(dict(zip(mapping_df['sequence_name'], mapping_df['matched_accession'])))
        else:
            logging.info(f"Processing chunk {idx+1}/{len(id_chunks)} with {len(ids_chunk)} IDs")
            try:
                job_id = submit_id_mapping('UniProtKB_AC-ID', 'UniProtKB', ids_chunk)
                logging.info(f"Submitted job with ID: {job_id}")
                if check_job_status(job_id):
                    logging.info("Job finished. Retrieving results...")
                    results_text = get_results(job_id)
                    mapping_df = process_results(results_text)
                    # Save mapping data to disk after each chunk
                    mapping_df.to_csv(chunk_file, index=False)
                    name_to_accession_mapping.update(
                        dict(zip(mapping_df['sequence_name'], mapping_df['matched_accession']))
                    )
                    failed_this_chunk = set(ids_chunk) - set(mapping_df['sequence_name'])
                    if failed_this_chunk:
                        failed_ids.extend(failed_this_chunk)
                    logging.info(f"Saved mapping data for chunk {idx+1} to {chunk_file}")
                else:
                    logging.error(f"Job {job_id} failed.")
                    failed_ids.extend(ids_chunk)
            except Exception as e:
                logging.error(f"An exception occurred while processing chunk {idx+1}: {e}")
                failed_ids.extend(ids_chunk)
                continue
        if failed_ids:
            with open(failed_id_path, 'a') as f:
                f.write('\n'.join(failed_ids) + '\n')
        failed_ids = []
    print(f"Mapping completed and results saved to {map_save_dir}.")
    return name_to_accession_mapping


def process_parquet_files(parq_paths, name_to_accession_mapping):
    total_sequences = 0
    unmatched_sequences = 0

    for parq_path in parq_paths:
        logging.info(f"Processing parquet file: {parq_path}")
        df = pd.read_parquet(parq_path)
        # Assuming 'accessions' column contains list of sequence names in the family
        df['matched_accessions'] = df['accessions'].apply(
            lambda seq_list: np.array([
                name_to_accession_mapping.get(seq_name.split("/")[0], None)
                for seq_name in seq_list
            ])
        )
        total = df['accessions'].apply(len).sum()
        unmatched = df['accessions'].apply(
            lambda seq_ls: sum(
                1 for seq_name in seq_ls if name_to_accession_mapping.get(seq_name.split("/")[0]) is None)
        ).sum()

        unmatched_sequences += unmatched
        total_sequences += total
        logging.info(f"Unmatched sequences in this file: {unmatched}/{total}")

        df.to_parquet(parq_path, index=False)
        logging.info(f"Overwritten parquet file: {parq_path}")

    # Log the proportion of names which cannot be matched
    if total_sequences > 0:
        proportion_unmatched = unmatched_sequences / total_sequences
        logging.info(f"Total unmatched sequences: {unmatched_sequences}/{total_sequences}")
        logging.info(f"Proportion of unmatched sequences: {proportion_unmatched}")
    else:
        logging.warning("No sequences were processed.")
