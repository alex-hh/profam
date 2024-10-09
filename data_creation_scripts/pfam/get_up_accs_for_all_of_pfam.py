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
from data_creation_scripts.pfam.consolidated_create_pfam_val_test import (
    extract_uniprotkb_ids,
    submit_id_mapping,
    check_job_status,
    get_results_url,
    get_results,
    process_results
)

"""
Iterates through all of the Pfam parquet files and
creates a mapping that maps from the sequence name 
(which is in the 'accession' column) to the UniProt accession.
"""

API_URL = "https://rest.uniprot.org"

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
        logging.info(f"Processing chunk {idx+1}/{len(id_chunks)} with {len(ids_chunk)} IDs")
        try:
            job_id = submit_id_mapping('UniProtKB_AC-ID', 'UniProtKB', ids_chunk)
            logging.info(f"Submitted job with ID: {job_id}")
            if check_job_status(job_id):
                logging.info("Job finished. Retrieving results...")
                results_text = get_results(job_id)
                mapping_df = process_results(results_text)
                all_mappings.append(mapping_df)
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
        df.to_parquet(output_path)
        logging.info(f"Saved updated parquet file to {output_path}")

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