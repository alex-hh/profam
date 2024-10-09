import os
from io import StringIO
import pandas as pd
from collections import defaultdict
import random
import requests
import json
import glob
import sys
import time
import csv
from collections import defaultdict
from data_creation_scripts.pfam.shuffle_pfam_parquets import shuffle_pfam_parquets

"""
Consolidated script for Pfam data processing.

This script performs the following:

1. Selects Pfam families that occur in both train and test splits for both clustered and random splits, 
excluding families with more than 10,000 members or fewer than 10 members, 
and families where the UniProt IDs are not present in the provided JSON mapping.

2. Creates parquet files for the different splits

3. Removes validation and test families from the Pfam training data 
parquet files and splits the data into train, validation, and test sets.

4. Generates the 'pfam_val_test_all_up_ids.json' file mapping Pfam families to UniProt IDs.

Usage:
    python consolidated_pfam_processing.py
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

def sample_fams_by_size(
    pfam_select_fam_path,
    pfam_uniprot_json_path,
    n_families=1000,
    min_seq_len=50,
    min_famsize=10,
    max_famsize=10000,
):
    """
    Select families that occur in train AND test for BOTH clustered AND random splits,
    exclude families with more than 10,000 members or fewer than 10 members,
    take twice as many families as needed to account for filtering missing accessions.
    """
    random.seed(42)
    pfam_families = set()
    aggregated_fam_counts = {}
    seq_lens = {}
    pfam_uniprot_mappings = json.load(open(pfam_uniprot_json_path))
    n_families_oversample = n_families * 2 # Oversample to account for filtering missing accessions
    for split_type in ['clustered_split', 'random_split']:
        for split in ['train', 'test']:
            dfs = []
            split_dir = os.path.join(external_pfam_dir, split_type, split)
            for fname in sorted(os.listdir(split_dir)):
                if fname.startswith('data'):
                    split_fam = pd.read_csv(os.path.join(split_dir, fname))
                    split_fam['split'] = split
                    split_fam['split_type'] = split_type
                    split_fam['family_accession'] = split_fam['family_accession'].apply(lambda x: x.split(".")[0])
                    dfs.append(split_fam)
            combined = pd.concat(dfs)
            fam_counts = combined['family_accession'].value_counts().to_dict()
            if not aggregated_fam_counts:
                aggregated_fam_counts = fam_counts
            else:
                # Update counts to minimum across splits
                for fam_id in aggregated_fam_counts:
                    aggregated_fam_counts[fam_id] = min(aggregated_fam_counts[fam_id], fam_counts.get(fam_id, 0))
                for fam_id in combined['family_accession'].unique():
                    min_len = combined[combined['family_accession'] == fam_id].sequence.apply(len).min()
                    if fam_id not in seq_lens:
                        seq_lens[fam_id] = min_len
                    else:
                        seq_lens[fam_id] = min(seq_lens[fam_id], min_len)


            if len(pfam_families) == 0:
                pfam_families = set(combined['family_accession'].unique())
            else:
                pfam_families = pfam_families.intersection(
                    set(combined['family_accession'].unique())
                )
            print(f"Number of families in {split_type} {split}: {len(combined.family_accession.unique())}")

    print(f"Total number of families to sample from (before filtering): {len(pfam_families)}")
    pfam_families = set(f for f in pfam_families if f in pfam_uniprot_mappings)
    print(f"Families with UniProt mappings: {len(pfam_families)}")
    pfam_families = set(f for f in pfam_families if seq_lens[f] > min_seq_len)
    print(f"Families with sequences longer than {min_seq_len}: {len(pfam_families)}")

    filtered_families = []
    for fam_id in pfam_families:
        fam_size = len(pfam_uniprot_mappings.get(fam_id, []))
        if min_famsize <= fam_size <= max_famsize:
            filtered_families.append(fam_id)

    print(f"Total number of families after filtering on fam size: {len(filtered_families)}")

    if len(filtered_families) < n_families_oversample:
        n_families_oversample = len(filtered_families)
        print(f"Reduced number of families to sample to {n_families_oversample} due to filtering.")

    selected_families = random.sample(filtered_families, n_families_oversample)

    # Split selected families into validation and test sets
    val_families = selected_families[:n_families_oversample // 2]
    test_families = selected_families[n_families_oversample // 2:]

    with open(pfam_select_fam_path, 'w') as f:
        f.write('family_accession,split\n')
        for fam in val_families:
            f.write(f'{fam},val\n')
        for fam in test_families:
            f.write(f'{fam},test\n')


def make_val_test_parquets(selected_families, parquet_save_dir, pfam_dir, flat_file_path):
    """
    Create parquet files for both split types in {random, clustered}
    Each split_type has val and test splits.
    For each family, create one row in the parquet file with the following structure:
    {
        "fam_id": PF00001,
        "accessions": np.array(["P12345", "P23456", ...]),  # UniProt accessions
        "sequences": np.array(["--MAG-..", "--MAG-...", ...]),  # Aligned sequences from 'train' split
        "completion_accessions": np.array(["P34567", "P45678", ...]),  # UniProt accessions for "test" split
        "completion_sequences": np.array(["--MAG-...", "--MAG-...", ...]),  # Aligned sequences from 'test' split
    }
    Additionally, create a single flat file (CSV) that has one row per sequence with the following columns:
    {
        "fam_id": PF00001,
        "accession": P12345,  # UniProt accession
        "sequence_name": Q5KPZ5_CRYNJ/238-462,  # From 'sequence_name' column
        "split": "val",
        "is_completion": False,
        "sequence": "MAG",  # From 'sequence' column
    }
    """
    flat_file_rows = []

    for split_type in ['clustered_split', 'random_split']:
        for eval_split in ['val', 'test']:
            print(f"Processing split_type: {split_type}, eval_split: {eval_split}")
            eval_families = selected_families[selected_families['split'] == eval_split]['family_accession'].unique()
            eval_families = set(fam.split('.')[0] for fam in eval_families)  # Remove version numbers

            fam_id_to_data = defaultdict(lambda: {
                'fam_id': '',
                'accessions': [],
                'sequences': [],
                'sequence_names': [],
                'sequence_choppings': [],
                'completion_accessions': [],
                'completion_sequences': [],
                'completion_sequence_names': [],
                'completion_sequence_choppings': [],
            })

            for within_family_split in ['train', 'test']:
                save_path = os.path.join(
                    parquet_save_dir,
                    f"selected_{split_type}_{within_family_split}_{eval_split}_uniprot_mapped.csv"
                )
                if not os.path.exists(save_path):
                    raise FileNotFoundError(f"File {save_path} does not exist.")

                df_w_accs = pd.read_csv(save_path)
                df_w_accs['family_accession'] = df_w_accs['family_accession'].apply(lambda x: x.split(".")[0])
                df_w_accs = df_w_accs[df_w_accs['family_accession'].isin(eval_families)]
                if df_w_accs.empty:
                    continue

                # Remove rows where 'Entry' is null
                df_w_accs = df_w_accs[~df_w_accs['Entry'].isnull()]
                if df_w_accs.empty:
                    continue

                # Extract necessary fields
                df_w_accs['accession'] = df_w_accs['Entry']
                df_w_accs['fam_id'] = df_w_accs['family_accession']
                df_w_accs['split'] = eval_split
                df_w_accs['is_completion'] = (within_family_split == 'test')
                df_w_accs['sequence_name'] = df_w_accs['sequence_name']
                df_w_accs['sequence'] = df_w_accs['sequence']
                df_w_accs['aligned_sequence'] = df_w_accs['aligned_sequence']
                df_w_accs['split_type'] = split_type
                # Add to flat file rows
                flat_file_rows.extend(
                    df_w_accs[[
                        'fam_id',
                        'accession',
                        'sequence_name',
                        'split',
                        'split_type',
                        'is_completion',
                        'sequence'
                    ]].to_dict('records')
                )

                # Organize data for parquet files
                for fam_id, group in df_w_accs.groupby('fam_id'):
                    sequence_names = group['sequence_name'].tolist()
                    sequence_choppings = [name.split('/')[1] for name in sequence_names]
                    if fam_id_to_data[fam_id]['fam_id'] == '':
                        fam_id_to_data[fam_id]['fam_id'] = fam_id
                    if within_family_split == 'train':
                        fam_id_to_data[fam_id]['accessions'].extend(group['accession'].tolist())
                        fam_id_to_data[fam_id]['sequences'].extend(group['aligned_sequence'].tolist())
                        fam_id_to_data[fam_id]['sequence_names'].extend(sequence_names)
                        fam_id_to_data[fam_id]['sequence_choppings'].extend(sequence_choppings)
                    else:  # Test split within family
                        fam_id_to_data[fam_id]['completion_sequences'].extend(group['aligned_sequence'].tolist())
                        fam_id_to_data[fam_id]['completion_accessions'].extend(group['accession'].tolist())
                        fam_id_to_data[fam_id]['completion_sequence_names'].extend(sequence_names)
                        fam_id_to_data[fam_id]['completion_sequence_choppings'].extend(sequence_choppings)

            # Convert fam_id_to_data to DataFrame and save as parquet
            if fam_id_to_data:
                parquet_df = pd.DataFrame.from_dict(fam_id_to_data, orient='index')
                parquet_df.reset_index(drop=True, inplace=True)
                parquet_filename = f"{eval_split}_{split_type}.parquet"
                parquet_path = os.path.join(parquet_save_dir, parquet_filename)
                os.makedirs(parquet_save_dir, exist_ok=True)  # Ensure the directory exists
                parquet_df.to_parquet(parquet_path, index=False)
                print(f"Saved parquet file: {parquet_path}")
            else:
                print(f"No data for {split_type} {eval_split}, skipping parquet saving.")

    # Save the flat file
    if flat_file_rows:
        flat_file_df = pd.DataFrame(flat_file_rows)
        flat_file_df.to_csv(flat_file_path, index=False)
        print(f"Saved flat file: {flat_file_path}")
    else:
        raise ValueError("Processed 0 rows")



def filter_fams_without_up_accs(
    selected_families: pd.DataFrame,
    n_families: int,
    pfam_save_dir: str,
    external_pfam_dir: str,
):
    """
    Looks up uniprot accessions codes based on the protein name
    families which have entries which cannot be mapped to a uniprot
    accession are dropped.
    :param selected_families:
    :return:
    """
    for split_type in ['clustered_split', 'random_split']:
        for eval_split in ['val', 'test']:
            eval_families = selected_families[selected_families['split'] == eval_split]['family_accession'].values
            families_with_null_accessions = []
            for within_family_split in ['train', 'test']:
                dfs = []
                save_path = os.path.join(pfam_save_dir, f"selected_{split_type}_{within_family_split}_{eval_split}_uniprot_mapped.csv")
                if not os.path.exists(save_path):
                    split_dir = os.path.join(external_pfam_dir, split_type, within_family_split)
                    for fname in sorted(os.listdir(split_dir)):
                        if fname.startswith('data'):
                            split_fam = pd.read_csv(os.path.join(split_dir, fname))
                            split_fam['family_accession'] = split_fam['family_accession'].apply(lambda x: x.split(".")[0])
                            split_fam = split_fam[split_fam['family_accession'].isin(eval_families)]
                            dfs.append(split_fam)
                    if dfs:
                        combined = pd.concat(dfs)

                        combined = get_uniprot_accessions_from_names(
                            combined,
                            save_path=save_path
                        )
                    else:
                        continue
                else:
                    combined = pd.read_csv(save_path)
                families_with_null_accessions.extend(combined[combined['Entry'].isnull()]['family_accession'].unique())
    selected_families = selected_families[~selected_families['family_accession'].isin(families_with_null_accessions)]

    val_families = selected_families[selected_families['split'] == 'val']['family_accession'].values
    test_families = selected_families[selected_families['split'] == 'test']['family_accession'].values
    assert not set(val_families).intersection(test_families)
    selected_families = selected_families.sort_values(by="split")
    selected_families.reset_index(drop=True, inplace=True)
    return selected_families


def remove_val_test_rows(val_test_df, old_parquet_dir, new_parquet_dir, limit_mb=125):
    val_test_fam_ids = set(val_test_df.family_accession.apply(lambda x: x.split(".")[0]))
    val_fam_ids = set(val_test_df[val_test_df['split'] == "val"].family_accession.apply(lambda x: x.split(".")[0]))
    test_fam_ids = set(val_test_df[val_test_df['split'] == "test"].family_accession.apply(lambda x: x.split(".")[0]))

    # Create output directories
    train_dir = os.path.join(new_parquet_dir, 'train')
    val_dir = os.path.join(new_parquet_dir, 'val')
    test_dir = os.path.join(new_parquet_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    val_buffer = ParquetBufferWriter(val_dir, name="val", mem_limit=limit_mb)
    test_buffer = ParquetBufferWriter(test_dir, name="test", mem_limit=limit_mb)
    train_accs = set()
    global new_index
    new_index = []  # needs to be accessed by the ParquetBufferWriter objects

    # Process each parquet file
    for parquet_file in glob.glob(os.path.join(old_parquet_dir, '*.parquet')):
        t_path = os.path.join(train_dir, f"train_{os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        df["pfam_version"] = df.fam_id.apply(lambda x: x.split(".")[1] if '.' in x else None)
        df["fam_id"] = df.fam_id.apply(lambda x: x.split(".")[0])
        train_df = df[~df['fam_id'].isin(val_test_fam_ids)]
        val_df = df[df.fam_id.isin(val_fam_ids)]
        test_df = df[df.fam_id.isin(test_fam_ids)]
        if not val_df.empty:
            val_buffer.update_buffer(val_df)
        if not test_df.empty:
            test_buffer.update_buffer(test_df)

        # Save train data
        if not train_df.empty:
            train_accs.update(set(train_df.fam_id))
            train_df.to_parquet(t_path)
            new_index.extend([(row['fam_id'], os.path.basename(t_path)) for _, row in train_df.iterrows()])
    val_buffer.write_dfs()
    test_buffer.write_dfs()
    return new_index

class ParquetBufferWriter:
    def __init__(self, outdir, name, mem_limit=250):
        self.outdir = outdir
        self.name = name
        self.mem_limit = mem_limit
        self.index = 0
        self.dfs = []
        self.mem_use = 0

    def update_buffer(self, df):
        """
        Increment memory usage; if over mem_limit, write dfs to parquet and reset buffer.
        """
        seqs_mb = sum([sys.getsizeof(seq) for family in df['sequences'] for seq in family]) / (1024 * 1024)
        print(f"Memory usage {self.name}: {round(self.mem_use, 6)} MB")
        if self.mem_use + seqs_mb < self.mem_limit:
            self.dfs.append(df)
            self.mem_use += seqs_mb
        else:
            self.write_dfs()
            self.dfs = [df]
            self.mem_use = seqs_mb

    def write_dfs(self):
        global new_index
        if len(self.dfs):
            combi_df = pd.concat(self.dfs)
            filepath = os.path.join(self.outdir, f"{self.name}_{str(self.index).zfill(3)}.parquet")
            print(f"Writing to {filepath}")
            combi_df.to_parquet(filepath)
            new_index.extend([(row['fam_id'], os.path.basename(filepath)) for _, row in combi_df.iterrows()])
            self.dfs = []
            self.index += 1
            self.mem_use = 0


def create_pfam_val_test_all_up_ids_json(
    selected_families,
    pfam_uniprot_json_path,
    pfam_save_dir,
    min_overlap_threshold=0.5,
):
    """
    Creates a JSON file mapping Pfam families to a list of UniProt IDs.
    Discards families where less than a specified proportion of the UniProt IDs from the CSV
    can be matched in the JSON mapping.

    :param selected_families: DataFrame of selected families
    :param pfam_uniprot_json_path: Path to the Pfam-UniProt JSON mapping
    :param output_json_path: Path to save the combined Pfam-UniProt mapping for selected families
    :param pfam_save_dir: Directory containing Pfam data
    :param min_overlap_threshold: Minimum required proportion of overlapping UniProt IDs
    :return: Filtered DataFrame of selected families
    """
    print("Loading Pfam to UniProt mapping...")
    pfam_to_uniprot = json.load(open(pfam_uniprot_json_path))

    fam_to_up = defaultdict(set)
    families_to_remove = set()
    proportions_overlap = []

    for split_type in ['clustered_split', 'random_split']:
        for eval_split in ['val', 'test']:
            eval_families = selected_families[selected_families['split'] == eval_split]['family_accession'].unique()
            eval_families = set(fam.split('.')[0] for fam in eval_families)  # Remove version numbers

            for within_family_split in ['train', 'test']:
                save_path = os.path.join(
                    pfam_save_dir,
                    f"selected_{split_type}_{within_family_split}_{eval_split}_uniprot_mapped.csv"
                )
                if not os.path.exists(save_path):
                    raise FileNotFoundError(
                        f"File {save_path} does not exist. Should have been created by get_uniprot_accessions_from_names()"
                    )
                df_w_accs = pd.read_csv(save_path)
                df_w_accs['family_accession'] = df_w_accs['family_accession'].apply(lambda x: x.split(".")[0])
                df_w_accs = df_w_accs[df_w_accs['family_accession'].isin(eval_families)]
                if df_w_accs.empty:
                    continue

                # Remove rows where 'Entry' is null
                df_w_accs = df_w_accs[~df_w_accs['Entry'].isnull()]
                if df_w_accs.empty:
                    continue

                # Process each family
                for fam_id in df_w_accs['family_accession'].unique():
                    fam_df = df_w_accs[df_w_accs['family_accession'] == fam_id]
                    csv_uniprot = set(fam_df['Entry'].dropna().astype(str))
                    json_uniprot = set(pfam_to_uniprot.get(fam_id, []))

                    if not csv_uniprot:
                        overlap_ratio = 0.0
                    else:
                        overlap_ratio = len(csv_uniprot.intersection(json_uniprot)) / len(csv_uniprot)

                    if overlap_ratio < min_overlap_threshold:
                        families_to_remove.add(fam_id)
                    else:
                        combined_uniprot = csv_uniprot.union(json_uniprot)
                        fam_to_up[fam_id].update(combined_uniprot)

                    proportions_overlap.append(overlap_ratio)

    # Remove families with low overlap
    print(f"Removing {len(families_to_remove)} families due to low UniProt ID overlap.")
    selected_families = selected_families[~selected_families['family_accession'].isin(families_to_remove)]
    selected_families.reset_index(drop=True, inplace=True)

    # Convert sets to lists for JSON serialization
    fam_to_up = {k: list(v) for k, v in fam_to_up.items() if k not in families_to_remove}

    average_overlap = sum(proportions_overlap) / len(proportions_overlap) if proportions_overlap else 0
    print(f"Average proportion of UniProt IDs overlapping between CSV and JSON: {average_overlap}")

    return selected_families, fam_to_up
    
def select_families(
    external_pfam_dir,
    pfam_save_dir,
    pfam_uniprot_json_path,
    output_json_path,
    n_families_total,
):
    """
    Filter the Pfam families so that they meet the following criteria:
    - Families that occur in both train and test for both clustered and random splits
    - Exclude families with more than 10,000 members or fewer than 10 members
    - Discard families where less than 50% of the UniProt IDs from the CSV can be matched in the JSON mapping

    After filtering, downsample to get the final number of families.

    :param external_pfam_dir: Directory containing Pfam data
    :param pfam_uniprot_json_path: Path to the Pfam-UniProt JSON mapping
    :param output_json_path: Path to save the combined Pfam-UniProt mapping for selected families
    :param n_families_total: Total number of families to select
    :return: DataFrame of selected families
    """
    # Paths for intermediate files
    pfam_select_fam_path = os.path.join(pfam_save_dir, 'eval_families_filtered.csv')
    pfam_select_fam_w_up_accs_path = pfam_select_fam_path.replace(".csv", "_w_unip_accs.csv")

    if not os.path.exists(pfam_select_fam_w_up_accs_path):
        if not os.path.exists(pfam_select_fam_path):
            sample_fams_by_size(
                pfam_select_fam_path,
                pfam_uniprot_json_path,
                n_families=n_families_total,
            )
        selected_families = pd.read_csv(pfam_select_fam_path)
        selected_families = filter_fams_without_up_accs(
            selected_families,
            n_families=n_families_total,
            pfam_save_dir=pfam_save_dir,
            external_pfam_dir=external_pfam_dir,
        )
        
        # Call the function to create the mapping and filter families based on overlap
        selected_families, fam_to_up_id_dict = create_pfam_val_test_all_up_ids_json(
            selected_families=selected_families,
            pfam_uniprot_json_path=pfam_uniprot_json_path,
            pfam_save_dir=pfam_save_dir,
            min_overlap_threshold=0.5,
        )
        
        # Downsample to get the final number of families
        val_families = selected_families[selected_families['split'] == 'val']
        test_families = selected_families[selected_families['split'] == 'test']
        
        if len(val_families) >= n_families_total // 2 and len(test_families) >= n_families_total // 2:
            val_families = val_families.sample(n=n_families_total // 2, random_state=42)
            test_families = test_families.sample(n=n_families_total // 2, random_state=42)
            selected_families = pd.concat([val_families, test_families], ignore_index=True)
        else:
            raise ValueError("Not enough families after filtering to downsample to the desired number.")
        fam_to_up_id_dict = {
            k: v for k, v in fam_to_up_id_dict.items() if k in selected_families['family_accession'].values
        }
        assert set(fam_to_up_id_dict.keys()) == set(selected_families['family_accession'].values)
        # Write the mapping to the output JSON file
        with open(output_json_path, "w") as f:
            json.dump(fam_to_up_id_dict, f, indent=2)
        selected_families.to_csv(pfam_select_fam_w_up_accs_path, index=False)
    else:
        selected_families = pd.read_csv(pfam_select_fam_w_up_accs_path)

    return selected_families


if __name__ == "__main__":
    external_pfam_dir = '../pfam_eval_splits'
    split_parquet_save_dir = "../data/pfam/train_test_split_parquets"
    index_csv_filename = "pfam_val_test_w_accessions.csv"
    index_csv_path = os.path.join(split_parquet_save_dir, index_csv_filename)
    pfam_uniprot_json_path = "../data/pfam/pfam_uniprot_mappings.json"
    output_json_path = os.path.join(
        split_parquet_save_dir,
        "pfam_val_test_all_up_ids.json"
    )
    pre_shuffled_parquet_dir = "../data/pfam/combined_parquets"
    shuffled_parquet_dir = "../data/pfam/shuffled_parquets"

    flat_file_path = os.path.join(split_parquet_save_dir, 'pfam_val_test_flat_file.csv')
    os.makedirs(split_parquet_save_dir, exist_ok=True)
    os.makedirs(shuffled_parquet_dir, exist_ok=True)
    n_families = 500
    limit_mb_per_parquet = 125

    print("Selecting pfam family IDs for val / test...")
    selected_families = select_families(
        external_pfam_dir=external_pfam_dir,
        pfam_save_dir=split_parquet_save_dir,
        pfam_uniprot_json_path=pfam_uniprot_json_path,
        n_families_total=n_families,
        output_json_path=output_json_path,
    )

    print("Creating parquet files for val / test...")
    if not os.path.exists(flat_file_path):
        make_val_test_parquets(
            selected_families=selected_families,
            parquet_save_dir=split_parquet_save_dir,
            pfam_dir=external_pfam_dir,
            flat_file_path=flat_file_path
        )


    if len(glob.glob(os.path.join(shuffled_parquet_dir, '*.parquet'))) == 0:
        print("Shuffling parquet files...")
        shuffle_pfam_parquets(
            indir=pre_shuffled_parquet_dir,
            outdir=shuffled_parquet_dir,
            limit_mb=limit_mb_per_parquet,
        )


    # Remove validation and test families from the Pfam training data
    new_index = remove_val_test_rows(
        val_test_df=selected_families,
        old_parquet_dir=shuffled_parquet_dir,
        new_parquet_dir=split_parquet_save_dir,
        limit_mb=125,
    )

    # Write new index.csv
    index_csv_output = os.path.join(split_parquet_save_dir, 'pfam_post_split_index.csv')
    with open(index_csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fam_id', 'parquet_file'])
        writer.writerows(new_index)
    print(f"New index CSV saved to {index_csv_output}")