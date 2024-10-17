import csv
import glob
import json
import logging
import os
import random
import sys
import shutil
from collections import defaultdict

import pandas as pd

from data_creation_scripts.pfam.get_up_accs_for_all_of_pfam import (
    setup_logging,
    get_name_to_accession_mapping,
    process_parquet_files,
    offline_get_name_to_accession_mapping,
    map_val_test_names_to_accessions
)
from data_creation_scripts.pfam.shuffle_pfam_parquets import shuffle_pfam_parquets
from data_creation_scripts.pfam.deduplicate_pfam import deduplicate_families

"""
Consolidated script for Pfam data processing.

Priot to running this script follow the instructions in:
data_creation_scripts/pfam/README_pfam.md

This script performs the following:

1. Selects Pfam families that occur in both train and test splits for both clustered and random splits, 
excluding families with more than 10,000 members or fewer than 10 members, 
and families where the UniProt IDs are not present in the provided JSON mapping.

2. Creates parquet files for the validation and test datasets

3. shuffles the pre-split pfam parquets

4. removes duplicated families from the pre-split pfam parquets

5. Removes validation and test families from the Pfam training data 
parquet files and splits the data into train, validation, and test sets.

6. Generates the 'pfam_val_test_all_up_ids.json' file mapping Pfam families to UniProt IDs.

7. Maps sequence names to UniProt accessions and adds a 'matched_accessions' column to the parquet files
(too many sequences to do this entirely via API calls, so first pass uses a downloaded version of 
UniProt ID mapping, which gets 98% of sequences, second pass uses API calls to get the remaining 2%).

This creates the following files: 

train_test_split_parquets_v3
├── eval_families_filtered_w_unip_accs.csv # 500 pfam families and whether in val or test
├── pfam_all_sequence_names.txt  # ~59M sequence names in pfam (used to look up uniprot accessions)
├── pfam_post_split_index.csv   # fam_id -> parquet file mapping
├── pfam_val_test_all_up_ids.json # pfam family -> uniprot accessions mapping (ALL accs associated with the pfam not just those actually used in the val/test set)
├── pfam_val_test_flat_file.csv  # one row per sequence with columns: fam_id,accession,matched_accession,split,split_type,is_completion
├── selected_clustered_split_test_test_uniprot_mapped.csv  │
├── selected_clustered_split_test_val_uniprot_mapped.csv   │
├── selected_clustered_split_train_test_uniprot_mapped.csv │
├── selected_clustered_split_train_val_uniprot_mapped.csv  ├── # contains aligned sequences for all val/ test sequences (redundant info with parquets)
├── selected_random_split_test_test_uniprot_mapped.csv     │
├── selected_random_split_test_val_uniprot_mapped.csv      │
├── selected_random_split_train_test_uniprot_mapped.csv    │
├── selected_random_split_train_val_uniprot_mapped.csv     │
├── test                                    │
│   ├── test_000.parquet                    │
│   ├── test_001.parquet                    │
├── train                                   │
│   ├── train_Domain_006.parquet            │
│   ├── train_Domain_007.parquet            ├── original training data now split into train, val, test
│   ├── train_Family_000.parquet            │
│   └── train_X0BZJ7_PF00067.27_0.parquet   │
├── val                                     │
│   ├── val_000.parquet                     │
│   ├── val_001.parquet                     │
│   └── val_002.parquet                     │
├── test_clustered_split.parquet │
├── test_random_split.parquet    │
├── val_clustered_split.parquet  ├── special parquet files that include completion sequences for val/test
└── val_random_split.parquet     │


Usage:
    python consolidated_pfam_processing.py
"""
API_URL = "https://rest.uniprot.org"
setup_logging()

def initial_filter_fams(
    pfam_select_fam_path,
    pfam_uniprot_json_path,
    fams_in_pfamA_full,
    n_families=1000,
    min_seq_len=50,
    min_famsize=10,
    max_famsize=10000,
):
    """
    Select families that occur in train AND test for BOTH clustered AND random splits,
    and 
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
            pre_filter = len(combined)
            combined = combined[combined['family_accession'].isin(fams_in_pfamA_full)]
            post_filter = len(combined)
            print(f"Filtered {pre_filter - post_filter} rows not present in pfamA_full")
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


def make_val_test_parquets(selected_families, parquet_save_dir, flat_file_path):
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
        "mapped_accession": P12345,  # UniProt accession
        "accession": Q5KPZ5_CRYNJ/238-462,  # From original pfam file
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
                'matched_accessions': [],
                'sequence_choppings': [],
                'completion_accessions': [],
                'completion_sequences': [],
                'completion_matched_accessions': [],
                'completion_sequence_choppings': [],
            })

            for within_family_split in ['train', 'test']:
                save_path = os.path.join(
                    parquet_save_dir,
                    f"selected_{split_type}_{within_family_split}_{eval_split}_uniprot_mapped.csv"
                )
                if not os.path.exists(save_path):
                    raise FileNotFoundError(f"File {save_path} does not exist.")
                print(f"Reading file: {save_path}")
                df_w_accs = pd.read_csv(save_path)
                print(df_w_accs.head())
                df_w_accs['family_accession'] = df_w_accs['family_accession'].apply(lambda x: x.split(".")[0])
                df_w_accs = df_w_accs[df_w_accs['family_accession'].isin(eval_families)]
                if df_w_accs.empty:
                    continue

                # Remove rows where 'matched_accession' is null
                df_w_accs = df_w_accs[~df_w_accs['matched_accession'].isnull()]
                if df_w_accs.empty:
                    continue

                # Extract necessary fields
                df_w_accs['accession'] = df_w_accs['sequence_name']
                df_w_accs['fam_id'] = df_w_accs['family_accession']
                df_w_accs['split'] = eval_split
                df_w_accs['is_completion'] = (within_family_split == 'test')
                df_w_accs['split_type'] = split_type
                # Add to flat file rows
                flat_file_rows.extend(
                    df_w_accs[[
                        'fam_id',
                        'accession',
                        'matched_accession',
                        'split',
                        'split_type',
                        'is_completion',
                    ]].to_dict('records')
                )

                # Organize data for parquet files
                for fam_id, group in df_w_accs.groupby('fam_id'):
                    accessions = group['accession'].tolist()
                    matched_accessions = group['matched_accession'].tolist()
                    sequence_choppings = [name.split('/')[1] for name in accessions]
                    if fam_id_to_data[fam_id]['fam_id'] == '':
                        fam_id_to_data[fam_id]['fam_id'] = fam_id
                    if within_family_split == 'train':
                        fam_id_to_data[fam_id]['accessions'].extend(accessions)
                        fam_id_to_data[fam_id]['sequences'].extend(group['aligned_sequence'].tolist())
                        fam_id_to_data[fam_id]['matched_accessions'].extend(matched_accessions)
                        fam_id_to_data[fam_id]['sequence_choppings'].extend(sequence_choppings)
                    else:  # Test split within family
                        fam_id_to_data[fam_id]['completion_sequences'].extend(group['aligned_sequence'].tolist())
                        fam_id_to_data[fam_id]['completion_accessions'].extend(accessions)
                        fam_id_to_data[fam_id]['completion_matched_accessions'].extend(matched_accessions)
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

                        combined = map_val_test_names_to_accessions(
                            combined,
                            save_path=save_path
                        )
                    else:
                        continue
                else:
                    combined = pd.read_csv(save_path)
                families_with_null_accessions.extend(combined[combined['matched_accession'].isnull()]['family_accession'].unique())
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

                # Remove rows where 'matched_accession' is null
                df_w_accs = df_w_accs[~df_w_accs['matched_accession'].isnull()]
                if df_w_accs.empty:
                    continue

                # Process each family
                for fam_id in df_w_accs['family_accession'].unique():
                    fam_df = df_w_accs[df_w_accs['family_accession'] == fam_id]
                    csv_uniprot = set(fam_df['matched_accession'].dropna().astype(str))
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
    pre_split_pfam_pfamily_to_file_index,
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
    assert os.path.exists(pre_split_pfam_pfamily_to_file_index)

    fams_present_in_pfamA_full = list(pd.read_csv(
        pre_split_pfam_pfamily_to_file_index
        ).fam_id.apply(lambda x: x.split(".")[0]).unique())
    logging.info(f"Number of families in pfamA_full: {len(fams_present_in_pfamA_full)}")
    print(fams_present_in_pfamA_full[:5])
    if not os.path.exists(pfam_select_fam_w_up_accs_path):
        if not os.path.exists(pfam_select_fam_path):
            print("Selecting families for val and test splits...")
            initial_filter_fams(
                pfam_select_fam_path=pfam_select_fam_path,
                pfam_uniprot_json_path=pfam_uniprot_json_path,
                fams_in_pfamA_full=fams_present_in_pfamA_full,
                n_families=n_families_total,
            )
        selected_families = pd.read_csv(pfam_select_fam_path)
        print("Filtering families without UniProt accessions...")
        selected_families = filter_fams_without_up_accs(
            selected_families,
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
        # Remove the intermediate file
        os.remove(pfam_select_fam_path)
    else:
        print(f"Reading pre-selected families from {pfam_select_fam_w_up_accs_path}")
        selected_families = pd.read_csv(pfam_select_fam_w_up_accs_path)

    return selected_families

def add_accessions_to_parquets(split_parquet_save_dir, map_save_dir, use_id_mapping_api=False):
    assert os.path.exists(split_parquet_save_dir)
    assert os.path.exists(map_save_dir)
    unmatched_names_path = os.path.join(map_save_dir, "unmatched_sequence_names.txt")
    if os.path.exists(unmatched_names_path):
        logging.info(f"Unmatched names file already exists: skipping add accessions step")
        return []
    all_sequence_names_path = os.path.join(split_parquet_save_dir, 'pfam_all_sequence_names.txt')
    parq_paths = glob.glob(f"{split_parquet_save_dir}/*/*.parquet")
    # Collect all unique sequence names from the parquet files
    if not os.path.exists(all_sequence_names_path):
        all_sequence_names = set()
        for parq_path in parq_paths:
            logging.info(f"Reading parquet file: {parq_path}")
            df = pd.read_parquet(parq_path)

            sequence_names = df['accessions'].explode().apply(lambda x: x.split("/")[0]).to_list()
            all_sequence_names.update(sequence_names)

        with open(all_sequence_names_path, 'w') as f:
            f.write("\n".join(all_sequence_names))
    else:
        with open(all_sequence_names_path, 'r') as f:
            all_sequence_names = set(f.read().splitlines())
    logging.info(f"Total unique sequence names collected: {len(all_sequence_names)}")

    # Get the mapping from sequence names to UniProt accessions
    logging.info("Mapping sequence names to UniProt accessions...")

    if use_id_mapping_api:
        print("Using ID mapping API")
        name_to_accession_mapping = get_name_to_accession_mapping(
            list(all_sequence_names),
            map_save_dir=map_save_dir,
        )
    else:
        print("Using downloaded flat file to map IDs")
        name_to_accession_mapping = offline_get_name_to_accession_mapping(
            all_sequence_names,
            map_save_dir=map_save_dir,
        )

    # Process parquet files to add 'matched_accessions' column and overwrite them
    logging.info("Adding 'matched_accessions' column to parquet files...")
    unmatched_names = process_parquet_files(parq_paths, name_to_accession_mapping)
    # Save unmatched names to a file
    with open(unmatched_names_path, 'w') as f:
        for name in unmatched_names:
            f.write(f"{name}\n")
    logging.info(f"Saved {len(unmatched_names)} unmatched sequence names to {unmatched_names_path}")

    if not use_id_mapping_api and len(unmatched_names) > 0:
        logging.info("Retrying unmatched names with ID mapping API...")
        retry_save_dir = f"{map_save_dir}/retry_api_mapping"
        os.makedirs(retry_save_dir, exist_ok=True)
        retried_name_to_accession_mapping = get_name_to_accession_mapping(
            list(unmatched_names),
            map_save_dir=retry_save_dir,
        )
        mapping_path = f"{map_save_dir}/filtered_id_mapping.tsv"
        logging.info(f"Appending new mappings to {mapping_path}")
        with open(mapping_path, 'a') as f:
            for name, acc in retried_name_to_accession_mapping.items():
                f.write(f"{name}\t{acc}\n")
        name_to_accession_mapping = offline_get_name_to_accession_mapping(
            all_sequence_names,
            map_save_dir=map_save_dir,
        )
        unmatched_names = process_parquet_files(parq_paths, name_to_accession_mapping)
        with open(unmatched_names_path, 'w') as f:
            for name in unmatched_names:
                f.write(f"{name}\n")
        logging.info(f"Saved {len(unmatched_names)} unmatched sequence names to {unmatched_names_path}")

    return unmatched_names

def concat_parquets_rowwise(parquet_paths):
    dfs = []
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def combine_val_test_parquets(split_parquet_save_dir):
    """
    Combine the val and test parquet files which are
    generated from the Pfam-A.full file and those
    which are downloaded in the csv with clustered splits

    csv sequences and accessions will be renamed to:
    input_accessions and input_sequences.
    """
    csv_full_missing = {}
    for split in ['val', 'test']:
        csv_full_missing[split] = {}
        pfam_full_parquet_paths = glob.glob(f"{split_parquet_save_dir}/{split}/*.parquet")
        pfam_full_df = concat_parquets_rowwise(pfam_full_parquet_paths)
        assert len(pfam_full_parquet_paths) > 0
        logging.info(f"Combined {len(pfam_full_parquet_paths)} parquet files for {split}")
        logging.info(f"Found {pfam_full_df.fam_id.nunique()} unique families in Pfam-full {split}")
        for split_type in ['clustered_split', 'random_split']:
            csv_full_missing[split][split_type] = {}
            csv_parquet_path = f"{split_parquet_save_dir}/{split}_{split_type}.parquet"
            assert os.path.exists(csv_parquet_path), f"File {csv_parquet_path} does not exist."
            csv_df = pd.read_parquet(csv_parquet_path)
            logging.info(f"Read {len(csv_df)} rows from {csv_parquet_path}")
            logging.info(f"Found {csv_df.fam_id.nunique()} unique families in {split_type} {split}")
            rename_map = {
                'fam_id': 'fam_id',
                'accessions': 'input_accessions',
                'sequences': 'input_sequences',
                'matched_accessions': 'input_matched_accessions',
                'sequence_choppings': 'input_sequence_choppings',
                'completion_accessions': 'completion_accessions',
                'completion_sequences': 'completion_sequences',
                'completion_matched_accessions': 'completion_matched_accessions',
                'completion_sequence_choppings': 'completion_sequence_choppings'
            }
            csv_df.rename(columns=rename_map, inplace=True)
            # join on fam_id
            combined_df = pd.merge(pfam_full_df, csv_df, on='fam_id', how='inner')
            fams_missing_from_csv = set(pfam_full_df.fam_id) - set(csv_df.fam_id)
            fams_missing_from_pfam_full = set(csv_df.fam_id) - set(pfam_full_df.fam_id)
            print(f"Families missing from CSV: {len(fams_missing_from_csv)}")
            print(f"Families missing from Pfam full: {len(fams_missing_from_pfam_full)}")
            csv_full_missing[split][split_type] = {
                'families_missing_from_csv': list(fams_missing_from_csv),
                'families_missing_from_pfam_full': list(fams_missing_from_pfam_full),
            }
            combined_df.to_parquet(csv_parquet_path.replace(".parquet", "_combined.parquet"), index=False)
    with open(f"{split_parquet_save_dir}/csv_full_missing.json", 'w') as f:
        json.dump(csv_full_missing, f, indent=2)


def remove_intermediate_files(split_parquet_save_dir):
    files_to_remove = [
        "selected_clustered_split_test_test_uniprot_mapped.csv",
        "selected_clustered_split_test_val_uniprot_mapped.csv",
        "selected_clustered_split_train_test_uniprot_mapped.csv",
        "selected_clustered_split_train_val_uniprot_mapped.csv",
        "selected_random_split_test_test_uniprot_mapped.csv",
        "selected_random_split_test_val_uniprot_mapped.csv",
        "selected_random_split_train_test_uniprot_mapped.csv",
        "selected_random_split_train_val_uniprot_mapped.csv",
        "test_random_split.parquet",
        "test_clustered_split.parquet",
        "val_random_split.parquet",
        "val_clustered_split.parquet",
    ]
    dirs_to_remove = [
        os.path.join(split_parquet_save_dir, 'val'),
        os.path.join(split_parquet_save_dir, 'test'),
    ]
    for f in files_to_remove:
        logging.info(f"Removing file: {f}")
        fpath = os.path.join(split_parquet_save_dir, f)
        if os.path.exists(fpath):
            os.remove(fpath)

    for d in dirs_to_remove:
        logging.info(f"Removing dir: {d}")
        if os.path.exists(d):
            shutil.rmtree(d)

if __name__ == "__main__":
    use_id_mapping_api = False # if false download ID mapping flat file from uniprot
    external_pfam_dir = '../data/pfam/pfam_eval_splits'  # data splits from other authors
    split_parquet_save_dir = "../data/pfam/train_test_split_parquets_v3"
    index_csv_filename = "pfam_val_test_w_accessions.csv"
    pfam_uniprot_json_path = "../data/pfam/pfam_uniprot_mappings.json"
    output_json_path = os.path.join(
        split_parquet_save_dir,
        "pfam_val_test_all_up_ids.json"
    )
    pre_shuffled_parquet_dir = "../data/pfam/combined_parquets"  # created by array_job_split_pfam.py
    shuffled_parquet_dir = "../data/pfam/shuffled_parquets"
    pre_split_pfam_pfamily_to_file_index = f"{shuffled_parquet_dir}/new_index.csv"
    map_save_dir = "../data/pfam/sequence_name_to_uniprot_mapping"
    flat_file_save_dir = "data/val_test/pfam"

    if not os.path.exists(pfam_uniprot_json_path):
        raise FileNotFoundError(f"File {pfam_uniprot_json_path} does not exist. Run get_up_accs_for_all_of_pfam.py first.")

    index_csv_path = os.path.join(split_parquet_save_dir, index_csv_filename)
    flat_file_path = os.path.join(flat_file_save_dir, 'pfam_val_test_flat_file.csv')
    os.makedirs(flat_file_save_dir, exist_ok=True)
    os.makedirs(split_parquet_save_dir, exist_ok=True)
    os.makedirs(shuffled_parquet_dir, exist_ok=True)
    os.makedirs(map_save_dir, exist_ok=True)
    n_families = 500  # Number of families to select for val + test
    limit_mb_per_parquet = 125

    if len(glob.glob(f"{shuffled_parquet_dir}/*.parquet")) < 50:
        print("Shuffling Pfam parquets")
        shuffle_pfam_parquets(
            indir=pre_shuffled_parquet_dir,
            outdir=shuffled_parquet_dir,
            limit_mb=limit_mb_per_parquet,
        )
    else:
        print("Shuffled parquets already exist. Skipping...")
    dropped_rows_json_path = os.path.join(shuffled_parquet_dir, 'duplicated_dropped_rows.json')
    
    if not os.path.exists(dropped_rows_json_path):
        print("removing duplicated entries from shuffled parquets")
        dropped_rows = deduplicate_families(shuffled_parquet_dir)
        try:
            with open(dropped_rows_json_path, 'w') as f:
                json.dump(dropped_rows, f, indent=2)
        except Exception as e:
            print(f"Failed to write dropped rows to JSON: {e}")
    else:
        print("Dropped rows JSON already exists so skipping deduplication...")

    selected_families = select_families(
        external_pfam_dir=external_pfam_dir,
        pfam_save_dir=split_parquet_save_dir,
        pfam_uniprot_json_path=pfam_uniprot_json_path,
        n_families_total=n_families,
        output_json_path=output_json_path,
        pre_split_pfam_pfamily_to_file_index=pre_split_pfam_pfamily_to_file_index,
    )
    print(f"Selected {len(selected_families)} families for val and test splits.")


    if not os.path.exists(flat_file_path):
        print("Creating parquet files for val / test...")
        make_val_test_parquets(
            selected_families=selected_families,
            parquet_save_dir=split_parquet_save_dir,
            flat_file_path=flat_file_path
        )


    # Remove validation and test families from the Pfam training data
    if not len(glob.glob(f"{split_parquet_save_dir}/train/*.parquet")):
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
    else:
        print("Train val test split of pfam training data already exists. Skipping...")

    print("Adding UniProt accessions to parquet files...")
    unmatched_names = add_accessions_to_parquets(
        split_parquet_save_dir,
        map_save_dir=map_save_dir,
        use_id_mapping_api=use_id_mapping_api,
    )

    print("Copying relevant files into the repo...")
    metadata_dir = "data/val_test/pfam/"
    os.makedirs(metadata_dir, exist_ok=True)
    val_test_fam_path = os.path.join(split_parquet_save_dir, "eval_families_filtered_w_unip_accs.csv")
    for f in [val_test_fam_path]:
        assert os.path.exists(f)
        logging.info(f"Copying {f} to {metadata_dir}")
        shutil.copy(f, metadata_dir)

    print("Combining val and test parquets...")
    combine_val_test_parquets(split_parquet_save_dir)

    print("Removing intermediate files...")
    remove_intermediate_files(split_parquet_save_dir)
    print("Done!")

