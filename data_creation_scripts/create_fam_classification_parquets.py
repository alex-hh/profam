import pandas as pd
import glob
import os
import numpy as np
import json
"""
Created by Jude Wells 2024-12-26

Creates custom parquet files with prompt sequences 
and completion sequences for family classification.
The idea is that we have a prompt of sequences from 
the family and then we condition these and calculate 
the likelihood of the completion sequences: 50% of 
which are from the family and 50% of which are not 
from the family. We evaluate how good the conditional 
likelihood score is for binary classification 
in-family or not.

Single parquet file for each dataset.

in-family and out-of-family completion sequences are
sampled from the same set of sequences to mitigate
the problem of intrinsically likely sequences
(irrespective of prompt conditioning).
"""

class BaseFamClassConfig:
    def __init__(self, parquet_dir: str, identifier_col: str = 'fam_id'):
        self.parquet_dir = parquet_dir
        self.max_tokens_in_prompt = 20000 # gaps are counted
        self.n_completion_seqs_per_fam = 100
        self.max_tokens_in_single_completion = 2000
        self.max_families_per_dataset = 300
        self.min_seqs_in_family = 2
        self.identifier_col = identifier_col
        self.random_seed = 42
    
    def write_config_to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

def print_parquet_report(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    print(f"Parquet file: {parquet_path}")
    print(f"Number of rows: {len(df)}")
    df["n_completions"] = df.completion_sequences_random.apply(len)
    print(f"Mean number of completions: {round(df['n_completions'].mean(), 3)}")
    print(f"Minimum number of completions: {df['n_completions'].min()}")
    print(f"Maximum number of completions: {df['n_completions'].max()}")
    df["n_prompt_sequences"] = df.prompt_sequences.apply(len)
    print(f"Mean number of prompt sequences: {round(df['n_prompt_sequences'].mean(), 3)}")
    print(f"Minimum number of prompt sequences: {df['n_prompt_sequences'].min()}")
    print(f"Maximum number of prompt sequences: {df['n_prompt_sequences'].max()}")
    df["proportion_in_family"] = df.fam_class_labels_random.apply(lambda x: (x == 1).sum() / len(x))
    print(f"Mean proportion of in-family completions: {round(df['proportion_in_family'].mean(), 3)}")
    print(f"Minimum proportion of in-family completions: {df['proportion_in_family'].min()}")
    print(f"Maximum proportion of in-family completions: {df['proportion_in_family'].max()}")
    df["tokens_in_prompt"] = df.prompt_sequences.apply(lambda x: sum(len(seq) for seq in x))
    print(f"Mean tokens in prompt: {round(df['tokens_in_prompt'].mean(), 3)}")
    print(f"Minimum tokens in prompt: {df['tokens_in_prompt'].min()}")
    print(f"Maximum tokens in prompt: {df['tokens_in_prompt'].max()}")
    print("\n")



def sample_families(config: BaseFamClassConfig):
    families_to_sample_from = []
    parquet_paths = glob.glob(os.path.join(config.parquet_dir, '*.parquet'))
    oversample_buffer = int(config.max_families_per_dataset * 0.5)
    max_families_to_sample = int(config.max_families_per_dataset + oversample_buffer)
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        df['n_seqs'] = df.sequences.apply(len)
        df = df[df['n_seqs'] >= config.min_seqs_in_family]
        families_to_sample_from.extend(df[config.identifier_col].unique())
    return np.random.choice(families_to_sample_from, min(max_families_to_sample, len(families_to_sample_from)), replace=False)

def sample_family_data(config: BaseFamClassConfig, sampled_families):
    """Gather sequences and accessions for all sampled families across parquet files."""
    family_data = {}  # {fam_id: {'sequences': [], 'accessions': []}}
    parquet_paths = glob.glob(os.path.join(config.parquet_dir, '*.parquet'))
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        df = df[df[config.identifier_col].isin(sampled_families)]
        
        for _, row in df.iterrows():
            fam_id = row[config.identifier_col]
            if fam_id not in family_data:
                family_data[fam_id] = {
                    'prompt_sequences': [], 
                    'prompt_accessions': [],
                    'completion_sequences_positive': [],
                    'completion_accessions_positive': [],                
                    }
            shuffled_indices = np.random.permutation(len(row['sequences']))
            # some fam_id may be in more than one parquet, so we need to keep track of the cumulative prompt length
            cumulative_prompt_length = sum([len(seq) for seq in family_data[fam_id]['prompt_sequences']])
            for i, idx in enumerate(shuffled_indices):
                if i % 2 == 0:
                    if cumulative_prompt_length + len(row['sequences'][idx]) <= config.max_tokens_in_prompt:
                        family_data[fam_id]['prompt_sequences'].append(row['sequences'][idx])
                        family_data[fam_id]['prompt_accessions'].append(row['accessions'][idx])
                        cumulative_prompt_length += len(row['sequences'][idx])
                else:
                    if len(family_data[fam_id]['completion_sequences_positive']) < config.n_completion_seqs_per_fam // 2:
                        family_data[fam_id]['completion_sequences_positive'].append(row['sequences'][idx])
                        family_data[fam_id]['completion_accessions_positive'].append(row['accessions'][idx])
                
                if len(family_data[fam_id]['prompt_sequences']) > config.max_tokens_in_prompt:
                    raise ValueError(
                        f"Prompt sequences for {fam_id} exceed max tokens in prompt: {len(family_data[fam_id]['prompt_sequences'])}"
                        )

    
    for fam_id in family_data: # do this last to handle case where fam_id is in more than one parquet
        family_data[fam_id]  = {k: np.array(v) for k,v in family_data[fam_id].items()}
        if any(len(v) == 0 for v in family_data[fam_id].values()):
            del family_data[fam_id]
    return family_data

def get_negative_completion_sequences(family_data, fam_id):
    """
    Use positive completion sequences from other families as 
    negative completion sequences for the current family.
    """
    other_fams = {k:v for k,v in family_data.items() if k != fam_id}
    neg_sequences = []
    neg_accessions = []
    for fam_id, fam_data in other_fams.items():
        neg_sequences.extend(fam_data['completion_sequences_positive'])
        neg_accessions.extend(fam_data['completion_accessions_positive'])
    neg_sequences = np.array(neg_sequences)
    neg_accessions = np.array(neg_accessions)
    return neg_sequences, neg_accessions

def create_fam_classification_data(config: BaseFamClassConfig):
    """Create family classification data for all parquet files in directory."""
    np.random.seed(config.random_seed)
    # First pass: sample families
    sampled_families = sample_families(config)

    # Second pass: sample sequences for each sampled family
    family_data = sample_family_data(config, sampled_families)

    # sample negative completion sequences for each fam from the positive completion sequences of other families
    new_rows = []
    output_path = os.path.join(config.parquet_dir, "family_classification", f"family_classification.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for fam_id, fam_data in family_data.items():
        new_row = {
            config.identifier_col: fam_id,
            'prompt_sequences': fam_data['prompt_sequences'],
            'prompt_accessions': fam_data['prompt_accessions']
        }
        neg_sequences, neg_accessions = get_negative_completion_sequences(family_data, fam_id)
        n_negative_completions = min(len(neg_sequences), len(fam_data['completion_sequences_positive']))
        neg_indices = np.random.choice(len(neg_sequences), n_negative_completions, replace=False)
        neg_completion_seqs = neg_sequences[neg_indices]
        neg_completion_accs = neg_accessions[neg_indices]
        new_row['completion_sequences_random'] = np.concatenate([
            fam_data['completion_sequences_positive'], 
            neg_completion_seqs
        ])
        new_row['completion_accessions_random'] = np.concatenate([
            fam_data['completion_accessions_positive'], 
            neg_completion_accs
        ])
        new_row['fam_class_labels_random'] = np.concatenate([
            np.ones(len(fam_data['completion_sequences_positive'])), 
            np.zeros(n_negative_completions)
        ])
        new_rows.append(new_row)
    if len(new_rows) > config.max_families_per_dataset: # due to oversample_buffer
        new_rows = list(np.random.choice(new_rows, config.max_families_per_dataset, replace=False))
    fam_class_df = pd.DataFrame(new_rows)
    fam_class_df.to_parquet(output_path)
    json_save_path = os.path.join(
        os.path.dirname(output_path),
        "fam_class_config.json"
    )
    config.write_config_to_json(json_save_path)
    print_parquet_report(output_path)

if __name__ == "__main__":
    # Process each dataset
    datasets = [
        "../data/funfams/s50_parquets/train_val_test_split/val",
        "../data/funfams/s50_parquets/train_val_test_split/test",
        "../data/funfams/s100_noali_parquets/train_val_test_split/val",
        "../data/funfams/s100_noali_parquets/train_val_test_split/test",
        "../data/ted/s50_parquets/train_val_test_split/val",
        "../data/ted/s50_parquets/train_val_test_split/test",
        "../data/foldseek/foldseek_s50_struct/train_val_test_split/val",
        "../data/foldseek/foldseek_s50_struct/train_val_test_split/test"
    ]
    
    for dataset_path in datasets:
        output_path = os.path.join(config.parquet_dir, "family_classification", f"family_classification.parquet")
        if os.path.exists(output_path):
            print(f"Skipping {dataset_path} because it already exists")
            continue
        print(f"Processing {dataset_path}")
        config = BaseFamClassConfig(dataset_path)
        create_fam_classification_data(config)
