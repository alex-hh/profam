import pandas as pd
import os
import numpy as np

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

For each val-test parquet we add the following columns:
- fam_class_prompt_sequences
- fam_class_accessions
- fam_class_completion_sequences_random
- fam_class_accessions_random
- fam_class_labels_random
- fam_class_completion_sequences_clustered
- fam_class_accessions_clustered
- fam_class_labels_clustered

do family classification for the following datasets:

FunFamsS50
"../data/funfams/s50_parquets/train_val_test_split/val"
"../data/funfams/s50_parquets/train_val_test_split/test"

FunFamsS100
"../data/funfams/s100_parquets/train_val_test_split/val"
"../data/funfams/s100_parquets/train_val_test_split/test"

TEDS50
../data/ted/s50_parquets/train_val_test_split/val
../data/ted/s50_parquets/train_val_test_split/test

FoldSeekS50
../data/foldseek/foldseek_s50_struct/train_val_test_split/val
../data/foldseek/foldseek_s50_struct/train_val_test_split/test
"""

class BaseFamClassConfig:
    def __init__(self, parquet_dir: str):
        self.parquet_dir = parquet_dir
        self.max_tokens_in_prompt = 10000
        self.n_completion_seqs_per_fam = 100
        self.max_tokens_in_single_completion = 2000

def create_fam_classification_data(config: BaseFamClassConfig):
    """Create family classification data for all parquet files in directory."""
    for parquet_file in os.listdir(config.parquet_dir):
        if not parquet_file.endswith('.parquet'):
            continue
            
        file_path = os.path.join(config.parquet_dir, parquet_file)
        df = pd.read_parquet(file_path)
        
        # Initialize new columns
        df['fam_class_prompt_sequences'] = None
        df['fam_class_accessions'] = None
        df['fam_class_completion_sequences_random'] = None
        df['fam_class_accessions_random'] = None
        df['fam_class_labels_random'] = None
        
        # Process each family
        for idx in df.index:
            fam_sequences = df.at[idx, 'sequences']
            fam_accessions = df.at[idx, 'accessions']
            
            # Split sequences for prompts and positive completions
            n_prompt_seqs = max(1, len(fam_sequences) // 2)  # Use half for prompts
            prompt_seqs = fam_sequences[:n_prompt_seqs]
            prompt_accs = fam_accessions[:n_prompt_seqs]
            
            # Get candidate completion sequences from this family
            pos_completion_candidates = fam_sequences[n_prompt_seqs:]
            pos_completion_accs = fam_accessions[n_prompt_seqs:]
            
            # Get negative completion sequences from other families
            neg_completion_mask = df.index != idx
            other_sequences = np.concatenate(df[neg_completion_mask]['sequences'].values)
            other_accessions = np.concatenate(df[neg_completion_mask]['accessions'].values)
            
            # Randomly sample completion sequences
            n_completions = min(config.n_completion_seqs_per_fam // 2, 
                              len(pos_completion_candidates),
                              len(other_sequences))
            
            pos_indices = np.random.choice(len(pos_completion_candidates), n_completions, replace=False)
            neg_indices = np.random.choice(len(other_sequences), n_completions, replace=False)
            
            completion_seqs = np.concatenate([
                pos_completion_candidates[pos_indices],
                other_sequences[neg_indices]
            ])
            completion_accs = np.concatenate([
                pos_completion_accs[pos_indices],
                other_accessions[neg_indices]
            ])
            completion_labels = np.concatenate([
                np.ones(n_completions),
                np.zeros(n_completions)
            ])
            
            # Store results
            df.at[idx, 'fam_class_prompt_sequences'] = prompt_seqs
            df.at[idx, 'fam_class_accessions'] = prompt_accs
            df.at[idx, 'fam_class_completion_sequences_random'] = completion_seqs
            df.at[idx, 'fam_class_accessions_random'] = completion_accs
            df.at[idx, 'fam_class_labels_random'] = completion_labels
        
        # Save updated parquet
        df.to_parquet(file_path)

if __name__ == "__main__":
    # Process each dataset
    datasets = [
        "../data/funfams/s50_parquets/train_val_test_split/val",
        "../data/funfams/s50_parquets/train_val_test_split/test",
        "../data/funfams/s100_parquets/train_val_test_split/val",
        "../data/funfams/s100_parquets/train_val_test_split/test",
        "../data/ted/s50_parquets/train_val_test_split/val",
        "../data/ted/s50_parquets/train_val_test_split/test",
        "../data/foldseek/foldseek_s50_struct/train_val_test_split/val",
        "../data/foldseek/foldseek_s50_struct/train_val_test_split/test"
    ]
    
    for dataset_path in datasets:
        print(f"Processing {dataset_path}")
        config = BaseFamClassConfig(dataset_path)
        create_fam_classification_data(config)
