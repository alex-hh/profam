import os
import pandas as pd
import random

"""
load all of the pfams that are in the seed sequences
randomly sample some families for evaluation
create fasta files for the train/test split for these sequences
remove these families from the pfam training data
"""

def make_pfam_select_fam(pfam_select_fam_path):
    random.seed(42)
    n_families = 300
    pfam_families = set()
    for split_type in ['clustered_split', 'random_split']:
        for split in ['train', 'test']:
            dfs = []
            split_dir = os.path.join(pfam_dir, split_type, split)
            for fname in sorted(os.listdir(split_dir)):
                if fname.startswith('data'):
                    split_fam = pd.read_csv(os.path.join(split_dir, fname))
                    split_fam['split'] = split
                    split_fam['split_type'] = split_type
                    dfs.append(split_fam)
            combined = pd.concat(dfs)
            if len(pfam_families) == 0:
                pfam_families = set(combined['family_accession'].unique())
            else:
                pfam_families = pfam_families.intersection(
                    set(combined['family_accession'].unique())
                )

        print(f"number of families in train {len(combined[combined['split'] == 'train'].family_accession.unique())}")
        print(f"number of families in test {len(combined[combined['split'] == 'test'].family_accession.unique())}")
    print(f"Total number of families to sample from {len(pfam_families)}")
    selected_families = random.sample(pfam_families, n_families)
    with open(pfam_select_fam_path, 'w') as f:
        f.write('family_accession\n')
        for fam in selected_families:
            f.write(fam + '\n')


def make_pfam_eval_fastas(selected_families):
    for split_type in ['clustered_split', 'random_split']:
        save_dir = os.path.join(pfam_dir, f'{split_type}_fastas')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for split in ['train', 'test']:
            dfs = []
            split_dir = os.path.join(pfam_dir, split_type, split)
            for fname in sorted(os.listdir(split_dir)):
                if fname.startswith('data'):
                    split_fam = pd.read_csv(os.path.join(split_dir, fname))
                    split_fam = split_fam[split_fam['family_accession'].isin(selected_families)]
                    dfs.append(split_fam)
            combined = pd.concat(dfs)

            for fam in selected_families:
                fam_df = combined[combined['family_accession'] == fam]
                assert len(fam_df) > 0
                print(f"number of sequences in {fam} for {split} {len(fam_df)}")
                fasta_path = os.path.join(save_dir, f"{fam}_{split}.fasta")
                with open(fasta_path, 'w') as f:
                    for i, row in fam_df.iterrows():
                        f.write(f'>{row["family_accession"]}\n')
                        f.write(f'{row["aligned_sequence"]}\n')



def main():
    pfam_select_fam_path = os.path.join(pfam_dir, 'eval_families.csv')
    if not os.path.exists(pfam_select_fam_path):
        make_pfam_select_fam(pfam_select_fam_path)
    selected_families = pd.read_csv(pfam_select_fam_path)
    make_pfam_eval_fastas(selected_families['family_accession'].values)


if __name__=="__main__":
    pfam_dir = '../data/pfam/pfam_eval_splits'
    main()