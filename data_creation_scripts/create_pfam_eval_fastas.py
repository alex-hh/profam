import os
import pandas as pd
import random

"""
clustered splits from:
console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam
referrenced in:
https://www.nature.com/articles/s41587-021-01179-w#Sec4

Following directories have aligned sequence dfs:
'../data/pfam/pfam_eval_splits/clustered_split/train' (1,296,280 seqs, 17,929 fams)
'../data/pfam/pfam_eval_splits/clustered_split/test' (21,293 seqs, 3097 fams)

for each pfam accession that is included in the validation or test set
 'train' sequences are used in the prompt and test seqs used for completions 

the families are further split into val (used for hparam tuning) and test sets

fastas are saved to:
../data/pfam/pfam_eval_splits/{val/test}/{clustered_split/random_split}/{fam}_{test/train}.fasta
"""

import os
import pandas as pd
import random



def make_pfam_select_fam(pfam_select_fam_path, n_families=500):
    """
    Select families that occur in train AND test for
    BOTH clustered AND random
    """
    random.seed(42)
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
            print(f"number of families in {split_type} {split}: {len(combined.family_accession.unique())}")

    print(f"Total number of families to sample from {len(pfam_families)}")
    pfam_families = sorted(list(pfam_families))
    selected_families = random.sample(pfam_families, n_families)

    # Split selected families into validation and test sets
    val_families = selected_families[:250]
    test_families = selected_families[250:]

    with open(pfam_select_fam_path, 'w') as f:
        f.write('family_accession,split\n')
        for fam in val_families:
            f.write(f'{fam},val\n')
        for fam in test_families:
            f.write(f'{fam},test\n')


def make_pfam_eval_fastas(selected_families, index_csv_path):
    index_rows = []
    for split_type in ['clustered_split', 'random_split']:
        for eval_split in ['val', 'test']:
            save_dir = os.path.join(pfam_dir, f'{eval_split}/{split_type}_fastas')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            eval_families = selected_families[selected_families['split'] == eval_split]['family_accession'].values

            for split in ['train', 'test']:
                dfs = []
                split_dir = os.path.join(pfam_dir, split_type, split)
                for fname in sorted(os.listdir(split_dir)):
                    if fname.startswith('data'):
                        split_fam = pd.read_csv(os.path.join(split_dir, fname))
                        split_fam = split_fam[split_fam['family_accession'].isin(eval_families)]
                        dfs.append(split_fam)
                combined = pd.concat(dfs)

                for fam in eval_families:
                    fam_df = combined[combined['family_accession'] == fam]
                    assert len(fam_df) > 0
                    print(f"number of sequences in {fam} for {split_type} {split} {eval_split}: {len(fam_df)}")
                    fasta_path = os.path.join(save_dir, f"{fam}_{split}.fasta")
                    with open(fasta_path, 'w') as f:
                        for i, row in fam_df.iterrows():
                            f.write(f'>{row.sequence_name}_{row["family_accession"]}\n')
                            f.write(f'{row["aligned_sequence"].replace(".", "-")}\n')
                            if '-' in row.aligned_sequence:
                                bp=1
                            index_rows.append({
                                "fam_id": row.family_accession,
                                "accession": row.sequence_name.split("_")[0],
                                "sequence_name": row.sequence_name,
                                "split": eval_split,
                            })
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(os.path.join(pfam_dir, index_csv_path), index=False)


def add_uniprot_accessions_to_csv(csv_path, mapping_path, output_path):

    df = pd.read_csv(csv_path)
    df["accession"] = df.sequence_name.apply(lambda x: x.split("/")[0])

    mapping = pd.read_csv(mapping_path, delimiter="\t")
    df2 = df.join(mapping.set_index("From", drop=False), on="accession", rsuffix="_up_map", how="left")
    print(f"Found uniprot accessions for {df2.Entry.notnull().sum()} out of {len(df2)}")
    df2.to_csv(output_path, index=False)

def sample_fams_make_fastas(pfam_dir, index_csv_path, mapping_path):
    pfam_select_fam_path = os.path.join(pfam_dir, 'eval_families_500.csv')
    n_families = 500
    if not os.path.exists(pfam_select_fam_path):
        make_pfam_select_fam(pfam_select_fam_path, n_families=n_families)
    selected_families = pd.read_csv(pfam_select_fam_path)

    if not os.path.exists(index_csv_path):
        make_pfam_eval_fastas(selected_families, index_csv_path=index_csv_path)
    add_uniprot_accessions_to_csv(
        index_csv_path,
        mapping_path,
        output_path=index_csv_path.replace(".csv", "_w_unip_accs.csv")
    )


if __name__ == "__main__":
    pfam_dir = '../data/pfam/pfam_eval_splits'
    index_csv_path = os.path.join(pfam_dir, "pfam_val_test_accessions.csv")
    mapping_path = os.path.join(pfam_dir, "val_test_uniprot_idmapping_2024_08_22.tsv")

    sample_fams_make_fastas(
        pfam_dir=pfam_dir,
        index_csv_path=index_csv_path,
        mapping_path=mapping_path,
    )