import os
import pandas as pd
import random
import json
import glob
import sys
import csv
from collections import defaultdict

"""
Consolidated script for Pfam data processing.

This script performs the following:

1. Selects Pfam families that occur in both train and test splits for both clustered and random splits, excluding families with more than 10,000 members or fewer than 10 members, and families where the UniProt IDs are not present in the provided JSON mapping.

2. Creates evaluation FASTA files for the selected families.

3. Removes validation and test families from the Pfam training data parquet files and splits the data into train, validation, and test sets.

4. Generates the 'pfam_val_test_all_up_ids.json' file mapping Pfam families to UniProt IDs.

Usage:
    python consolidated_pfam_processing.py
"""

def make_pfam_select_fam(pfam_select_fam_path, pfam_uniprot_json_path, n_families=500):
    """
    Select families that occur in train AND test for BOTH clustered AND random splits,
    exclude families with more than 10,000 members or fewer than 10 members,
    and exclude families where the UniProt IDs are not included in the provided JSON.
    """
    random.seed(42)
    pfam_families = set()
    family_counts = {}
    pfam_uniprot_mappings = json.load(open(pfam_uniprot_json_path))

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
            fam_counts = combined['family_accession'].value_counts().to_dict()
            if not family_counts:
                family_counts = fam_counts
            else:
                # Update counts to minimum across splits
                for fam_id in family_counts:
                    if fam_id in fam_counts:
                        family_counts[fam_id] = min(family_counts[fam_id], fam_counts.get(fam_id, 0))
                    else:
                        family_counts[fam_id] = 0
            if len(pfam_families) == 0:
                pfam_families = set(combined['family_accession'].unique())
            else:
                pfam_families = pfam_families.intersection(
                    set(combined['family_accession'].unique())
                )
            print(f"Number of families in {split_type} {split}: {len(combined.family_accession.unique())}")

    print(f"Total number of families to sample from (before filtering): {len(pfam_families)}")

    # Filter families based on criteria
    filtered_families = []
    for fam_id in pfam_families:
        count = family_counts.get(fam_id, 0)
        if 10 <= count <= 10000:
            if fam_id in pfam_uniprot_mappings and len(pfam_uniprot_mappings[fam_id]) > 0:
                filtered_families.append(fam_id)

    print(f"Total number of families after filtering: {len(filtered_families)}")

    # Ensure we have enough families to sample
    if len(filtered_families) < n_families:
        n_families = len(filtered_families)
        print(f"Reduced number of families to sample to {n_families} due to filtering.")

    selected_families = random.sample(filtered_families, n_families)

    # Split selected families into validation and test sets
    val_families = selected_families[:n_families // 2]
    test_families = selected_families[n_families // 2:]

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
                if dfs:
                    combined = pd.concat(dfs)
                else:
                    continue

                for fam in eval_families:
                    fam_df = combined[combined['family_accession'] == fam]
                    if fam_df.empty:
                        continue
                    print(f"Number of sequences in {fam} for {split_type} {split} {eval_split}: {len(fam_df)}")
                    fasta_path = os.path.join(save_dir, f"{fam}_{split}.fasta")
                    with open(fasta_path, 'w') as f:
                        for i, row in fam_df.iterrows():
                            f.write(f'>{row.sequence_name}_{row["family_accession"]}\n')
                            f.write(f'{row["aligned_sequence"].replace(".", "-")}\n')
                            index_rows.append({
                                "fam_id": row.family_accession,
                                "accession": row.sequence_name.split("/")[0],
                                "sequence_name": row.sequence_name,
                                "split": eval_split,
                            })
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(os.path.join(pfam_dir, index_csv_path), index=False)

def add_uniprot_accessions_to_csv(csv_path, mapping_path, output_path):
    df = pd.read_csv(csv_path)

    mapping = pd.read_csv(mapping_path, delimiter="\t", names=["From", "Entry"], usecols=[0, 1])
    df2 = df.join(mapping.set_index("From", drop=False), on="accession", rsuffix="_up_map", how="left")
    print(f"Found UniProt accessions for {df2.Entry.notnull().sum()} out of {len(df2)}")
    df2.to_csv(output_path, index=False)

def remove_val_test_rows(val_test_csv_path, parquet_dir, output_dir, mem_limit=125):
    # Read the validation and test family IDs
    val_test_df = pd.read_csv(val_test_csv_path)
    val_test_fam_ids = set(val_test_df.fam_id.apply(lambda x: x.split(".")[0]))
    val_fam_ids = set(val_test_df[val_test_df['split'] == "val"].fam_id.apply(lambda x: x.split(".")[0]))
    test_fam_ids = set(val_test_df[val_test_df['split'] == "test"].fam_id.apply(lambda x: x.split(".")[0]))

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    val_buffer = ParquetBufferWriter(val_dir, name="val", mem_limit=mem_limit)
    test_buffer = ParquetBufferWriter(test_dir, name="test", mem_limit=mem_limit)
    train_accs = set()
    global new_index
    new_index = []

    # Process each parquet file
    for parquet_file in glob.glob(os.path.join(parquet_dir, '*.parquet')):
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
        seqs_mb = sum([sys.getsizeof(s) for s in df['sequences']]) / (1024 * 1024)
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

def create_pfam_val_test_all_up_ids_json(val_test_csv_path, pfam_uniprot_json_path, output_json_path):
    df = pd.read_csv(val_test_csv_path)
    print(f"Loaded Pfam val test CSV with {len(df)} rows")
    pfam_to_uniprot = json.load(open(pfam_uniprot_json_path))
    null_counter = 0
    fam_to_up = defaultdict(set)
    proportion_in_json = []
    proportion_by_fam = {}

    for i, row in df.iterrows():
        fam_id = row['fam_id'].split(".")[0]
        json_uniprot = set(pfam_to_uniprot.get(fam_id, []))
        if pd.isnull(row['Entry']):
            fam_to_up[fam_id] = json_uniprot
            null_counter += 1
        else:
            csv_uniprot = set(str(row['Entry']).split(','))
            # Combine UniProt IDs from both sources
            combined_uniprot = csv_uniprot.union(json_uniprot)
            fam_to_up[fam_id] = combined_uniprot

            # Sense check
            in_both = csv_uniprot.intersection(json_uniprot)
            hit_prop = len(in_both) / len(csv_uniprot) if len(csv_uniprot) > 0 else 0
            proportion_in_json.append(hit_prop)
            if fam_id not in proportion_by_fam:
                proportion_by_fam[fam_id] = []
            proportion_by_fam[fam_id].append(hit_prop)

    with open(output_json_path, "w") as f:
        json.dump({k: list(v) for k, v in fam_to_up.items()}, f, indent=2)
    print(f"Null counter: {null_counter} of {len(df)}")
    print(f"Complete: {len([i for i in proportion_in_json if i == 1])}")
    print(f"Incomplete: {len([i for i in proportion_in_json if i != 1])}")
    family_props = {k: sum(v) / len(v) for k, v in proportion_by_fam.items()}
    print(f"Num families with 0 matches: {len([i for i in family_props.values() if i == 0])}")
    print(f"Num families with more than 85% matches: {len([i for i in family_props.values() if i > 0.85])}")
    
def sample_fams_make_fastas(pfam_dir, index_csv_path, mapping_path, pfam_uniprot_json_path):
    pfam_select_fam_path = os.path.join(pfam_dir, 'eval_families_filtered.csv')
    n_families = 500
    if not os.path.exists(pfam_select_fam_path):
        make_pfam_select_fam(pfam_select_fam_path, pfam_uniprot_json_path, n_families=n_families)
    selected_families = pd.read_csv(pfam_select_fam_path)

    if not os.path.exists(index_csv_path):
        make_pfam_eval_fastas(selected_families, index_csv_path=index_csv_path)

    index_w_accessions_path = index_csv_path.replace(".csv", "_w_unip_accs.csv")
    if not os.path.exists(index_w_accessions_path):
        add_uniprot_accessions_to_csv(
            index_csv_path,
            mapping_path,
            output_path=index_w_accessions_path
        )
    else:
        print(f"Index with UniProt accessions already exists at {index_w_accessions_path}")

if __name__ == "__main__":
    pfam_dir = 'data/val_test/pfam'
    index_csv_filename = "pfam_val_test_accessions.csv"
    index_csv_path = os.path.join(pfam_dir, index_csv_filename)
    mapping_path = os.path.join(pfam_dir, "val_test_uniprot_idmapping_2024_08_22.tsv")
    pfam_uniprot_json_path = "../data/pfam/pfam_uniprot_mappings.json"
    output_json_path = "../data/pfam/pfam_val_test_all_up_ids.json"
    parquet_dir = "../data/pfam/shuffled_parquets/"
    output_dir = "../data/pfam/train_test_split_parquets"

    sample_fams_make_fastas(
        pfam_dir=pfam_dir,
        index_csv_path=index_csv_path,
        mapping_path=mapping_path,
        pfam_uniprot_json_path=pfam_uniprot_json_path
    )

    val_test_csv_path = index_csv_path.replace(".csv", "_w_unip_accs.csv")

    if not os.path.exists(output_json_path):
        create_pfam_val_test_all_up_ids_json(
            val_test_csv_path=val_test_csv_path,
            pfam_uniprot_json_path=pfam_uniprot_json_path,
            output_json_path=output_json_path
        )
    else:
        print(f"Pfam val test all UniProt IDs JSON already exists at {output_json_path}")

    remove_val_test_rows(
        val_test_csv_path=val_test_csv_path,
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        mem_limit=125,
    )

    # Write new index.csv
    index_csv_output = os.path.join(output_dir, 'new_index.csv')
    with open(index_csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fam_id', 'parquet_file'])
        writer.writerows(new_index)
    print(f"New index CSV saved to {index_csv_output}")