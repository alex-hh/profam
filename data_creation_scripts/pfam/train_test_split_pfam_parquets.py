import csv
import os
import pandas as pd
import glob
import sys

"""
Created by Jude Wells 2024-08-24
this script splits pfam data 
(assumed already in parquet format)
into train/val/test splits

"""

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
        increment mem use, if over mem_limit
        write dfs to parquet and reset buffer
        """
        seqs_mb = sum([sys.getsizeof(s) for i, s_ls in df.sequences.items() for s in s_ls]) / 1024 / 1024
        print(f"mem use {self.name}: {round(self.mem_use, 6)}")
        if self.mem_use + seqs_mb < self.mem_limit:
            self.dfs.append(df)
            self.mem_use += seqs_mb

        else:
            self.write_dfs()
            self.dfs = [df]
            self.mem_use = seqs_mb


    def write_dfs(self):
        if len(self.dfs):
            combi_df = pd.concat(self.dfs)
            filepath = os.path.join(self.outdir, f"{self.name}_{str(self.index).zfill(3)}.parquet")
            print(f"writing to {filepath}")
            combi_df.to_parquet(filepath)
            new_index.extend([(row['fam_id'], os.path.basename(filepath)) for _, row in combi_df.iterrows()])
            self.dfs = []
            self.index += 1
            self.mem_use = 0

def remove_val_test_rows(val_test_csv_path, parquet_dir, output_dir, mem_limit=125):
    # Read the validation and test family IDs
    val_test_df = pd.read_csv(val_test_csv_path)
    val_test_fam_ids = set(val_test_df.fam_id.apply(lambda x: x.split(".")[0]))
    val_fam_ids = set(val_test_df[val_test_df.split=="val"].fam_id.apply(lambda x: x.split(".")[0]))
    test_fam_ids = set(val_test_df[val_test_df.split=="test"].fam_id.apply(lambda x: x.split(".")[0]))

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
    # Process each parquet file
    for parquet_file in glob.glob(os.path.join(parquet_dir, '*.parquet')):
        t_path = os.path.join(train_dir, f"train_{os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        df["pfam_version"] = df.fam_id.apply(lambda x: x.split(".")[1])
        df["fam_id"] = df.fam_id.apply(lambda x: x.split(".")[0])
        train_df = df[~df['fam_id'].isin(val_test_fam_ids)]
        val_df = df[df.fam_id.isin(val_fam_ids)]
        test_df = df[df.fam_id.isin(test_fam_ids)]
        if len(val_df):
            val_buffer.update_buffer(val_df)
        if len(test_df):
            test_buffer.update_buffer(test_df)

        # Save train data
        if not train_df.empty:
            train_accs.update(set(train_df.fam_id))
            train_df.to_parquet(t_path)
            new_index.extend([(row['fam_id'], os.path.basename(t_path)) for _, row in train_df.iterrows()])
    val_buffer.write_dfs()
    test_buffer.write_dfs()


if __name__ == "__main__":
    new_index = []
    val_test_csv_path = "../data/pfam/pfam_eval_splits/pfam_val_test_accessions_w_unip_accs.csv"
    parquet_dir = "../data/pfam/shuffled_parquets/"
    output_dir = "../data/pfam/train_test_split_parquets"

    remove_val_test_rows(
        val_test_csv_path=val_test_csv_path,
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        mem_limit=125,
    )
    # Write new index.csv
    with open(os.path.join(output_dir, 'new_index.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fam_id', 'parquet_file'])
        writer.writerows(new_index)

