import os
import pandas as pd
import glob
import sys


class BufferWriter:
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
        seqs_mb = sum([sys.getsizeof(seq) / 1024 / 1024 for seq in df.sequences])
        if self.mem_use + seqs_mb < self.mem_limit:
            self.dfs.append(df)
            self.mem_use += seqs_mb
        else:
            self.write_dfs()
            self.dfs = [df]


    def write_dfs(self):
        if len(self.dfs):
            combi_df = pd.concat(self.dfs)
            filepath = os.path.join(self.outdir, f"{self.name}_{str(self.index).zfill(3)}.parquet")
            combi_df.to_parquet(filepath)
            self.dfs = []
            self.index += 1
            self.mem_use = 0

def remove_val_test_rows(index_file_path, val_test_csv_path, parquet_dir, output_dir, max_mb_per_entry=100):
    # Read the index file
    index_df = pd.read_csv(index_file_path)

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

    val_buffer = BufferWriter(val_dir, name="val", mem_limit=125)
    test_buffer = BufferWriter(test_dir, name="test", mem_limit=125)
    train_accs = set()
    # Process each parquet file
    for parquet_file in glob.glob(os.path.join(parquet_dir, '*.parquet')):
        t_path = os.path.join(train_dir, f"train_{os.path.basename(parquet_file)}")
        if os.path.exists(t_path):
            continue
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
            train_df.to_parquet(os.path.join(train_dir, f"train_{os.path.basename(parquet_file)}"))
    val_buffer.write_dfs()
    test_buffer.write_dfs()


if __name__ == "__main__":
    index_file_path = "../data/pfam/shuffled_parquets/index.csv"
    val_test_csv_path = "../data/pfam/pfam_eval_splits/pfam_val_test_accessions_w_unip_accs.csv"
    parquet_dir = "../data/pfam/shuffled_parquets/"
    output_dir = "../data/pfam/train_test_split_parquets"

    remove_val_test_rows(index_file_path, val_test_csv_path, parquet_dir, output_dir)