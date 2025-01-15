"""
Iterates over parquet files 
and creates new parquet files where there is none of the 
structural data. Columns to keep:
- "sequences"
- "fam_id"
- "accessions"
- "af50_cluster_id"
"""

import pandas as pd
import os
parquet_dir = "../data/afdb_s50_single/train_val_test_split"
new_parquet_dir = "../data/afdb_s50_single_seq_only/train_val_test_split"
os.makedirs(new_parquet_dir, exist_ok=True)
for split_dir in ["train", "val", "test"]:
    save_path = os.path.join(new_parquet_dir, split_dir)
    os.makedirs(save_path, exist_ok=True)
    for file in os.listdir(os.path.join(parquet_dir, split_dir)):
        df = pd.read_parquet(os.path.join(parquet_dir, split_dir, file))
        df = df[["sequences", "fam_id", "accessions", "af50_cluster_id"]]
        df.to_parquet(os.path.join(save_path, file), index=False)


