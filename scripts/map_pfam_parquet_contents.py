"""
map which pfam families are in each parquet file
calculate the size in bytes of each family
create a shuffled list of all families
chunk the families into parquets that have
roughly equal size in bytes.
save the new shuffled and resized parquets.
"""
import os
import glob
import sys
import pandas as pd
import pickle


def create_parquet_map(pfam_parquet_dir, mapping_path):
    df_rows = []
    for parquet_file in glob.glob(f"{pfam_parquet_dir}/*.parquet"):
        df = pd.read_parquet(parquet_file)
        for i, row in df.iterrows():
            fam_id = row["pfam_acc"]
            size_mb = sys.getsizeof(row["text"]) * 1024 * 1024
            fname = os.path.basename(parquet_file)
            df_rows.append({"pfam_acc": fam_id, "size_mb": size_mb, "parquet_file": fname})
    df = pd.DataFrame(df_rows)
    total_rows = df.shape[0]
    print(f"total rows from all parquets: {total_rows}")
    df.to_csv("/SAN/orengolab/cath_plm/ProFam/data/pfam/old_pfam_parquet_map.csv", index=False)
    df = df.sample(frac=1).reset_index(drop=True)
    limit_mb = 250
    parq_ix = 0
    current_size = 0
    new_mapping = {parq_ix: {}}
    for i, row in df.iterrows():
        fam_id = row["pfam_acc"]
        size_mb = row["size_mb"]
        parquet_file = row["parquet_file"]
        if parquet_file not in new_mapping[parq_ix]:
            new_mapping[parq_ix][parquet_file] = []
        new_mapping[parq_ix][parquet_file].append(fam_id)
        current_size += size_mb
        if current_size > limit_mb:
            print(f"parquet {parq_ix} size: {current_size}")
            parq_ix += 1
            current_size = 0
            new_mapping[parq_ix] = {}
    with open(mapping_path, "wb") as f:
        pickle.dump(new_mapping, f)

if __name__=="__main__":
    pfam_parquet_dir = "/SAN/orengolab/cath_plm/ProFam/data/pfam/combined_parquets"
    mapping_path = "/SAN/orengolab/cath_plm/ProFam/data/pfam/new_pfam_parquet_map.pkl"
    if not os.path.exists(mapping_path):
        create_parquet_map(pfam_parquet_dir, mapping_path)

