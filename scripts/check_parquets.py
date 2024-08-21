"""
Iterate through all the parquet files
and make sure that there is some minimum
text in the text column
"""
import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

if __name__ == "__main__":
    foldseek_dir = "../data/foldseek"
    pfam_dom_dir = "../data/pfam/parquets/Domain"
    pfam_fam_dir = "../data/pfam/parquets/Family"

    for d in [foldseek_dir, pfam_dom_dir, pfam_fam_dir]:
        parquet_files = glob.glob(os.path.join(d, "*.parquet"))
        print(f"Found {len(parquet_files)} parquet files in {d}")
        for parquet_file in tqdm(parquet_files):
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            assert "text" in df.columns

            if not df["text"].apply(lambda x: len(x) > 0).all():
                orig_len = len(df)
                # remove empty text rows
                df = df[df["text"].apply(lambda x: len(x) > 5)]
                print(f"Removed {orig_len - len(df)} rows from {parquet_file}")
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, parquet_file)
