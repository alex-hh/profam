import glob
import pandas as pd


fpaths = glob.glob("../data/openfold/uniclust30_filtered_parquet/*.parquet")

for fpath in fpaths:
    # Load the Parquet file
    df = pd.read_parquet(fpath)
    df = df.drop(columns=['file_name'])
    df.to_parquet(fpath, index=False)