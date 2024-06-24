import glob
import pandas as pd


fpaths = glob.glob("../data/openfold/uniclust30_filtered_parquet/*.parquet")

for fpath in fpaths:
    # Load the Parquet file
    df = pd.read_parquet(fpath)
    # Rename the 'content' column to 'text'
    df = df.rename(columns={'content': 'text'})
    # Save the DataFrame back to a Parquet file
    df.to_parquet(fpath, index=False)