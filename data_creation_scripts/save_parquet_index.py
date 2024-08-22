"""Save a mapping between document identifiers and parquet files.

Can be used to e.g. load a set of documents by accession (corresponding to a validation set for example)
(We could consider building separate parquets for holdout documents of course.)
"""
import argparse
import glob
import os
import pandas as pd


def main(args):
    with open(args.index_file_path, "w") as f:
        files = glob.glob(args.data_file_pattern)
        f.write("identifier,parquet_file\n")
        for file in files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                f.write(f"{row[args.identifier_col]},{os.path.basename(file)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file_path")
    parser.add_argument("data_file_pattern")
    parser.add_argument("--identifier_col", default="cluster_id")
    args = parser.parse_args()
    main(args)
