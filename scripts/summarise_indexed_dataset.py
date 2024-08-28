"""TODO integrate with dataset configs so we can run with a dataset name"""
import argparse

import pandas as pd


def main(args):
    df = pd.read_csv(args.index_file_path)
    num_unique_clusters = len(df["identifier"].unique())
    maximum_cluster_size = df["cluster_size"].max()
    minimum_cluster_size = df["cluster_size"].min()
    if num_unique_clusters != len(df):
        print("WARNING: some clusters are duplicated in the index file.")
    print(f"Total clusters {len(df)} (unique clusters: {num_unique_clusters})")
    print(f"Maximum cluster size: {maximum_cluster_size}")
    print(f"Minimum cluster size: {minimum_cluster_size}")
    print(f"Number of empty clusters: {len(df[df['cluster_size'] == 0])}")
    print(
        f"Mean cluster size {df['cluster_size'].mean()}, median {df['cluster_size'].median()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file_path")
    args = parser.parse_args()
    main(args)
