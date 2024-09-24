"""Script to save an index file mapping UniProt accessions to parquet files."""
import argparse
import os
import pandas as pd
import pickle
from src.constants import PROFAM_DATA_DIR


def main(args):
    with open(os.path.join(PROFAM_DATA_DIR, "afdb/foldseek_cluster_dict.pkl"), "rb") as f:
        cluster_dict = pickle.load(f)
    print("Number of clusters:", len(cluster_dict))

    if args.include_af50:
        af50_dict_path = os.path.join(PROFAM_DATA_DIR, "afdb", "af50_cluster_dict.pkl")
        print("loading af50 dictionary")
        with open(af50_dict_path, "rb") as f:
            af50_dict = pickle.load(f)

    index_df = pd.read_csv(os.path.join(PROFAM_DATA_DIR, args.data_folder, "index.csv")).set_index("identifier")
    print(index_df.head())
    print("Number of indexed clusters", len(index_df))

    with open(os.path.join(PROFAM_DATA_DIR, args.data_folder, "accession_index.csv"), "w") as f:
        for cluster_id, member_ids in cluster_dict.items():
            cluster_accessions = []
            assert cluster_id in member_ids
            try:
                parquet_file = index_df.loc[cluster_id]["parquet_file"]
            except:
                print(f"Cluster {cluster_id} not found in index")
                continue
            if args.include_foldseek_members:
                for member_id in member_ids:
                    cluster_accessions.append(member_id)
                    if args.include_af50_members:
                        cluster_accessions += af50_dict.get(member_id, [])
            else:
                cluster_accessions.append(cluster_id)
            for member_id in cluster_accessions:
                f.write(f"{member_id},{parquet_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str)
    parser.add_argument("--include_foldseek_members", action="store_true")
    parser.add_argument("--include_af50_members", action="store_true")
    args = parser.parse_args()
    if args.include_af50_members:
        assert args.include_foldseek_members, "Must include foldseek members to include af50 members"
    main(args)
