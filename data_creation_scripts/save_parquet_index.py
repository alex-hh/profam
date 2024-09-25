"""Save a mapping between document identifiers and parquet files.

Can be used to e.g. load a set of documents by accession (corresponding to a validation set for example)
(We could consider building separate parquets for holdout documents of course.)
"""
import argparse
import glob
import os
import pandas as pd
import tqdm


def main(args):
    # TODO: add number of pdbs
    data_dir = os.environ.get("DATA_DIR", "/SAN/orengolab/cath_plm/ProFam/data")
    with open(os.path.join(data_dir, args.data_folder, "index.csv"), "w") as f:
        files = glob.glob(os.path.join(data_dir, args.data_folder, "*.parquet"))
        print(f"Found {len(files)} parquet files in {os.path.join(data_dir, args.data_folder)}")
        f.write("identifier,parquet_file,cluster_size,sequence_length,num_pdb_ids\n")
        for file in tqdm.tqdm(files):
            try:
                df = pd.read_parquet(file)
            except Exception as e:
                print(f"Could not read {file}")
                raise e
            for _, row in df.iterrows():
                representative = row[args.identifier_col]
                try:
                    representative_index = list(row["accessions"]).index(representative)
                    representative_sequence = row["sequences"][representative_index]
                except:
                    print(f"Could not find representative {representative} in {row['accessions']} (file {file})")
                    representative_sequence = ""
                num_pdb_ids = len(row['pdb_ids']) if 'pdb_ids' in row and row['pdb_ids'] is not None else 0
                f.write(f"{row[args.identifier_col]},{os.path.basename(file)},{len(row['sequences'])},{len(representative_sequence)},{num_pdb_ids}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder")
    parser.add_argument("--identifier_col", default="fam_id")
    args = parser.parse_args()
    main(args)
