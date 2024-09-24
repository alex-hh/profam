"""Add PDB coordinates to foldseek parquet files, where available.

Select 1 PDB structure for each UniProt accession (highest coverage).
"""
import argparse
import os
import pandas as pd
from src.constants import PROFAM_DATA_DIR
from src.tools.sifts import SIFTS


def main(args):
    # if full load requires too much memory we could use some sed command
    # or something like that to build a temp file for specific parquet index
    # or use lmdb or similar...
    df = pd.read_csv(os.path.join(PROFAM_DATA_DIR, args.data_folder, "accession_index.csv"))
    if args.parquet_index is None:
        parquet_files = df["parquet_file"].unique()
    else:
        parquet_files = [f"{args.parquet_index}.parquet"]

    print("Number of parquet files:", len(parquet_files))
    for parquet_file in parquet_files:
        # TODO: check this preserves float 16
        parquet_accession_df = df[df["parquet_file"] == parquet_file]
        accessions_with_pdbs = set(parquet_accession_df["accession"].unique())
        parquet_df = pd.read_parquet(os.path.join(PROFAM_DATA_DIR, args.data_folder, parquet_file))
        parquet_df["pdb_ids"] = None

        def annotate_row_with_pdbs(row):
            accessions = row["accessions"]
            pdb_ids = []
            pdb_N = []
            pdb_CA = []
            pdb_C = []
            pdb_O = []
            for accession in accessions:
                if accession in accessions_with_pdbs:
                    sifts = SIFTS(sifts_table_file=os.path.join(PROFAM_DATA_DIR, "sifts", "evcouplings_sifts.csv"))
                    hits = sifts.by_uniprot_id(accession, reduce_chains=True)
                    if len(hits) > 1:
                        # select hit with highest coverage
                        # TODO: if I use from_id, does this save a pdb file anywhere?
                        pass
                else:
                    pdb_ids.append(None)
                    pdb_N.append(None)
                    pdb_CA.append(None)
                    pdb_C.append(None)
                    pdb_O.append(None)
            row["pdb_ids"] = pdb_ids
            row["pdb_N"] = pdb_N
            row["pdb_CA"] = pdb_CA
            row["pdb_C"] = pdb_C
            row["pdb_O"] = pdb_O
            return row
        parquet_df = parquet_df.apply(annotate_row_with_pdbs, axis=1)
        parquet_df.to_parquet(os.path.join(PROFAM_DATA_DIR, args.data_folder, parquet_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str)
    parser.add_argument("--parquet_index", type=str)
    args = parser.parse_args()
    main(args)
