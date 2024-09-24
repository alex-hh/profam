"""Add PDB coordinates to foldseek parquet files, where available.

Select 1 PDB structure for each UniProt accession (highest coverage).
"""
import argparse
import os
import pandas as pd
from src.constants import PROFAM_DATA_DIR
from src.structure import sifts as sifts_utils
from src.tools.sifts import SIFTS


def main(args):
    # if full load requires too much memory we could use some sed command
    # or something like that to build a temp file for specific parquet index
    # or use lmdb or similar...
    df = pd.read_csv(os.path.join(PROFAM_DATA_DIR, args.data_folder, "accession_index.csv"))
    sifts = SIFTS(sifts_table_file=os.path.join(PROFAM_DATA_DIR, "sifts", "evcouplings_sifts.csv"))
    accessions_with_pdbs = set(sifts.table["uniprot_ac"].unique())
    # grep ',specific_parquet_file$' your_file.csv > temp.csv

    if args.parquet_index is None:
        parquet_files = df["parquet_file"].unique()
    else:
        parquet_files = [f"{args.parquet_index}.parquet"]

    print("Number of parquet files:", len(parquet_files))
    for parquet_file in parquet_files:
        # TODO: check this preserves float 16
        parquet_accession_df = df[df["parquet_file"] == parquet_file]
        parquet_accessions_with_pdbs = set(parquet_accession_df["accession"].unique()).intersection(accessions_with_pdbs)
        parquet_df = pd.read_parquet(os.path.join(PROFAM_DATA_DIR, args.data_folder, parquet_file))

        def annotate_row_with_pdbs(row):
            pdb_ids = []
            pdb_N = []
            pdb_CA = []
            pdb_C = []
            pdb_O = []
            has_pdb_mask = []
            for accession, sequence in zip(row["accessions"], row["sequences"]):
                if accession in parquet_accessions_with_pdbs:
                    hits = sifts.by_uniprot_id(accession, reduce_chains=True)
                    prot, pdb_id = sifts_utils.build_highest_coverage_protein_from_pdb_hits(hits, sequence)
                    pdb_ids.append(pdb_id)
                    pdb_N.append(prot.backbone_coords[:, 0, :])
                    pdb_CA.append(prot.backbone_coords[:, 1, :])
                    pdb_C.append(prot.backbone_coords[:, 2, :])
                    pdb_O.append(prot.backbone_coords[:, 3, :])
                    has_pdb_mask.append(True)

                else:
                    has_pdb_mask.append(False)

            # in this format clusters with no hits are represented with empty lists plus pdb mask
            row["pdb_ids"] = pdb_ids
            row["pdb_N"] = pdb_N
            row["pdb_CA"] = pdb_CA
            row["pdb_C"] = pdb_C
            row["pdb_O"] = pdb_O
            row["pdb_index_mask"] = has_pdb_mask   # to reconstruct a dense array we'll use values[has_pdb_mask] = pdb_N for example
            return row

        # TODO: if we want to use float16 column, how should we represent missing values? what's data efficient way?
        # if no pdb matches at all in the cluster, what's an efficient way to represent that?
        parquet_df = parquet_df.apply(annotate_row_with_pdbs, axis=1)
        parquet_df.to_parquet(os.path.join(PROFAM_DATA_DIR, args.data_folder, parquet_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str)
    parser.add_argument("--parquet_index", type=str)
    args = parser.parse_args()
    main(args)
