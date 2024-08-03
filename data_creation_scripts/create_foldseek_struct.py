"""
Loads PDB files for the foldseek clusters and
converts them into parquet files. The parquet files contain sequences
as well as backbone coordinates (N, Ca, C, O).
"""
import argparse
import dask
import dask.dataframe as dd
from typing import List
import biotite
from biotite.sequence import ProteinSequence
from biotite.structure import filter_amino_acids, get_chains, apply_residue_wise
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues
import numpy as np
import random
import sys
import sqlite3
import pickle
import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


import zipfile
import os


def read_mapping():
    # One thing we need to be careful about is making sure we can map correctly
    df = dd.read_csv(
        "/SAN/bioinf/afdb_domain/zipmaker/zip_index",
        sep="\t",
        names=["afdb_id", "seq_hash", "zip_file"]
    )
    # df["num_parts"] = df["afdb_id"].apply(lambda x: len(x.split("-")))
    # min_val, max_val, mean_val = dask.compute(df["num_parts"].min(), df["num_parts"].max(), df["num_parts"].mean())
    df["uniprot_id"] = df["afdb_id"].apply(lambda x: x.split("-")[1], meta=("x", "str"))
    df.set_index("uniprot_id")
    return df

# Read the mapping file to get the pdb to zip file mapping
pdb_to_zip = read_mapping()


def extract_pdb_file(uniprot_id, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract the specified PDB files
    try:
        afdb_id = pdb_to_zip.loc[uniprot_id, "afdb_id"].compute()
        zip_filename = pdb_to_zip.loc[uniprot_id, "zip_file"].compute()
        zip_filepath = os.path.join("/SAN/bioinf/afdb_domain/zipfiles", zip_filename+".zip")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            if afdb_id + ".pdb" in zip_ref.namelist():
                zip_ref.extract(afdb_id + ".pdb", output_folder)
                print(f"Extracted {afdb_id} from {zip_filename} to {output_folder}")
                return True
            else:
                print(f"{afdb_id} not found in {zip_filename}")
                return False
    except Exception as e:
        print(e)
        print(f"No zip file containing {uniprot_id} found in mapping")
        return False


def make_cluster_dictionary(cluster_path):
    line_counter = 0
    cluster_dict = {}
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            entry_id = line[0]
            rep_id = line[1]
            if rep_id not in cluster_dict:
                cluster_dict[rep_id] = []
            cluster_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return cluster_dict


backbone_atoms = ["N", "CA", "C", "O"]


def _filter_atom_names(array, atom_names):
    return np.isin(array.atom_name, atom_names)


def custom_filter_backbone(array):
    """
    Filter all peptide backbone atoms of one array.

    N, CA, C and O

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where an atom
        is a part of the peptide backbone.
    """

    return _filter_atom_names(array, backbone_atoms) & filter_amino_acids(array)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Source: esm inverse folding
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return apply_residue_wise(struct, struct, filterfn)


def load_structure(fpath, chain=None):
    """
    Modified from esm inverse folding utils to not remove O
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith("cif"):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith("pdb"):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = custom_filter_backbone(structure)
    # bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def save_pdbs_to_parquet(save_dir, clusters_to_save, cluster_counter, verbose=False):
    # Save the pdbs to parquet
    results = []
    for cluster_id in clusters_to_save:
        pdbs = glob.glob(os.path.join(save_dir, cluster_id, "pdbs/*.pdb"))
        sequences = []
        all_coords = {"N": [], "CA": [], "C": [], "O": []}

        for pdb in pdbs:
            structure = load_structure(pdb, chain="A")
            # TODO - fix O loading
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)  # residues, atoms, xyz
            residue_identities = get_residues(structure)[1]
            seq = "".join(
                [ProteinSequence.convert_letter_3to1(r) for r in residue_identities]
            )
            sequences.append(seq)
            for ix, atom_name in enumerate(["N", "CA", "C", "O"]):
                all_coords[atom_name].append(coords[:, ix, :].flatten())
            # TODO: save representative?
            results.append(
                {
                    "sequences": sequences,
                    "cluster_id": cluster_id,
                    "N": all_coords["N"],
                    "CA": all_coords["CA"],
                    "C": all_coords["C"],
                    "O": all_coords["O"],
                }
            )
            os.remove(pdb)
        df = pd.DataFrame(results)
        table = pa.Table.from_pandas(df)
        output_file = f'{save_dir}/{cluster_counter}.parquet'
        pq.write_table(table, output_file)
        print(f"Saved {clusters_to_save} clusters to {output_file}")
        return output_file


def create_foldseek_parquets(cluster_dict, save_dir, minimum_cluster_size=1, verbose=False):
    seq_fail_counter = 0
    seq_success_counter = 0
    cluster_counter = 0
    seq_fail_path = save_dir + "failed_sequences.txt"

    # shuffle first so that we de-correlate cluster identities in parquet files
    clusters = list(cluster_dict.keys())
    random.shuffle(clusters)
    clusters_to_save = []

    with open(seq_fail_path, "w") as fail_file:
        for cluster_id in clusters:
            members = cluster_dict[cluster_id]
            if len(members) >= minimum_cluster_size:
                clusters_to_save.append(cluster_id)
                cluster_counter += 1

                for member in members:
                    extracted = extract_pdb_file(member, os.path.join(save_dir, cluster_id, "pdbs"))
                    if extracted:
                        seq_success_counter += 1
                    else:
                        seq_fail_counter += 1
                        fail_file.write(member + "\n")

                if cluster_counter % 10000 == 0:
                    print("\nProcessed", cluster_counter, "clusters")
                    print("Number of failed sequences:", seq_fail_counter)
                    print("Number of successful sequences:", seq_success_counter, flush=True)
                    save_pdbs_to_parquet(save_dir, clusters_to_save, cluster_counter)
                    clusters_to_save = []

    print("\nProcessed", cluster_counter, "clusters")
    print("Number of failed sequences:", seq_fail_counter)
    print("Number of successful sequences:", seq_success_counter, flush=True)
    save_pdbs_to_parquet(save_dir, cluster_counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("--minimum_cluster_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    scratch_dir = sys.argv[1]
    save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/"
    cluster_dict_pickle_path = os.path.join(save_dir, "foldseek_cluster_dict.pkl")

    if not os.path.exists(cluster_dict_pickle_path):
        print("Creating foldseek dataset", flush=True)
        cluster_dict = make_cluster_dictionary(args.cluster_path)
        print("Number of clusters:", len(cluster_dict))
        print("Saving cluster dictionary")
        with open(save_dir + "foldseek_cluster_dict.pkl", "wb") as f:
            pickle.dump(cluster_dict, f)
    else:
        print("Loading cluster dictionary", flush=True)
        with open(cluster_dict_pickle_path, "rb") as f:
            cluster_dict = pickle.load(f)
        print("Number of clusters:", len(cluster_dict))

    create_foldseek_parquets(cluster_dict, save_dir, minimum_cluster_size=args.minimum_cluster_size, verbose=args.verbose)
