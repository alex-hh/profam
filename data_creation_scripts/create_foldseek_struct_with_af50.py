"""
Loads PDB files for the foldseek clusters and
converts them into parquet files. The parquet files contain sequences
as well as backbone coordinates (N, Ca, C, O).

N.B. this is slow and will require optimisation/parallelisation:
10000 clusters takes around a day.

There are 1000000 proteome files (i.e. a factor of 200 reduction in
number of files if we use these rather than pdb files.)
"""
import argparse
import shutil
from collections import defaultdict
from biotite.sequence import ProteinSequence
from biotite.structure.residues import get_residues
import numpy as np
import time
import pickle
import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import os
from src.data.pdb import get_atom_coords_residuewise, load_structure


def make_zip_dictionary():
    line_counter = 0
    af2zip = {}
    with open("/SAN/bioinf/afdb_domain/zipmaker/zip_index", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            afdb_id = line[0]
            uniprot_id = afdb_id.split("-")[1]
            zip_file = line[2]
            af2zip[uniprot_id] = (zip_file, afdb_id)

            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return af2zip

# Read the mapping file to get the pdb to zip file mapping
af2zip = make_zip_dictionary()


def extract_pdb_file(uniprot_id, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract the specified PDB files
    try:
        zip_filename, afdb_id = af2zip[uniprot_id]
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


def extract_multi_pdb_files(cluster_ids, afdb_ids, zip_filename, output_folder):
    assert len(cluster_ids) == len(afdb_ids)
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract the specified PDB files
    zip_filepath = os.path.join("/SAN/bioinf/afdb_domain/zipfiles", zip_filename+".zip")
    successes = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        names = zip_ref.namelist()
        for cluster_id, afdb_id in zip(cluster_ids, afdb_ids):
            cluster_output_folder = os.path.join(output_folder, cluster_id, "pdbs")
            if afdb_id + ".pdb" in names:
                zip_ref.extract(afdb_id + ".pdb", cluster_output_folder)
                print(f"Extracted {afdb_id} from {zip_filename} to {cluster_output_folder}")
                successes.append(True)
            else:
                print(f"{afdb_id} not found in {zip_filename}")
                successes.append(False)
    return successes


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


def save_pdbs_to_parquet(save_dir, clusters_to_save, cluster_counter, metadata_lookup, verbose=False):
    # Save the pdbs to parquet
    results = []
    for cluster_id in clusters_to_save:
        pdbs = glob.glob(os.path.join(save_dir, cluster_id, "pdbs/*.pdb"))
        sequences = []
        accessions = []
        is_foldseek_representative = []
        is_af50_representative = []
        all_coords = {"N": [], "CA": [], "C": [], "O": []}

        for pdb in pdbs:
            afdb_id = os.path.splitext(os.path.basename(pdb))[0]
            metadata = metadata_lookup[afdb_id]
            accessions.append(metadata["accession"])
            is_foldseek_representative.append(metadata["is_foldseek_representative"])
            is_af50_representative.append(metadata["is_af50_representative"])
            structure = load_structure(pdb, chain="A")
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)  # residues, atoms, xyz
            residue_identities = get_residues(structure)[1]
            seq = "".join(
                [ProteinSequence.convert_letter_3to1(r) for r in residue_identities]
            )
            sequences.append(seq)
            for ix, atom_name in enumerate(["N", "CA", "C", "O"]):
                all_coords[atom_name].append(coords[:, ix, :].flatten())
            os.remove(pdb)

        # TODO: save representative?
        results.append(
            {
                "sequences": sequences,
                "cluster_id": cluster_id,
                "N": all_coords["N"],
                "CA": all_coords["CA"],
                "C": all_coords["C"],
                "O": all_coords["O"],
                "accessions": accessions,
                "is_foldseek_representative": is_foldseek_representative,
                "is_af50_representative": is_af50_representative,
            }
        )
        print("Deleting directory", os.path.join(save_dir, cluster_id), flush=True)
        shutil.rmtree(os.path.join(save_dir, cluster_id))

    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    output_file = os.path.join(f'{save_dir}', f'{cluster_counter}.parquet')
    pq.write_table(table, output_file)
    print(f"Saved {clusters_to_save} clusters to {output_file}")
    return output_file


def create_foldseek_parquets(cluster_dict, af50_dict, save_dir, minimum_cluster_size=1, verbose=False, skip_af50=False):
    seq_fail_counter = 0
    seq_success_counter = 0
    cluster_counter = 0
    seq_fail_path = save_dir + "failed_sequences.txt"

    # shuffle first so that we de-correlate cluster identities in parquet files
    clusters = list(cluster_dict.keys())
    rng = np.random.default_rng(seed=42)
    rng.shuffle(clusters)
    metadata_lookup = dict()
    pdb_lookup = defaultdict(list)

    with open(seq_fail_path, "w") as fail_file:
        for cluster_id in clusters:
            members = cluster_dict[cluster_id]
            if len(members) >= minimum_cluster_size:
                clusters_to_save.append(cluster_id)
                cluster_counter += 1

                for member in members:
                    zip_filename, afdb_id = af2zip[member]
                    pdb_lookup[zip_filename].append((cluster_id, afdb_id))
                    metadata_lookup[afdb_id] = {
                        "cluster_id": cluster_id,
                        "accession": member,
                        "is_foldseek_representative": member == cluster_id,
                        "is_af50_representative": True,
                    }

                    if not skip_af50 and member in af50_dict:
                        for af50_member in af50_dict[member]:
                            assert not af50_member == member
                            zip_filename, afdb_id = af2zip[af50_member]
                            pdb_lookup[zip_filename].append((cluster_id, afdb_id))
                            metadata_lookup[afdb_id] = {
                                "cluster_id": cluster_id,
                                "accession": member,
                                "is_foldseek_representative": False,
                                "is_af50_representative": False,
                            }

                if cluster_counter % 10000 == 0:
                    for zip_filename, _ids in pdb_lookup.items():
                        cluster_ids = [x[0] for x in _ids]
                        afdb_ids = [x[1] for x in _ids]
                        t0 = time.time()
                        successes = extract_multi_pdb_files(
                            cluster_ids, afdb_ids, zip_filename, save_dir,
                        )
                        t1 = time.time()
                        print("Extracted", len(afdb_ids), "pdbs in", t1 - t0, "seconds", zip_filename, flush=True)
                        seq_success_counter += sum(successes)
                        seq_fail_counter += len(successes) - sum(successes)
                    print("\nProcessed", cluster_counter, "clusters")
                    print("Number of failed sequences:", seq_fail_counter)
                    print("Number of successful sequences:", seq_success_counter, flush=True)
                    save_pdbs_to_parquet(save_dir, clusters_to_save, cluster_counter, metadata_lookup)
                    clusters_to_save = []
                    pdb_lookup = defaultdict(list)
                    metadata_lookup = dict()

    if len(clusters_to_save) > 0:
        print("\nProcessed", cluster_counter, "clusters")
        print("Number of failed sequences:", seq_fail_counter)
        print("Number of successful sequences:", seq_success_counter, flush=True)
        save_pdbs_to_parquet(save_dir, clusters_to_save, cluster_counter, metadata_lookup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("--minimum_cluster_size", type=int, default=1)
    parser.add_argument("--cluster_start", type=int, default=0)
    parser.add_argument("--cluster_end", type=int, default=None)
    args = parser.parse_args()
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

    create_foldseek_parquets(cluster_dict, save_dir, minimum_cluster_size=args.minimum_cluster_size)
