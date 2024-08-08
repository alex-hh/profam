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
import multiprocessing
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
from src.data.structure import get_atom_coords_residuewise, load_structure


def make_af50_dictionary(shared_dict, af50_path):
    line_counter = 0
    with open(af50_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rep_id = line[0]
            entry_id = line[1]
            # 1: clustered in AFDB50, 2: clustered in AFDB clusters, 3/4: removed (fragments/singletons)
            clu_flag = int(line[2])  # 1
            # n.b. the 2s are duplicates of the other cluster dict
            # n.b. we don't include the representative in its own cluster atm
            if clu_flag == 1:
                if rep_id not in shared_dict:
                    shared_dict[rep_id] = []
                shared_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return af50_dict


def make_zip_dictionary(shared_dict):
    line_counter = 0
    with open("/SAN/bioinf/afdb_domain/zipmaker/zip_index", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            afdb_id = line[0]
            uniprot_id = afdb_id.split("-")[1]
            zip_file = line[2]
            shared_dict[uniprot_id] = (zip_file, afdb_id)

            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return af2zip


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


def make_cluster_dictionary(shared_dict, cluster_path):
    line_counter = 0
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            entry_id = line[0]
            rep_id = line[1]
            if rep_id not in shared_dict:
                shared_dict[rep_id] = []
            shared_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return shared_dict


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


def build_single_parquet(
    parquet_id,
    cluster_ids,
    af2zip,
    af50_dict,
    save_dir,
    minimum_cluster_size=1,
    skip_af50=False,
):
    seq_fail_counter = 0
    seq_success_counter = 0
    cluster_counter = 0
    seq_fail_path = save_dir + f"failed_sequences_{parquet_id}.txt"

    pdb_lookup = defaultdict(list)
    metadata_lookup = dict()

    with open(seq_fail_path, "w") as fail_file:
        for cluster_id in cluster_ids:
            members = cluster_dict[cluster_id]
            if len(members) >= minimum_cluster_size:
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
                save_pdbs_to_parquet(save_dir, cluster_ids, cluster_counter, metadata_lookup)


def create_foldseek_parquets(
    cluster_dict,
    af2zip,
    af50_dict,
    save_dir,
    minimum_cluster_size=1,
    verbose=False,
    skip_af50=False,
    parquet_id=None,
):
    # shuffle first so that we de-correlate cluster identities in parquet files
    clusters = list(cluster_dict.keys())
    rng = np.random.default_rng(seed=42)
    rng.shuffle(clusters)

    parquet_size = 10000  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [clusters[i:i + parquet_size] for i in range(0, len(clusters), parquet_size)]
    if parquet_id is None:
        # TODO: enable multiprocessing here
        for parquet_ix, cluster_ids in enumerate(clusters_to_save):
            build_single_parquet(
                parquet_ix,
                cluster_ids=cluster_ids,
                af2zip=af2zip,
                af50_dict=af50_dict,
                save_dir=save_dir,
                minimum_cluster_size=minimum_cluster_size,
                skip_af50=skip_af50,
            )
    else:
        build_single_parquet(
            parquet_id,
            cluster_ids=clusters_to_save[parquet_id],
            af2zip=af2zip,
            af50_dict=af50_dict,
            save_dir=save_dir,
            minimum_cluster_size=minimum_cluster_size,
            skip_af50=skip_af50,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("--minimum_cluster_size", type=int, default=1)
    parser.add_argument("--cluster_start", type=int, default=0)
    parser.add_argument("--cluster_end", type=int, default=None)
    parser.add_argument("--parquet_id", type=int, default=None)
    args = parser.parse_args()
    save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/"
    cluster_dict_pickle_path = os.path.join(save_dir, "foldseek_cluster_dict.pkl")
    af50_dict_pickle_path = os.path.join(save_dir, "af50_cluster_dict.pkl")

    manager = multiprocessing.Manager()
    cluster_dict = manager.dict()
    af2zip = manager.dict()
    af50_dict = manager.dict()

    make_zip_dictionary(af2zip)

    if not os.path.exists(cluster_dict_pickle_path):
        print("Creating foldseek dataset", flush=True)
        make_cluster_dictionary(cluster_dict, args.cluster_path)
        print("Number of clusters:", len(cluster_dict))
        print("Saving cluster dictionary")
        with open(save_dir + "foldseek_cluster_dict.pkl", "wb") as f:
            pickle.dump(cluster_dict, f)
    else:
        print("Loading cluster dictionary", flush=True)
        with open(cluster_dict_pickle_path, "rb") as f:
            cluster_dict = pickle.load(f)
        print("Number of clusters:", len(cluster_dict))

    if not os.path.exists(af50_dict_pickle_path):
        af50_dict = make_af50_dictionary(af50_dict, af50_path)
        with open(af50_dict_pickle_path, "wb") as f:
            pickle.dump(af50_dict, f)
    else:
        with open(af50_dict_pickle_path, "rb") as f:
            af50_dict = pickle.load(f)

    create_foldseek_parquets(
        cluster_dict=cluster_dict,
        af2zip=af2zip,
        af50_dict=af50_dict,
        save_dir=save_dir,
        minimum_cluster_size=args.minimum_cluster_size,
        parquet_id=args.parquet_id,
    )
