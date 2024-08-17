"""
Loads PDB files for the foldseek clusters and
converts them into parquet files. The parquet files contain sequences
as well as backbone coordinates (N, Ca, C, O).

N.B. this is slow and will require optimisation/parallelisation:
10000 clusters takes around a day.

There are 1000000 proteome files (i.e. a factor of 200 reduction in
number of files if we use these rather than pdb files.)

For parallelisation, there are two options: parallelising within
building of single parquets, and parallelising across building of
distinct parquets.
"""
import argparse
import shutil
from collections import defaultdict
import multiprocessing
from biotite.sequence import ProteinSequence
from biotite.structure.residues import get_residues, get_residue_starts
import numpy as np
import time
import pickle
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import os
from src.data.pdb import get_atom_coords_residuewise, load_structure


af50_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv"


def make_af50_dictionary(clusters_to_include=None):
    line_counter = 0
    af50_dict = {}
    with open(af50_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rep_id = line[0]
            entry_id = line[1]
            # 1: clustered in AFDB50, 2: clustered in AFDB clusters, 3/4: removed (fragments/singletons)
            clu_flag = int(line[2])  # 1
            # n.b. the 2s are duplicates of the other cluster dict
            # n.b. we don't include the representative in its own cluster atm
            if clu_flag == 1 and (clusters_to_include is None or rep_id in clusters_to_include):
                if rep_id not in af50_dict:
                    af50_dict[rep_id] = []
                af50_dict[rep_id].append(entry_id)
            line_counter += 1
    return af50_dict


def make_zip_dictionary(accessions_to_include=None):
    line_counter = 0
    af2zip = {}
    with open("/SAN/bioinf/afdb_domain/zipmaker/zip_index", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            afdb_id = line[0]
            uniprot_id = afdb_id.split("-")[1]
            assert afdb_id == f"AF-{uniprot_id}-F1-model_v4"
            zip_file = line[2]
            if accessions_to_include is None or uniprot_id in accessions_to_include:
                af2zip[uniprot_id] = zip_file

            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for zip file dictionary", flush=True)
    return af2zip


def extract_multi_pdb_files(afdb_ids, zip_filename, output_folder):
    # Extract the specified PDB files
    zip_filepath = os.path.join("/SAN/bioinf/afdb_domain/zipfiles", zip_filename+".zip")
    successes = []
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            names = zip_ref.namelist()
            for afdb_id in afdb_ids:
                if afdb_id + ".pdb" in names:
                    # TODO: print worker...
                    assert not os.path.isfile(os.path.join(output_folder, afdb_id + ".pdb")), f"{afdb_id} already exists in {output_folder} {afdb_id}, {zip_filename}, {afdb_ids}"
                    zip_ref.extract(afdb_id + ".pdb", output_folder)
                    print(f"Extracted {afdb_id} from {zip_filename} to {output_folder}", os.path.isfile(os.path.join(output_folder, afdb_id + ".pdb")))
                    successes.append(True)
                else:
                    print(f"{afdb_id} not found in {zip_filename}", flush=True)
                    successes.append(False)
    except Exception as e:
        print(f"Error extracting {zip_filename} {afdb_ids} {e}", flush=True)
        successes = [False] * len(afdb_ids)
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
                print("Processed", line_counter, "lines for cluster dictionary", flush=True)
    return cluster_dict


def save_pdbs_to_parquet(save_dir, scratch_dir, clusters_to_save, parquet_id, metadata_lookup, verbose=False):
    # Save the pdbs to parquet
    results = []
    for cluster_id, cluster_members in clusters_to_save.items():
        sequences = []
        accessions = []
        is_foldseek_representative = []
        is_af50_representative = []
        all_coords = {"N": [], "CA": [], "C": [], "O": []}
        all_b_factors = []

        for afdb_id in cluster_members:
            pdb = os.path.join(scratch_dir, str(parquet_id), afdb_id + ".pdb")
            metadata = metadata_lookup[afdb_id]
            accessions.append(metadata["accession"])
            is_foldseek_representative.append(metadata["is_foldseek_representative"])
            is_af50_representative.append(metadata["is_af50_representative"])
            structure = load_structure(pdb, chain="A", extra_fields=["b_factor"])
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)  # residues, atoms, xyz
            residue_identities = get_residues(structure)[1]
            b_factors = structure.b_factor[get_residue_starts(structure)]
            seq = "".join(
                [ProteinSequence.convert_letter_3to1(r) for r in residue_identities]
            )
            all_b_factors.append(b_factors)
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
                "plddts": all_b_factors,
                "accessions": accessions,
                "is_foldseek_representative": is_foldseek_representative,
                "is_af50_representative": is_af50_representative,
            }
        )

    print("Deleting directory", os.path.join(scratch_dir, str(parquet_id)), flush=True)
    shutil.rmtree(os.path.join(scratch_dir, str(parquet_id)))

    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    output_file = os.path.join(f'{save_dir}', f'{parquet_id}.parquet')
    pq.write_table(table, output_file)
    print(f"Saved {clusters_to_save} clusters to {output_file}")
    return output_file


def extract_pdbs(zip_filename, afdb_ids, save_dir, zip_index):
    # TODO: for improved efficiency, extract the relevant parts from the pdb file at this point.
    print("Extracting pdbs", zip_filename, afdb_ids, "cluster index", zip_index, flush=True)
    t0 = time.time()
    successes = extract_multi_pdb_files(
         afdb_ids, zip_filename, save_dir,
    )
    t1 = time.time()
    print("Extracted", len(afdb_ids), "pdbs in", t1 - t0, "seconds", zip_filename, flush=True)
    return sum(successes), len(successes) - sum(successes)


def make_job_list(
    parquet_id,
    save_dir,
    minimum_foldseek_cluster_size=1,
    skip_af50=False,
):
    # TODO: instead of loading the cluster dictionary we can just save a file which lists the cluster sizes.
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
        print("Number of clusters:", len(cluster_dict), flush=True)

    # shuffle first so that we de-correlate cluster identities in parquet files
    cluster_dict = {k: v for k, v in cluster_dict.items() if len(v) >= minimum_foldseek_cluster_size}
    cluster_ids = list(cluster_dict.keys())
    print(f"Number of clusters after filtering by cluster size >= {minimum_foldseek_cluster_size}:", len(cluster_dict.keys()), flush=True)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cluster_ids)

    parquet_size = 250 if skip_af50 else 100  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + parquet_size] for i in range(0, len(cluster_ids), parquet_size)]
    cluster_ids = clusters_to_save[parquet_id]

    cluster_counter = 0

    all_accessions = [member_id for cluster_id in cluster_ids for member_id in cluster_dict[cluster_id]]
    if not skip_af50:
        af50_dict = make_af50_dictionary(clusters_to_include=all_accessions)
        all_accessions = all_accessions + [cluster_id for cluster_members in af50_dict.values() for cluster_id in cluster_members]

    af2zip = make_zip_dictionary(all_accessions)

    pdb_lookup = defaultdict(list)
    cluster_membership = defaultdict(list)  # TODO: make this a single dictionary by combining the records from the two files.
    metadata_lookup = dict()

    t1 = time.time()
    for ix, cluster_id in enumerate(cluster_ids):
        if ix % 500 == 0:
            print(f"Processing cluster {ix} of {len(cluster_ids)}", flush=True)
        members = cluster_dict[cluster_id]
        if len(members) >= minimum_foldseek_cluster_size:
            cluster_counter += 1

            for member in members:
                zip_filename = af2zip[member]
                afdb_id = f"AF-{member}-F1-model_v4"
                pdb_lookup[zip_filename].append(afdb_id)
                metadata_lookup[afdb_id] = {
                    "cluster_id": cluster_id,
                    "accession": member,
                    "is_foldseek_representative": member == cluster_id,
                    "is_af50_representative": True,
                }
                cluster_membership[cluster_id].append(afdb_id)

                if not skip_af50:  # this is the af50 representative
                    af50_members = af50_dict.get(member, [])
                    for af50_member in af50_members:
                        if af50_member.uniprot_id != member:
                            zip_filename = af2zip[af50_member]
                            afdb_id = f"AF-{af50_member}-F1-model_v4"
                            pdb_lookup[zip_filename].append(afdb_id)
                            cluster_membership[cluster_id].append(afdb_id)
                            metadata_lookup[afdb_id] = {
                                "cluster_id": cluster_id,
                                "accession": af50_member,
                                "is_foldseek_representative": False,
                                "is_af50_representative": False,
                            }

    t2 = time.time()
    print("Built lookup in", t2 - t1, "seconds", flush=True)
    print("Number of zip files: ", len(pdb_lookup), flush=True)
    return pdb_lookup, metadata_lookup, cluster_membership


def extract_pdbs_for_parquet(pdb_lookup, scratch_dir, parquet_id, num_processes):
    seq_fail_counter = 0
    seq_success_counter = 0
    # Parallel extraction of pdb files
    t0 = time.time()
    output_dir = os.path.join(scratch_dir, str(parquet_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for zip_index, (zip_filename, afdb_ids) in enumerate(pdb_lookup.items()):
            print("Zip filename", zip_filename, "ids", afdb_ids, flush=True)
            result = pool.apply_async(
                extract_pdbs,
                args=(zip_filename, afdb_ids, output_dir, zip_index)
            )
            results.append(result)

        for result in results:
            success_count, fail_count = result.get()
            seq_success_counter += success_count
            seq_fail_counter += fail_count

    print("Number of failed sequences:", seq_fail_counter)
    print("Number of successful sequences:", seq_success_counter, flush=True)
    t1 = time.time()
    print("Extracted all pdbs in", t1 - t0, "seconds", flush=True)


def create_foldseek_parquets(
    save_dir,
    scratch_dir,
    minimum_foldseek_cluster_size=1,
    skip_af50=False,
    parquet_ids=None,
    num_processes=None,
):
    if parquet_ids is None:
        # 2302908 clusterss
        parquet_ids = range(231)
    for parquet_id in parquet_ids:
        pdb_lookup, metadata_lookup, cluster_membership = make_job_list(
            parquet_id,
            save_dir=save_dir,
            minimum_foldseek_cluster_size=minimum_foldseek_cluster_size,
            skip_af50=skip_af50,
        )
        extract_pdbs_for_parquet(
            pdb_lookup=pdb_lookup,
            scratch_dir=scratch_dir,
            parquet_id=parquet_id,
            num_processes=num_processes,
        )
        save_pdbs_to_parquet(
            save_dir=save_dir,
            scratch_dir=scratch_dir,
            clusters_to_save=cluster_membership,
            parquet_id=parquet_id,
            metadata_lookup=metadata_lookup,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("scratch_dir")
    parser.add_argument("--minimum_foldseek_cluster_size", type=int, default=1)
    parser.add_argument("--parquet_ids", type=int, default=None, nargs="+")
    parser.add_argument("--skip_af50", action="store_true")
    parser.add_argument("--num_processes", type=int, default=None)
    args = parser.parse_args()

    if args.num_processes is None:
        args.num_processes = os.cpu_count()
    print("Num cpus", os.cpu_count(), flush=True)

    if args.skip_af50:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/"
    else:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct/"

    create_foldseek_parquets(
        save_dir=save_dir,
        scratch_dir=args.scratch_dir,
        minimum_foldseek_cluster_size=args.minimum_foldseek_cluster_size,
        parquet_ids=args.parquet_ids,
        skip_af50=args.skip_af50,
        num_processes=args.num_processes,
    )
