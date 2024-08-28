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
import numpy as np
import time
import pickle
import os
import os


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
                print("Processed", line_counter, "lines for zip file dictionary")
    return af2zip


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


def make_job_list(
    parquet_id,
    save_dir,
    minimum_foldseek_cluster_size=1,
    skip_af50=False,
):
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

    # shuffle first so that we de-correlate cluster identities in parquet files
    cluster_dict = {k: v for k, v in cluster_dict.items() if len(v) >= minimum_foldseek_cluster_size}
    cluster_ids = list(cluster_dict.keys())
    print(f"Number of clusters after filtering by cluster size >= {minimum_foldseek_cluster_size}:", len(cluster_dict.keys()))
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cluster_ids)

    parquet_size = 250 if skip_af50 else 100  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + parquet_size] for i in range(0, len(cluster_ids), parquet_size)]
    cluster_ids = clusters_to_save[parquet_id]

    cluster_counter = 0

    all_accessions = [cluster_id for cluster_members in cluster_dict.values() for cluster_id in cluster_members]
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

                if not skip_af50 and member in af50_dict:
                    for af50_member in af50_dict[member]:
                        try:
                            assert not af50_member == member
                            zip_filename = af2zip[af50_member]
                            afdb_id = f"AF-{af50_member}-F1-model_v4"
                            pdb_lookup[zip_filename].append(afdb_id)
                            cluster_membership[cluster_id].append(afdb_id)
                            metadata_lookup[afdb_id] = {
                                "cluster_id": cluster_id,
                                "accession": member,
                                "is_foldseek_representative": False,
                                "is_af50_representative": False,
                            }
                        except:
                            print("Error looking up", af50_member)

    t2 = time.time()
    print("Built lookup in", t2 - t1, "seconds", flush=True)
    print("Number of zip files: ", len(pdb_lookup), flush=True)
    return pdb_lookup, metadata_lookup, cluster_membership


def create_foldseek_parquets(
    save_dir,
    minimum_foldseek_cluster_size=1,
    skip_af50=False,
    parquet_ids=None,
):
    if parquet_ids is None:
        # 2302908 clusterss
        parquet_ids = range(231)
    for parquet_id in parquet_ids:
        pdb_lookup, _, _ = make_job_list(
            parquet_id,
            save_dir=save_dir,
            minimum_foldseek_cluster_size=minimum_foldseek_cluster_size,
            skip_af50=skip_af50,
        )
        for zip_filename in pdb_lookup.keys():
            shutil.copyfile(os.path.join("/SAN/bioinf/afdb_domain/zipfiles", zip_filename + ".zip"), "/SAN/orengolab/cath_plm/ProFam/data/afdb_domain/zipfiles/" + zip_filename + ".zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("--minimum_foldseek_cluster_size", type=int, default=1)
    parser.add_argument("--parquet_ids", type=int, default=None, nargs="+")
    parser.add_argument("--skip_af50", action="store_true")
    args = parser.parse_args()

    if args.skip_af50:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct_example/"
    else:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct/"

    create_foldseek_parquets(
        save_dir=save_dir,
        minimum_foldseek_cluster_size=args.minimum_foldseek_cluster_size,
        parquet_ids=args.parquet_ids,
        skip_af50=args.skip_af50,
    )
