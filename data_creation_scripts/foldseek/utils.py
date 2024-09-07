import os
import multiprocessing
import re
import zipfile
import time
import modin.pandas as pd
from dask import dataframe as dd
from collections import defaultdict


def make_af50_dictionary(af50_path, clusters_to_include=None):
    line_counter = 0
    af50_dict = {}
    with open(af50_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        try:
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
        except Exception as e:
            print("Error processing line", line, flush=True)
            raise e
    return af50_dict


def dask_make_af50_dictionary(af50_path, clusters_to_include):
    ddf = dd.read_csv(af50_path, sep="\t", header=None, names=["rep_id", "entry_id", "clu_flag", "tax_id"])
    ddf = ddf[(ddf["clu_flag"] == 1)&(ddf["rep_id"].isin(clusters_to_include))].compute()  # TODO: support num workers...
    af50_dict = defaultdict(list)
    for _, row in ddf.iterrows():
        af50_dict[row["rep_id"]].append(row["entry_id"])
    return af50_dict


def modin_make_af50_dictionary(af50_path, clusters_to_include):
    df = pd.read_csv(af50_path, sep="\t", header=None, names=["rep_id", "entry_id", "clu_flag", "tax_id"])
    df = df[(df["clu_flag"] == 1)&(df["rep_id"].isin(clusters_to_include))]
    af50_dict = defaultdict(list)
    for _, row in df.iterrows():
        af50_dict[row["rep_id"]].append(row["entry_id"])
    return af50_dict


def make_zip_dictionary(zip_index):
    line_counter = 0
    af2zip = {}
    with open(zip_index, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split("\t")
        afdb_id = line[0]
        uniprot_id = afdb_id.split("-")[1]
        # todo just make this an if statement
        assert afdb_id == f"AF-{uniprot_id}-F1-model_v4", f"AFDB ID mismatch: {afdb_id} {uniprot_id}"
        zip_file = line[2]
        af2zip[uniprot_id] = zip_file

        line_counter += 1
        if line_counter % 100000 == 0:
            print("Processed", line_counter, "lines for zip file dictionary", flush=True)
    return af2zip


# def make_zip_dictionary(zip_index, accessions_to_include=None):
#     line_counter = 0
#     af2zip = {}
#     # TODO: finish early if we get all the accessions.
#     with open(zip_index, "r", buffering=1024*1024*1024) as f:  # 1 GB buffer
#         for line in f:
#             line = line.strip().split("\t")
#             afdb_id = line[0]
#             uniprot_id = afdb_id.split("-")[1]
#             # todo just make this an if statement
#             assert afdb_id == f"AF-{uniprot_id}-F1-model_v4", f"AFDB ID mismatch: {afdb_id} {uniprot_id}"
#             zip_file = line[2]
#             if accessions_to_include is None or uniprot_id in accessions_to_include:
#                 af2zip[uniprot_id] = zip_file

#             line_counter += 1
#             if line_counter % 100000 == 0:
#                 print("Processed", line_counter, "lines for zip file dictionary", flush=True)
#     return af2zip


def dask_make_zip_dictionary(zip_index, accessions_to_include):
    ddf = dd.read_csv(zip_index, sep="\t", header=None, names=["afdb_id", "uniprot_id", "zip_file"])
    ddf = ddf[ddf["uniprot_id"].isin(accessions_to_include)].compute()
    return ddf.set_index("uniprot_id")["zip_file"].to_dict()


def modin_make_zip_dictionary(zip_index, accessions_to_include):
    df = pd.read_csv(zip_index, sep="\t", header=None, names=["afdb_id", "uniprot_id", "zip_file"])
    df = df[df["uniprot_id"].isin(accessions_to_include)]
    return df.set_index("uniprot_id")["zip_file"].to_dict()


def make_cluster_dictionary(cluster_path):
    line_counter = 0
    cluster_dict = {}
    with open(cluster_path, "r") as f:
        lines = f.readlines()
    for line in lines:
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


def make_sequence_dictionary(fasta_path):
    """Should be <80 GB memory required to load all sequences;
    considerably lower for just cluster representatives.

    To handle larger files, we could partition the cluster dict
    (or e.g. first process representatives then process the rest.)
    """
    sequence_dict = {}
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                # TODO: check this
                uniprot_acc = re.search("UA=(\w+)", line).group(1)
                sequence = next(f).strip()
                sequence_dict[uniprot_acc] = sequence
    print("Number of sequences in dictionary:", len(sequence_dict))
    return sequence_dict


def extract_pdbs_from_zips(pdb_lookup, output_dir, num_processes):
    print("Extracting pdbs with", num_processes, "processes", flush=True)
    seq_fail_counter = 0
    seq_success_counter = 0
    # Parallel extraction of pdb files
    t0 = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if num_processes is None:
        for zip_index, (zip_filename, afdb_ids) in enumerate(pdb_lookup.items()):
            print("Zip filename", zip_filename, "ids", afdb_ids, flush=True)
            success_count, fail_count = extract_pdbs(zip_filename, afdb_ids, output_dir, zip_index)
            seq_success_counter += success_count
            seq_fail_counter += fail_count
    else:
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
                    assert not os.path.isfile(
                        os.path.join(output_folder, afdb_id + ".pdb")
                    ), f"{afdb_id} already exists in {output_folder} {afdb_id}, {zip_filename}, {afdb_ids}"
                    zip_ref.extract(afdb_id + ".pdb", output_folder)
                    print(
                        f"Extracted {afdb_id} from {zip_filename} to {output_folder}",
                        os.path.isfile(os.path.join(output_folder, afdb_id + ".pdb"))
                    )
                    successes.append(True)
                else:
                    print(f"{afdb_id} not found in {zip_filename}", flush=True)
                    successes.append(False)
    except Exception as e:
        print(f"Error extracting {zip_filename} {afdb_ids} {e}", flush=True)
        successes = [False] * len(afdb_ids)
    return successes


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