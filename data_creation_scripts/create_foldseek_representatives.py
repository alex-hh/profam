"""
Loads PDB files for the foldseek clusters and
converts them into parquet files. The parquet files contain sequences,
foldmason MSTA alignments (both aa and 3di),
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
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import os
from src.data.pdb import get_atom_coords_residuewise, load_structure
from src.tools.foldseek import convert_pdbs_to_3di


def get_af50_representatives(af50_path):
    representatives = []
    with open(af50_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rep_id = line[0]
            # 1: clustered in AFDB50, 2: clustered in AFDB clusters, 3/4: removed (fragments/singletons)
            clu_flag = int(line[2])  # 1

            if (clu_flag == 1 or clu_flag == 2) and rep_id not in representatives:
                representatives.append(rep_id)
    return representatives


def make_zip_dictionary(zip_index, accessions_to_include=None):
    line_counter = 0
    af2zip = {}
    # TODO: finish early if we get all the accessions.
    with open(zip_index, "r") as f:
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
                    assert (
                        not os.path.isfile(os.path.join(output_folder, afdb_id + ".pdb"))
                    ), (
                        f"{afdb_id} already exists in {output_folder} {afdb_id}, {zip_filename}, {afdb_ids}"
                    )
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


def get_foldseek_representatives(cluster_path):
    line_counter = 0
    representatives = []
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rep_id = line[1]
            if rep_id not in representatives:
                representatives.append(rep_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary", flush=True)
    return representatives


def get_cluster_ids(cluster_path, use_af50_representatives=False):
    if use_af50_representatives:
        return get_af50_representatives(cluster_path)
    else:
        return get_foldseek_representatives(cluster_path)


def save_pdbs_to_parquet(save_dir, scratch_dir, clusters_to_save, parquet_id, verbose=False):
    # Save the pdbs to parquet
    results = []
    pdb_files = [os.path.join(scratch_dir, str(parquet_id), f"AF-{cluster_id}-F1-model_v4" + ".pdb") for cluster_id in clusters_to_save]
    structure_tokens = convert_pdbs_to_3di(pdb_files, os.path.join(scratch_dir, str(parquet_id), "foldseek_descriptors.tsv"))
    assert len(structure_tokens) == len(pdb_files)
    for cluster_id, seq_3di, pdb in zip(clusters_to_save, structure_tokens, pdb_files):
        structure = load_structure(pdb, chain="A", extra_fields=["b_factor"])
        coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)  # residues, atoms, xyz
        residue_identities = get_residues(structure)[1]
        b_factors = structure.b_factor[get_residue_starts(structure)]
        seq = "".join(
            [ProteinSequence.convert_letter_3to1(r) for r in residue_identities]
        )
        coords = {}
        for ix, atom_name in enumerate(["N", "CA", "C", "O"]):
            coords[atom_name] = coords[:, ix, :].flatten()

        # https://github.com/steineggerlab/foldseek/issues/273
        # for preprocessing, one idea would be to batch sequences into pseudo-documents, separated
        # by an end-of-document token.
        # during a batched map, we will get lists of items compatible with existing preprocessors.
        # although we need to be careful about sep token.

        results.append(
            {
                "sequence": seq,
                "accession": cluster_id,
                "N": coords["N"],
                "CA": coords["CA"],
                "C": coords["C"],
                "O": coords["O"],
                "plddt": b_factors,
                "3di": seq_3di,
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
    cluster_path,
    use_af50_representatives=False,
    zip_index_file=None,
):
    cluster_ids = get_cluster_ids(cluster_path, use_af50_representatives=use_af50_representatives)

    # shuffle first so that we de-correlate cluster identities in parquet files
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cluster_ids)

    parquet_size = 1000  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + parquet_size] for i in range(0, len(cluster_ids), parquet_size)]
    cluster_ids = clusters_to_save[parquet_id]

    zip_index = zip_index_file or "/SAN/bioinf/afdb_domain/zipmaker/zip_index"
    print("reading zip index from file", zip_index, flush=True)
    af2zip = make_zip_dictionary(zip_index, cluster_ids)

    pdb_lookup = defaultdict(list)

    print("Building lookup", flush=True)

    t1 = time.time()
    for ix, cluster_id in enumerate(cluster_ids):
        if ix % 500 == 0:
            print(f"Processing cluster {ix} of {len(cluster_ids)}", flush=True)
        
        zip_filename = af2zip[cluster_id]
        afdb_id = f"AF-{cluster_id}-F1-model_v4"
        pdb_lookup[zip_filename].append(afdb_id)

    t2 = time.time()
    print("Built lookup in", t2 - t1, "seconds", flush=True)
    print("Number of zip files: ", len(pdb_lookup), flush=True)
    return pdb_lookup, cluster_ids


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
    use_af50_representatives=False,
    parquet_ids=None,
    num_processes=None,
):
    if parquet_ids is None:
        if use_af50_representatives:
            parquet_ids = range(52330)
        else:
            # 2302908 clusters
            parquet_ids = range(2310)

    print(f"Post-shuffle cluster ids: {cluster_ids[:10]}", flush=True)
    af50_path = os.path.join(scratch_dir, "5-allmembers-repId-entryId-cluFlag-taxId.tsv")
    for parquet_id in parquet_ids:
        print("Processing parquet id", parquet_id, flush=True)
        pdb_lookup, cluster_ids = make_job_list(
            parquet_id,
            cluster_path=af50_path if use_af50_representatives else cluster_path,
            use_af50_representatives=use_af50_representatives,
            zip_index_file=os.path.join(scratch_dir, "zip_index"),
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
            clusters_to_save=cluster_ids,
            parquet_id=parquet_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("scratch_dir")
    parser.add_argument("--af50", action="store_true")
    parser.add_argument("--parquet_ids", type=int, default=None, nargs="+")
    parser.add_argument("--num_processes", type=int, default=None)
    args = parser.parse_args()

    if args.af50:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_representatives/"
    else:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_representatives_example/"

    if args.num_processes is None:
        args.num_processes = os.cpu_count()
    print("Num cpus", os.cpu_count(), flush=True)

    create_foldseek_parquets(
        save_dir=save_dir,
        scratch_dir=args.scratch_dir,
        parquet_ids=args.parquet_ids,
        num_processes=args.num_processes,
    )
