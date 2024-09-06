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
import pickle
import multiprocessing
from biotite.sequence import ProteinSequence
from biotite.structure.residues import get_residues, get_residue_starts
import numpy as np
import time
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from .utils import make_zip_dictionary, extract_pdbs_from_zips
from src.data.pdb import get_atom_coords_residuewise, load_structure


def get_af50_representatives(af50_path):
    representatives = []
    with open(af50_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip().split("\t")
        rep_id = line[0]
        # 1: clustered in AFDB50, 2: clustered in AFDB clusters, 3/4: removed (fragments/singletons)
        clu_flag = int(line[2])  # 1

        if (clu_flag == 1 or clu_flag == 2) and rep_id not in representatives:
            representatives.append(rep_id)
    return representatives


def get_foldseek_representatives(cluster_path):
    line_counter = 0
    representatives = []
    with open(cluster_path, "r") as f:
        lines = f.readlines()
    for line in lines:
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


def save_pdbs_to_parquet(save_dir, scratch_dir, clusters_to_save, parquet_id):
    # Save the pdbs to parquet
    results = []
    for cluster_id in clusters_to_save:
        afdb_id = f"AF-{cluster_id}-F1-model_v4"
        pdb = os.path.join(scratch_dir, str(parquet_id), afdb_id + ".pdb")

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

        # TODO: save representative?
        results.append(
            {
                "sequence": seq,
                "accession": cluster_id,
                "N": coords["N"],
                "CA": coords["CA"],
                "C": coords["C"],
                "O": coords["O"],
                "plddts": b_factors,
            }
        )
        os.remove(pdb)

    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    output_file = os.path.join(f'{save_dir}', f'{parquet_id}.parquet')
    pq.write_table(table, output_file)
    print(f"Saved {clusters_to_save} clusters to {output_file}")
    return output_file


def create_foldseek_parquets(
    save_dir,
    scratch_dir,
    cluster_path,
    use_af50_representatives=False,
    parquet_ids=None,
    num_processes=None,
):
    af50_path = os.path.join(scratch_dir, "5-allmembers-repId-entryId-cluFlag-taxId.tsv")
    cluster_path = af50_path if use_af50_representatives else cluster_path
    # TODO: get ids by loading appropriate dictionary and listing keys...
    all_cluster_ids = get_cluster_ids(cluster_path, use_af50_representatives=use_af50_representatives)

    # shuffle first so that we de-correlate cluster identities in parquet files
    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_cluster_ids)

    print(f"Post-shuffle cluster ids: {all_cluster_ids[:10]}", flush=True)

    parquet_size = 1000  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [all_cluster_ids[i:i + parquet_size] for i in range(0, len(all_cluster_ids), parquet_size)]

    check_accessions = True
    if parquet_ids is None:
        check_accessions = False
        parquet_ids = list(range(len(clusters_to_save)))

    cluster_ids_to_save = [cluster_id for parquet_id in parquet_ids for cluster_id in clusters_to_save[parquet_id]]


    zip_index_file = os.path.join(scratch_dir, "zip_index")
    if not check_accessions:
        zip_dict_path = os.path.join("/SAN/orengolab/cath_plm/ProFam/data/afdb", "zip_index_dict.pkl")
        if os.path.isfile(zip_dict_path):
            print("loading precomputed zip index")
            with open(zip_dict_path, "rb") as f:
                af2zip = pickle.load(f)
        else:
            print("reading zip index from file", zip_index_file, flush=True)
            af2zip = make_zip_dictionary(zip_index_file, None)
            with open(zip_dict_path, "wb") as f:
                pickle.dump(af2zip, f)
    else:
        print("reading zip index from file", zip_index_file, flush=True)
        af2zip = make_zip_dictionary(zip_index_file, cluster_ids_to_save)

    manager = multiprocessing.Manager()
    pdb_lookup = manager.dict()

    print("Building lookup", flush=True)

    t1 = time.time()
    for ix, cluster_id in enumerate(cluster_ids_to_save):
        if ix % 500 == 0:
            print(f"Processing cluster {ix} of {len(cluster_ids_to_save)}", flush=True)
        
        zip_filename = af2zip[cluster_id]
        afdb_id = f"AF-{cluster_id}-F1-model_v4"
        if zip_filename not in pdb_lookup:
            pdb_lookup[zip_filename] = []
        pdb_lookup[zip_filename].append(afdb_id)

    t2 = time.time()
    print("Built lookup in", t2 - t1, "seconds", flush=True)
    print("Number of zip files: ", len(pdb_lookup), flush=True)

    job_prefix = "-".join([str(i) for i in parquet_ids])
    extract_pdbs_from_zips(
        pdb_lookup=pdb_lookup,
        output_dir=os.path.join(scratch_dir, job_prefix),
        num_processes=num_processes,
    )

    for parquet_id in parquet_ids:
        print("Processing parquet id", parquet_id, flush=True)
        cluster_ids_for_parquet = clusters_to_save[parquet_id]
        save_pdbs_to_parquet(
            save_dir=save_dir,
            scratch_dir=scratch_dir,
            clusters_to_save=cluster_ids_for_parquet,
            parquet_id=parquet_id,
        )

    shutil.rmtree(os.path.join(scratch_dir, job_prefix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path", type=str, help="Path to the cluster file")
    parser.add_argument("scratch_dir")
    parser.add_argument("--af50", action="store_true")
    parser.add_argument("--parquet_ids", type=int, default=None, nargs="+")
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    if args.save_dir is not None:
        if args.af50:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_representatives/"
        else:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_representatives/"
    else:
        save_dir = args.save_dir

    create_foldseek_parquets(
        save_dir=save_dir,
        scratch_dir=args.scratch_dir,
        parquet_ids=args.parquet_ids,
        num_processes=args.num_processes,
        cluster_path=args.cluster_path,
    )
