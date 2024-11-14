"""
Loads PDB files for the foldseek clusters and
converts them into parquet files. The parquet files contain sequences,
foldmason MSTA alignments (both aa and 3di),
as well as backbone coordinates (N, Ca, C, O).

TODO: consider adding option to merge parquets (for use_representative)
"""
import argparse
import shutil
from collections import defaultdict
import multiprocessing
import tqdm
from biotite.sequence import ProteinSequence
from biotite.structure.residues import get_residues, get_residue_starts
import time
import os
import numpy as np
import pandas as pd
from modin import pandas as mpd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from src.constants import PROFAM_DATA_DIR
from src.sequence.fasta import read_fasta
from src.structure.pdb import get_atom_coords_residuewise, load_structure
from src.tools.foldmason import run_foldmason_on_pdbs
from src.tools.foldseek import convert_pdbs_to_3di
from .utils import extract_pdbs_from_zips


def save_pdbs_to_parquet(
    save_dir,
    pdbs_dir,
    clusters_to_save,
    parquet_id,
    metadata_lookup,
    run_foldmason=False,
    max_cluster_size_for_foldmason=None,
    convert_to_3di: bool = False,
    keep_pdbs: bool = False,
):
    # TODO: it would be cleaner for clusters_to_save values to be metadata-augmented dicts
    # Save the pdbs to parquet
    results = []
    for cluster_id, cluster_members in clusters_to_save.items():
        sequences = []
        accessions = []
        af50_cluster_id = []
        all_coords = {"N": [], "CA": [], "C": [], "O": []}
        all_b_factors = []
        cluster_filelist = []

        for afdb_id in cluster_members:
            pdb = os.path.join(pdbs_dir, afdb_id + ".pdb")
            try:
                structure = load_structure(pdb, chain="A", extra_fields=["b_factor"])
            except Exception as e:
                print(f"Error loading {pdb}: {e}")  # not added to filelist
                continue
            metadata = metadata_lookup[afdb_id]
            accessions.append(metadata["accession"])
            af50_cluster_id.append(metadata["af50_cluster_id"])
            cluster_filelist.append(pdb)
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)  # residues, atoms, xyz
            residue_identities = get_residues(structure)[1]
            b_factors = np.array(structure.b_factor[get_residue_starts(structure)]).astype("float16")
            seq = "".join(
                [ProteinSequence.convert_letter_3to1(r) for r in residue_identities]
            )
            all_b_factors.append(b_factors)
            sequences.append(seq)
            for ix, atom_name in enumerate(["N", "CA", "C", "O"]):
                # N.B. need to use recent versions of pyarrow for half float support
                all_coords[atom_name].append(coords[:, ix, :].flatten().astype("float16"))
            
        # Run FoldMason on the cluster
        has_foldmason_results = False
        if run_foldmason:
            if (
                    (max_cluster_size_for_foldmason is not None and len(cluster_filelist) > max_cluster_size_for_foldmason)
                    or len(cluster_filelist) < 3
            ):
                print(f"Skipping FoldMason for {cluster_id} due to size {len(cluster_filelist)}", flush=True)
            else:
                foldmason_outdir = os.path.join(pdbs_dir, cluster_id)
                os.makedirs(foldmason_outdir)
                run_foldmason_on_pdbs(cluster_filelist, foldmason_outdir, foldmason_outdir)

                # Read AA and 3Di alignments, skip the accessions
                labels, msta_seqs = read_fasta(os.path.join(foldmason_outdir, "result_aa.fa"))
                perm = [labels.index(afdb_id) for afdb_id in cluster_members]
                msta_seqs = [msta_seqs[ix] for ix in perm]
                struct_labels, msta_3di = read_fasta(os.path.join(foldmason_outdir, "result_3di.fa"))
                assert labels == struct_labels
                msta_3di = [msta_3di[ix] for ix in perm]
                shutil.rmtree(foldmason_outdir)
                has_foldmason_results = True

        if convert_to_3di:
            sequences_3di = convert_pdbs_to_3di(cluster_filelist, os.path.join(pdbs_dir, f"{cluster_id}_3di.tsv"))

        if not keep_pdbs:
            for pdb in cluster_filelist:
                os.remove(pdb)

        # TODO: save representative?
        res = {
            "sequences": sequences,
            "fam_id": cluster_id,
            "N": all_coords["N"],
            "CA": all_coords["CA"],
            "C": all_coords["C"],
            "O": all_coords["O"],
            "plddts": all_b_factors,
            "accessions": accessions,
            "af50_cluster_id": af50_cluster_id,
        }
        if has_foldmason_results:
            res["msta_seqs"] = msta_seqs
            res["msta_3di"] = msta_3di
        if convert_to_3di:
            res["sequences_3di"] = sequences_3di
        results.append(res)

    df = pd.DataFrame(results)
    # Q. why not just df.to_parquet?
    table = pa.Table.from_pandas(df)
    # https://github.com/apache/arrow/pull/42103 suggests that float16 is supported??
    print(table.schema)
    output_file = os.path.join(f'{save_dir}', f'{parquet_id}.parquet')
    pq.write_table(table, output_file)
    print(f"Saved {clusters_to_save} clusters to {output_file}")
    return output_file


def load_db(parquet_index=None):
    assert parquet_index is not None
    if parquet_index is None:
        df = mpd.read_csv_glob(os.path.join(PROFAM_DATA_DIR, "afdb/foldseek_job_files/job*.csv"))
    else:
        df = pd.read_csv(os.path.join(PROFAM_DATA_DIR, f"afdb/foldseek_job_files/job_{parquet_index}.csv"))
    return df


def make_job_list_for_parquet(
    db,
    pdb_lookup,
    skip_af50=False,
    cluster_col="cluster_id",
    minimum_foldseek_cluster_size=1,
    show_tqdm=False,
):
    cluster_ids = db[cluster_col].unique()
    t0 = time.time()
    cluster_counter = 0

    cluster_membership = defaultdict(list)
    metadata_lookup = dict()

    print("Building lookup", flush=True)

    for ix, cluster_id in tqdm.tqdm(enumerate(cluster_ids), disable=not show_tqdm):
        if ix % 50 == 0:
            print(f"Processing cluster {ix} of {len(cluster_ids)}", flush=True)
        if skip_af50:
            members = db[(db[cluster_col]==cluster_id)&(db["af50_cluster_id"]==db["accession"])]
        else:
            members = db[db[cluster_col]==cluster_id]
        cluster_counter += 1
        if len(members) < minimum_foldseek_cluster_size:
            continue

        for member, entry in members.iterrows():
            afdb_id = f"AF-{member}-F1-model_v4"
            if entry["zip_filename"] not in pdb_lookup:
                pdb_lookup[entry["zip_filename"]] = []
            pdb_lookup[entry["zip_filename"]].append(afdb_id)
            metadata_lookup[afdb_id] = {
                "cluster_id": entry["cluster_id"],
                "accession": member,
                "af50_cluster_id": entry["af50_cluster_id"],
            }
            cluster_membership[cluster_id].append(afdb_id)

    t1 = time.time()
    print("Built lookups in", t1 - t0, "seconds", flush=True)
    print("Number of zip files: ", len(pdb_lookup), flush=True)
    return metadata_lookup, cluster_membership


def create_foldseek_parquets(
    save_dir,
    scratch_dir,
    minimum_foldseek_cluster_size=1,
    skip_af50=False,
    parquet_ids=None,
    num_processes=None,
    representative_only=False,
    af50_representative_only=False,
    show_tqdm=False,
    run_foldmason=False,
    max_cluster_size_for_foldmason=None,
    keep_pdbs=False,
):
    print("Creating foldseek parquets", parquet_ids, flush=True)
    if parquet_ids is None:
        db = load_db()
        parquet_ids = list(range(len(db["parquet_index"].unique())))
    else:
        assert len(parquet_ids) == 1
        db = load_db(parquet_ids[0])

    db = db[(db["zip_filename"]!="")&(~db["zip_filename"].isnull())]

    cluster_col = "cluster_id"
    if representative_only:
        db = db[(db["cluster_id"] == db["accession"])]
    elif af50_representative_only:
        db = db[(db["af50_cluster_id"] == db["accession"])]
        cluster_col = "af50_cluster_id"

    db = db.set_index("accession", drop=False)

    if num_processes is None:
        pdb_lookup = dict()
    else:
        # TODO: debug:
        # 'ForkAwareLocal' object has no attribute 'connection'
        with multiprocessing.Manager() as manager:
            pdb_lookup = manager.dict()

    metadata_lookups = []
    cluster_memberships = []
    for parquet_id in parquet_ids:
        print("Making job list for parquet", parquet_id, flush=True)
        parquet_metadata_lookup, parquet_cluster_membership = make_job_list_for_parquet(
            db[db["parquet_index"] == parquet_id],
            pdb_lookup=pdb_lookup,
            skip_af50=skip_af50,
            minimum_foldseek_cluster_size=minimum_foldseek_cluster_size,
            cluster_col=cluster_col,
            show_tqdm=show_tqdm,
        )
        metadata_lookups.append(parquet_metadata_lookup)
        cluster_memberships.append(parquet_cluster_membership)

    print("Pdb lookup", pdb_lookup, flush=True)

    job_prefix = f"{parquet_ids[0]}-{parquet_ids[-1]}"
    if not keep_pdbs:
        os.makedirs(os.path.join(scratch_dir, job_prefix), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_dir, job_prefix), exist_ok=True)
    extract_pdbs_from_zips(
        pdb_lookup=pdb_lookup,
        output_dir=os.path.join(scratch_dir, job_prefix) if not keep_pdbs else os.path.join(save_dir, job_prefix),
        num_processes=num_processes,
    )

    for ix, parquet_id in enumerate(parquet_ids):
        print("Saving pdbs for parquet", parquet_id, parquet_cluster_membership, flush=True)
        parquet_cluster_membership = cluster_memberships[ix]
        parquet_metadata_lookup = metadata_lookups[ix]
        save_pdbs_to_parquet(
            save_dir=save_dir,
            pdbs_dir=os.path.join(scratch_dir, job_prefix) if not keep_pdbs else os.path.join(save_dir, job_prefix),
            clusters_to_save=parquet_cluster_membership,
            parquet_id=parquet_id,
            metadata_lookup=parquet_metadata_lookup,
            max_cluster_size_for_foldmason=max_cluster_size_for_foldmason,
            run_foldmason=run_foldmason and not (representative_only or af50_representative_only),
            convert_to_3di=(representative_only or af50_representative_only),
            keep_pdbs=keep_pdbs,
        )
    if not keep_pdbs:
        shutil.rmtree(os.path.join(scratch_dir, job_prefix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch_dir")
    parser.add_argument("--show_tqdm", action="store_true")
    parser.add_argument("--minimum_foldseek_cluster_size", type=int, default=1)
    parser.add_argument("--parquet_ids", type=int, default=None, nargs="+")
    parser.add_argument("--skip_af50", action="store_true")
    parser.add_argument("--run_foldmason", action="store_true")
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--representative_only", action="store_true")
    parser.add_argument("--af50_representative_only", action="store_true")
    parser.add_argument("--max_cluster_size_for_foldmason", type=int, default=None)
    parser.add_argument("--keep_pdbs", action="store_true")
    args = parser.parse_args()

    if args.save_dir is None:
        if args.representative_only:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_representatives/"
            assert not args.af50_representative_only
        elif args.af50_representative_only:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_representatives/"
        elif args.skip_af50:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/"
        else:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct/"
        print("Saving to inferred directory", save_dir, flush=True)
    else:
        print("Saving to passed directory", args.save_dir, flush=True)
        save_dir = args.save_dir

    create_foldseek_parquets(
        save_dir=save_dir,
        scratch_dir=args.scratch_dir,
        minimum_foldseek_cluster_size=args.minimum_foldseek_cluster_size,
        parquet_ids=args.parquet_ids,
        skip_af50=args.skip_af50,
        num_processes=args.num_processes,
        show_tqdm=args.show_tqdm,
        run_foldmason=args.run_foldmason,
        representative_only=args.representative_only,
        af50_representative_only=args.af50_representative_only,
        max_cluster_size_for_foldmason=args.max_cluster_size_for_foldmason,
        keep_pdbs=args.keep_pdbs,
    )
