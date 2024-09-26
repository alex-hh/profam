"""
Precompute filtered zip index and af50 dictionaries for FoldSeek dataset
Save dictionaries for each parquet index.

Idea is that we can run this once and then use to build job files for
building of any of parquet sets.

N.B. these job files double up as a db.

Requires around 60GB memory.
"""
import argparse
import pandas as pd
import numpy as np
import time
import pickle
import os
import tqdm


AFDB_INDEX_FILES_PATH = os.environ.get("AFDB_INDEX_FILES_PATH", "/SAN/orengolab/cath_plm/ProFam/data/afdb")


def make_job_df(
    cluster_dict,
    af2zip,
    af50_dict,
    parquet_index,
    skip_af50=False,
):
    cluster_counter = 0
    t0 = time.time()

    print("Building job df", flush=True)
    records = []

    for ix, (cluster_id, members) in enumerate(cluster_dict.items()):
        if ix % 500 == 0:
            print(f"Processing cluster {ix} of {len(cluster_dict)}", flush=True)
        members = cluster_dict[cluster_id]
        cluster_counter += 1

        for member in members:
            try:
                zip_filename = af2zip[member]
            except:
                print("Error looking up", member)
                zip_filename = ""
            records.append(
                {
                    "cluster_id": cluster_id,
                    "accession": member,
                    "af50_cluster_id": member,
                    "zip_filename": zip_filename,
                    "parquet_index": parquet_index,
                }
            )

            if not skip_af50:  # this is the af50 representative
                af50_members = af50_dict.get(member, [])
                for af50_member in af50_members:
                    assert af50_member != member
                    try:
                        zip_filename = af2zip[af50_member]
                    except:
                        print("Error looking up", af50_member)
                        zip_filename = ""
                    records.append(
                        {
                            "cluster_id": cluster_id,
                            "af50_cluster_id": member,
                            "accession": af50_member,
                            "zip_filename": zip_filename,
                            "parquet_index": parquet_index,
                        }
                    )

    t1 = time.time()
    print("Built job df in", t1 - t0, "seconds", flush=True)
    return pd.DataFrame.from_records(records)


def create_foldseek_job_files(
    save_dir,
    minimum_foldseek_cluster_size=1,
    skip_af50=False,
    show_tqdm=False,
):
    
    t0 = time.time()
    # TODO: instead of loading the cluster dictionary we can just save a file which lists the cluster sizes.
    cluster_dict_pickle_path = os.path.join(AFDB_INDEX_FILES_PATH, "foldseek_cluster_dict.pkl")

    with open(cluster_dict_pickle_path, "rb") as f:
        cluster_dict = pickle.load(f)
    print("Number of clusters:", len(cluster_dict), flush=True)

    # shuffle first so that we de-correlate cluster identities in parquet files
    cluster_dict = {k: v for k, v in cluster_dict.items() if len(v) >= minimum_foldseek_cluster_size}
    cluster_ids = sorted(list(cluster_dict.keys()))
    # 442338 clusters with >= 10 members
    print(f"Number of clusters after filtering by cluster size >= {minimum_foldseek_cluster_size}:", len(cluster_dict.keys()), flush=True)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cluster_ids)
    print(f"Post-shuffle cluster ids: {cluster_ids[:10]}", flush=True)

    parquet_size = 250  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + parquet_size] for i in range(0, len(cluster_ids), parquet_size)]
    print(f"Number of parquet files: {len(clusters_to_save)}", flush=True)
    # cluster_ids = clusters_to_save[parquet_id]
    t1 = time.time()
    print("Finished reading cluster dictionary", t1 - t0, flush=True)

    with open(os.path.join(AFDB_INDEX_FILES_PATH, "af50_cluster_dict.pkl"), "rb") as f:
        af50_dict = pickle.load(f)

    print("Finished reading af50 dictionary", flush=True)

    with open(os.path.join(AFDB_INDEX_FILES_PATH, "af2zip.pkl"), "rb") as f:
        af2zip = pickle.load(f)

    print("Finished reading af2zip dictionary", flush=True)

    for parquet_id, parquet_cluster_ids in tqdm.tqdm(enumerate(clusters_to_save), disable=not show_tqdm):
        output_file = os.path.join(save_dir, f"job_{parquet_id}.csv")
        if not os.path.isfile(output_file):
            parquet_cluster_dict = {cluster_id: cluster_dict[cluster_id] for cluster_id in parquet_cluster_ids}

            # we might want to build dicts just once, including af50, and skip those later by appropriate filtering.
            job_df = make_job_df(
                parquet_cluster_dict,
                af2zip,
                af50_dict,
                parquet_index=parquet_id,
                skip_af50=skip_af50,
            )
            job_df.to_csv(output_file, index=False)
            print(f"Saved job {parquet_id} with {len(job_df)} entries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_foldseek_cluster_size", type=int, default=1)
    parser.add_argument("--skip_af50", action="store_true")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--show_tqdm", action="store_true")
    args = parser.parse_args()

    if args.save_dir is None:
        if args.skip_af50:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/"
        else:
            save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct/"
        print("Saving to inferred directory", save_dir, flush=True)
    else:
        print("Saving to passed directory", args.save_dir, flush=True)
        save_dir = args.save_dir

    create_foldseek_job_files(
        save_dir=save_dir,
        minimum_foldseek_cluster_size=args.minimum_foldseek_cluster_size,
        skip_af50=args.skip_af50,
        show_tqdm=args.show_tqdm,
    )
