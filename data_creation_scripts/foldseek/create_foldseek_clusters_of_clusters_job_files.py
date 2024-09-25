"""Uses the all-vs-all foldseek results to create clusters of clusters, similar to PoET.

We already have clusters within the parquet files. So we want to use the parquet index
to do cluster lookups and the all-vs-all file to do cluster merges.

We then write the outputs to a new set of parquet files.
"""
import argparse
import os
import time
from dask import dataframe as dd
import pandas as pd
from src.constants import PROFAM_DATA_DIR


AFDB_DATA_PATH = "/SAN/orengolab/cath_plm/ProFam/data/afdb/"


def load_all_vs_all(all_vs_all_path):
    ddf = dd.read_csv(all_vs_all_path, sep="\t", header=None, names=["query_id", "target_id", "evalue"])
    return ddf


def get_cluster_of_cluster_members(cluster_ids, ddf, evalue_threshold=1e-3, num_processes=None):
    # Q. does a cluster id search against itself? A. yes
    # so what we should do is just find query is cluster_id
    result = ddf[
        (ddf["query_id"].isin(cluster_ids))&(ddf["evalue"]<=evalue_threshold)
    ].compute(scheduler="threads", num_workers=num_processes or 1)
    return [result[result["query_id"] == cluster_id]["target_id"].tolist() for cluster_id in cluster_ids]


def save_job_file(ddf, index_df, parquet_index, output_dir, evalue_threshold=1e-3, num_processes=None):
    """N.B. clusters in all vs all file can be missing in parquets: in particular, fragments."""
    parquet_df = index_df[index_df["parquet_file"]==f"{parquet_index}.parquet"]
    cluster_ids = parquet_df.index.tolist()
    records = []

    t0 = time.time()
    all_cluster_of_cluster_members = get_cluster_of_cluster_members(
        cluster_ids,
        ddf,
        evalue_threshold=evalue_threshold,
        num_processes=num_processes,
    )
    t1 = time.time()
    print("Time to query all-vs-all:", t1 - t0, "seconds", flush=True)
    assert len(all_cluster_of_cluster_members) == len(cluster_ids), "Mismatch in number of clusters"
    for cluster_id, cluster_of_cluster_members in zip(cluster_ids, all_cluster_of_cluster_members):
        parquet_files = []
        members_to_keep = []
        total_clusters = 0
        total_members = 0
        for member in cluster_of_cluster_members:
            if member in index_df.index:
                member_info = index_df.loc[member]
                parquet_files.append(member_info["parquet_file"])
                members_to_keep.append(member)
                total_clusters += 1
                total_members += member_info["cluster_size"]

        res = {
            "cluster_id": cluster_id,
            "members": members_to_keep,  # will include self I think
            "parquet_files": parquet_files,
            "total_clusters": total_clusters,
            "total_members": total_members,
        }
        records.append(res)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, f"job_{parquet_index}.csv"), index=False)


def main(args):
    # need to use index df bc we need to know which parquet files to look up
    index_df = pd.read_csv(os.path.join(PROFAM_DATA_DIR, args.data_folder, "index.csv")).set_index("identifier")
    ddf = load_all_vs_all(args.all_vs_all_path)
    # n.b. really should be same for all - clusters of clusters don't change, so as long as cluster to parquet doesn't change we're good, but maybe cant assume that
    output_dir = os.path.join(PROFAM_DATA_DIR, "afdb/foldseek_job_files", args.data_folder)
    os.makedirs(output_dir, exist_ok=True)

    if args.parquet_id is None:
        parquet_ids = [int(f.replace(".parquet", "")) for f in index_df["parquet_file"].unique()]
    else:
        parquet_ids = [args.parquet_id]

    for parquet_id in parquet_ids:
        save_job_file(ddf, index_df, parquet_id, output_dir, evalue_threshold=args.evalue_threshold, num_processes=args.num_processes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder")
    parser.add_argument(
        "--all_vs_all_path",
        type=str,
        default=f"{AFDB_DATA_PATH}/6-all-vs-all-similarity-queryId_targetId_eValue.tsv",
    )
    parser.add_argument("--parquet_id", type=int, default=None)
    parser.add_argument("--evalue_threshold", type=float, default=1e-3)
    parser.add_argument("--num_processes", type=int, default=None)
    args = parser.parse_args()
    main(args)
