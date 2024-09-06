"""Uses the all-vs-all foldseek results to create clusters of clusters, similar to PoET.

We already have clusters within the parquet files. So we want to use the parquet index
to do cluster lookups and the all-vs-all file to do cluster merges.

We then write the outputs to a new set of parquet files.
"""
import argparse
import os
import time
from dask import dataframe as dd
import numpy as np
import pandas as pd
import pickle

from .utils import make_cluster_dictionary


AFDB_DATA_PATH = "/SAN/orengolab/cath_plm/ProFam/data/afdb/"


def load_parquet_index(index_file_path):
    index_df = pd.read_csv(index_file_path).set_index("identifier")
    print(index_df.head(), flush=True)
    cluster_to_parquet = index_df["parquet_file"].to_dict()
    return cluster_to_parquet


def load_all_vs_all(all_vs_all_path):
    ddf = dd.read_csv(all_vs_all_path, sep="\t", header=None, names=["query_id", "target_id", "evalue"])
    return ddf


def get_cluster_of_cluster_members(cluster_ids, ddf, evalue_threshold=1e-3, num_processes=None):
    # Q. does a cluster id search against itself? A. yes
    # so what we should do is just find query is cluster_id
    result = ddf[
        (ddf["query_id"].isin(cluster_ids))&(ddf["evalue"]<=evalue_threshold)
    ].compute(scheduler="threads", num_workers=num_processes)
    return [result[result["query_id"] == cluster_id]["target_id"].tolist() for cluster_id in cluster_ids]


def save_single_parquet(
    cluster_ids,
    output_file,
    ddf,
    parquet_index,
    parquet_dir,
    evalue_threshold=0.001,
    minimum_cluster_size=1,
    identifier_col="cluster_id",
    with_structure=False,
    num_processes=None,
):
    """N.B. clusters can be missing: in particular, fragments."""
    records = []

    t0 = time.time()
    all_cluster_of_cluster_members = get_cluster_of_cluster_members(
        cluster_ids,
        ddf,
        evalue_threshold=evalue_threshold,
        num_processes=num_processes
    )
    t1 = time.time()
    print("Time to query all-vs-all:", t1 - t0, "seconds", flush=True)
    assert len(all_cluster_of_cluster_members) == len(cluster_ids), "Mismatch in number of clusters"
    for cluster_id, cluster_of_cluster_members in zip(cluster_ids, all_cluster_of_cluster_members):
        # TODO: check if self-comparison is included in all-vs-all file. It is.
        # TODO: verify that foldseek clusters are unique so there are no duplicates
        # TODO: we actually have a problem with loading from parquets which is that the
        # clusters with fewer than 10 members are not currently included in the parquet files.
        # So to include these we need to manually load the corresponding pdb files.
        if len(cluster_of_cluster_members) >= minimum_cluster_size:  # this is at the foldseek level i guess rather than af50 level.
            sequences = []
            accessions = []
            Ns = []
            CAs = []
            Cs = []
            Os = []
            foldseek_cluster_ids = []
            # TODO: add af50 cluster ids...
            is_foldseek_representative = []
            is_af50_representative = []
            plddts = []

            missing_members = []
            for member_id in cluster_of_cluster_members:
                try:
                    # N.B. parquet index contains foldseek cluster representatives - not af50 representatives
                    # what does all vs all contain? I think also foldseek cluster representatives - but it
                    # also contains fragments that were removed (i.e. entries assigned flag 3 or 4 in 5-allmembers))
                    parquet_file = os.path.join(parquet_dir, parquet_index[member_id])
                    df = pd.read_parquet(parquet_file).set_index(identifier_col)
                    entry = df.loc[member_id]

                    sequences += entry["sequences"].tolist()
                    accessions += entry["accessions"].tolist()
                    foldseek_cluster_ids += entry[args.identifier_col].tolist()
                    is_foldseek_representative += entry["is_foldseek_representative"].tolist()
                    is_af50_representative += entry["is_af50_representative"].tolist()
                    if with_structure:
                        Ns += entry["N"].tolist()
                        CAs += entry["CA"].tolist()
                        Cs += entry["C"].tolist()
                        Os += entry["O"].tolist()
                        plddts += entry["plddts"].tlist()
                except KeyError:
                    # print(f"Could not find member {member_id} in parquet index")
                    missing_members.append(member_id)

            print(f"Could not find {missing_members} in parquet index", flush=True)
            if len(sequences) > 0:
                # TODO: should we run foldmason on these clusters of clusters? they might be too divergent...
                assert len(set(accessions)) == len(accessions), "Accessions are not unique"
                d = {
                    "fam_id": cluster_id,
                    "sequences": sequences,
                    "accessions": accessions,
                    "is_foldseek_representative": is_foldseek_representative,
                    "is_af50_representative": is_af50_representative,
                }
                if with_structure:
                    d["N"] = Ns
                    d["CA"] = CAs
                    d["C"] = Cs
                    d["O"] = Os
                    d["plddts"] = plddts
                records.append(d)

    df = pd.DataFrame(records)
    df.to_parquet(output_file)
    t2 = time.time()
    print("Time to build parquet", t2 - t1, "seconds", flush=True)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # TODO: instead of loading the cluster dictionary we can just save a file which lists the cluster sizes.
    cluster_dict_pickle_path = os.path.join(args.save_dir, "foldseek_cluster_dict.pkl")
    parquet_dir = os.path.dirname(args.index_file_path)

    if not os.path.exists(cluster_dict_pickle_path):
        print("Creating foldseek dataset", flush=True)
        cluster_dict = make_cluster_dictionary(args.cluster_path)
        print("Number of clusters:", len(cluster_dict))
        print("Saving cluster dictionary")
        with open(args.save_dir + "foldseek_cluster_dict.pkl", "wb") as f:
            pickle.dump(cluster_dict, f)
    else:
        print("Loading cluster dictionary", flush=True)
        with open(cluster_dict_pickle_path, "rb") as f:
            cluster_dict = pickle.load(f)
        print("Number of clusters:", len(cluster_dict), flush=True)

    # shuffle first so that we de-correlate cluster identities in parquet files
    cluster_ids = sorted(list(cluster_dict.keys()))
    # 442338 clusters with >= 10 members
    rng = np.random.default_rng(seed=83)  # may as well seed differently here than for other foldseek
    rng.shuffle(cluster_ids)
    print(f"Post-shuffle cluster ids: {cluster_ids[:10]}", flush=True)


    # all_accessions = [member_id for cluster_id in cluster_ids for member_id in cluster_dict[cluster_id]]
    print("Loading parquet index", flush=True)
    parquet_index = load_parquet_index(args.index_file_path)
    ddf = load_all_vs_all(args.all_vs_all_path)

    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + args.parquet_size] for i in range(0, len(cluster_ids), args.parquet_size)]

    if args.parquet_id is None:
        parquet_ids = range(len(clusters_to_save))
    else:
        parquet_ids = [args.parquet_id]

    for parquet_id in parquet_ids:
        output_file = os.path.join(args.save_dir, f"{parquet_id}.parquet")
        if args.force_rerun or not os.path.isfile(output_file):
            cluster_ids = clusters_to_save[parquet_id]
            print(
                f"Saving {len(cluster_ids)} clusters (loading member info from parquet"
                f"dir {parquet_dir}) to parquet file {parquet_id}", flush=True
            )
            t0 = time.time()
            save_single_parquet(
                cluster_ids=cluster_ids,
                output_file=output_file,
                ddf=ddf,
                parquet_index=parquet_index,
                evalue_threshold=args.evalue_threshold,
                minimum_cluster_size=args.minimum_cluster_size,
                identifier_col=args.identifier_col,
                with_structure=args.with_structure,
                parquet_dir=parquet_dir,
                num_processes=args.num_processes,
            )
            t1 = time.time()
            print("Saved in", t1 - t0, "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file_path", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument(
        "--cluster_path",
        type=str,
        default=f"{AFDB_DATA_PATH}/1-AFDBClusters-entryId_repId_taxId.tsv",
    )
    parser.add_argument(
        "--all_vs_all_path",
        type=str,
        default=f"{AFDB_DATA_PATH}/6-all-vs-all-similarity-queryId_targetId_eValue.tsv",
    )
    parser.add_argument("--parquet_id", type=int, default=None)
    parser.add_argument("--parquet_size", type=int, default=100)
    parser.add_argument("--identifier_col", default="fam_id")
    parser.add_argument("--with_structure", action="store_true")
    parser.add_argument("--minimum_cluster_size", type=int, default=1)
    parser.add_argument("--evalue_threshold", type=float, default=1e-3)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--num_processes", type=int, default=None)
    args = parser.parse_args()
    main(args)
