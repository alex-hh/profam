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


def load_parquet_index(index_file_path):
    index_df = pd.read_csv(index_file_path).set_index("identifier")
    cluster_to_parquet = index_df["parquet_file"].to_dict()
    return cluster_to_parquet


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


def load_all_vs_all(all_vs_all_path):
    ddf = dd.read_csv(all_vs_all_path, sep="\t", header=None, names=["query_id", "target_id", "evalue"])
    return ddf


def get_cluster_of_cluster_members(cluster_id, ddf, evalue_threshold=1e-3):
    # Q. does a cluster id search against itself? A. yes
    # so what we should do is just find query is cluster_id
    result = ddf[(ddf["query_id"]==cluster_id)&(ddf["evalue"]<=evalue_threshold)].compute()
    return result["target_id"].tolist()


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
):
    records = []

    for cluster_id in cluster_ids:
        # TODO: check if self-comparison is included in all-vs-all file. It is.
        # TODO: verify that foldseek clusters are unique so there are no duplicates
        # TODO: we actually have a problem with loading from parquets which is that the
        # clusters with fewer than 10 members are not currently included in the parquet files.
        # So to include these we need to manually load the corresponding pdb files.
        cluster_of_cluster_members = get_cluster_of_cluster_members(cluster_id, ddf, evalue_threshold=evalue_threshold)
        if len(cluster_of_cluster_members) >= minimum_cluster_size:  # this is at the foldseek level i guess rather than af50 level.
            sequences = []
            accessions = []
            Ns = []
            CAs = []
            Cs = []
            Os = []
            is_foldseek_representative = []
            is_af50_representative = []
            plddts = []

            for member_id in cluster_of_cluster_members:
                parquet_file = os.path.join(parquet_dir, parquet_index[member_id])
                df = pd.read_parquet(parquet_file).set_index(identifier_col)
                entry = df.loc[cluster_id]
                sequences += entry["sequences"]
                accessions += entry["accessions"]
                is_foldseek_representative += entry["is_foldseek_representative"]
                is_af50_representative += entry["is_af50_representative"]
                if with_structure:
                    Ns += entry["N"]
                    CAs += entry["CA"]
                    Cs += entry["C"]
                    Os += entry["O"]
                    plddts += entry["plddts"]

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


def main(args):
    save_dir = os.path.join("/SAN/orengolab/cath_plm/ProFam/data/", args.output_folder)
    os.makedirs(save_dir, exist_ok=True)

    # TODO: instead of loading the cluster dictionary we can just save a file which lists the cluster sizes.
    cluster_dict_pickle_path = os.path.join(save_dir, "foldseek_cluster_dict.pkl")
    parquet_dir = os.path.dirname(args.index_file_path)

    if not os.path.exists(cluster_dict_pickle_path):
        print("Creating foldseek dataset", flush=True)
        cluster_dict = make_cluster_dictionary("/SAN/orengolab/cath_plm/ProFam/data/afdb/1-AFDBClusters-entryId_repId_taxId.tsv")
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
    cluster_ids = sorted(list(cluster_dict.keys()))
    # 442338 clusters with >= 10 members
    rng = np.random.default_rng(seed=83)  # may as well seed differently here than for other foldseek
    rng.shuffle(cluster_ids)
    print(f"Post-shuffle cluster ids: {cluster_ids[:10]}", flush=True)


    # all_accessions = [member_id for cluster_id in cluster_ids for member_id in cluster_dict[cluster_id]]
    print("Loading parquet index", flush=True)
    parquet_index = load_parquet_index(args.index_file_path)
    ddf = load_all_vs_all("/SAN/orengolab/cath_plm/ProFam/data/afdb/6-all-vs-all-similarity-queryId_targetId_eValue.tsv")

    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + args.parquet_size] for i in range(0, len(cluster_ids), args.parquet_size)]

    if args.parquet_id is None:
        parquet_ids = range(len(clusters_to_save))
    else:
        parquet_ids = [args.parquet_id]

    for parquet_id in parquet_ids:
        output_file = os.path.join(save_dir, f"{parquet_id}.parquet")
        if args.force_rerun or not os.path.isfile(output_file):
            cluster_ids = clusters_to_save[parquet_id]
            print(f"Saving {len(cluster_ids)} clusters (loading member info from parquet dir {parquet_dir}) to parquet file {args.parquet_id}", flush=True)
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
            )
            t1 = time.time()
            print("Saved in", t1 - t0, "seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file_path", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("--parquet_id", type=int, default=None)
    parser.add_argument("--parquet_size", type=int, default=100)
    parser.add_argument("--identifier_col", default="cluster_id")
    parser.add_argument("--with_structure", action="store_true")
    parser.add_argument("--minimum_cluster_size", type=int, default=1)
    parser.add_argument("--evalue_threshold", type=float, default=1e-3)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()
    main(args)
