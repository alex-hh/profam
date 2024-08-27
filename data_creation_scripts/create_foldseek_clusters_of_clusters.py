"""Uses the all-vs-all foldseek results to create clusters of clusters, similar to PoET.

We already have clusters within the parquet files. So we want to use the parquet index
to do cluster lookups and the all-vs-all file to do cluster merges.

We then write the outputs to a new set of parquet files.
"""
import argparse
import os
from dask import dataframe as dd
import numpy as np
import pandas as pd
import pickle


def load_parquet_index(index_file_path):
    cluster_to_parquet = {}
    with open(index_file_path, "r") as f:
        for line in f:
            if line.startswith("identifier,parquet_file"):
                continue
            identifier, parquet_file = line.strip().split(",")
            cluster_to_parquet[identifier] = parquet_file
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


def main(args):
    if args.skip_af50:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_aug_struct"
    else:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_aug_af50_struct"

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
    cluster_ids = sorted(list(cluster_dict.keys()))
    # 442338 clusters with >= 10 members
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cluster_ids)
    print(f"Post-shuffle cluster ids: {cluster_ids[:10]}", flush=True)

    parquet_size = 250 if args.skip_af50 else 100  # number of clusters to save in each parquet file
    # What we want to do here is build a list of cluster ids to save within each parquet file.
    clusters_to_save = [cluster_ids[i:i + parquet_size] for i in range(0, len(cluster_ids), parquet_size)]
    cluster_ids = clusters_to_save[args.parquet_id]

    print(f"Saving {len(cluster_ids)} clusters to parquet file {args.parquet_id}", flush=True)

    # all_accessions = [member_id for cluster_id in cluster_ids for member_id in cluster_dict[cluster_id]]
    parquet_index = load_parquet_index(args.index_file_path)
    ddf = load_all_vs_all("/SAN/orengolab/cath_plm/ProFam/data/afdb/6-all-vs-all-similarity-queryId_targetId_eValue.tsv")

    records = []

    for cluster_id in cluster_ids:
        # TODO: check if self-comparison is included in all-vs-all file
        # TODO: would dask help
        # TODO: verify that foldseek clusters are unique so there are no duplicates
        # TODO: we actually have a problem with loading from parquets which is that the
        # singleton clusters are not included in the parquet files. So to include these
        # we need to manually load the corresponding pdb files.
        cluster_of_cluster_members = get_cluster_of_cluster_members(cluster_id, ddf, evalue_threshold=args.evalue_threshold)
        if len(cluster_of_cluster_members) >= args.minimum_cluster_size:  # this is at the foldseek level i guess rather than af50 level.
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
                parquet_file = parquet_index[member_id]
                df = pd.read_parquet(parquet_file).set_index("cluster_id")
                entry = df.loc[cluster_id]
                sequences += entry["sequences"]
                accessions += entry["accessions"]
                Ns += entry["N"]
                CAs += entry["CA"]
                Cs += entry["C"]
                Os += entry["O"]
                is_foldseek_representative += entry["is_foldseek_representative"]
                is_af50_representative += entry["is_af50_representative"]
                plddts += entry["plddts"]
            
            # TODO: should we run foldmason on these clusters of clusters? they might be too divergent...
            assert len(set(accessions)) == len(accessions), "Accessions are not unique"
            records.append({
                "cluster_id": cluster_id,
                "sequences": sequences,
                "accessions": accessions,
                "N": Ns,
                "CA": CAs,
                "C": Cs,
                "O": Os,
                "is_foldseek_representative": is_foldseek_representative,
                "is_af50_representative": is_af50_representative,
                "plddts": plddts
            })
        
    df = pd.DataFrame(records)
    output_file = os.path.join(save_dir, f"{args.parquet_id}.parquet")
    df.to_parquet(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_id", type=int, default=0)
    parser.add_argument("index_file_path", type=str)
    parser.add_argument("--skip_af50", action="store_true")
    parser.add_argument("--evalue_threshold", type=float, default=1e-3)
