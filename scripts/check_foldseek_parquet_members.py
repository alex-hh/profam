"""Iterate over parquets.

For each parquet, sample a random set of clusters, and verify that the cluster members are
in agreement with the cluster dictionary.
"""
import argparse
import glob
import os
import pandas as pd

def make_af50_dictionary(af50_path, clusters_to_include=None):
    line_counter = 0
    af50_dict = {}
    with open(af50_path, "r") as f:
        for line in f:
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


def main(args):
    cluster_dict = make_cluster_dictionary("/SAN/orengolab/cath_plm/ProFam/afdb/1-AFDBClusters-entryId_repId_taxId.tsv")
    if args.include_af50:
        af50_dict = make_af50_dictionary("/SAN/orengolab/cath_plm/ProFam/afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv")
    with open(args.index_file_path, "w") as f:
        files = glob.glob(args.data_file_pattern)
        print(f"Found {len(files)} files matching pattern {args.data_file_pattern}")
        for file in files:
            df = pd.read_parquet(file)
            df = df.sample(n=args.sample_size)
            for _, row in df.iterrows():
                accessions = row["accessions"]
                expected_accessions = cluster_dict[row[args.identifier_col]]
                if args.include_af50:
                    for af50_accession in expected_accessions:
                        extra_members = af50_dict.get(af50_accession, [])
                        expected_accessions.extend(extra_members)

                assert set(accessions) == set(expected_accessions), f"Accessions do not match for {row[args.identifier_col]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file_path")
    parser.add_argument("data_file_pattern")
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--include_af50", action="store_true")
    parser.add_argument("--identifier_col", default="cluster_id")
    args = parser.parse_args()
    main(args)
