"""
Creates fasta files for the foldseek clusters
converts them into parquet files.

first create a dictionary for the clusters
then iterate through the file.

N.B. this runs very quickly.
"""
import argparse
import random
import pickle
import numpy as np
import os
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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


def make_cluster_dictionary(cluster_path):
    line_counter = 0
    cluster_dict = {}
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            entry_id = line[0]
            rep_id = line[1]
            tax_id = line[2]
            if rep_id not in cluster_dict:
                cluster_dict[rep_id] = []
            cluster_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return cluster_dict


def make_af50_dictionary(af50_path):
    line_counter = 0
    af50_dict = {}
    with open(af50_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rep_id = line[0]
            entry_id = line[1]
            # 1: clustered in AFDB50, 2: clustered in AFDB clusters, 3/4: removed (fragments/singletons)
            clu_flag = int(line[2])  # 1
            # n.b. the 2s are duplicates of the other cluster dict
            # n.b. we don't include the representative in its own cluster atm
            if clu_flag == 1:
                if rep_id not in af50_dict:
                    af50_dict[rep_id] = []
                af50_dict[rep_id].append(entry_id)
            line_counter += 1
            if line_counter % 100000 == 0:
                print("Processed", line_counter, "lines for cluster dictionary")
    return af50_dict


def save_clusters_to_parquet(cluster_df, save_dir, cluster_counter):
    table = pa.Table.from_pandas(cluster_df)
    output_file = os.path.join(f'{save_dir}', f'{cluster_counter}.parquet')
    pq.write_table(table, output_file)
    print(f"Saved {len(cluster_df)} clusters to {output_file}")
    return output_file


def create_foldseek_parquets(cluster_dict, af50_dict, sequence_dict, save_dir, skip_af50=False):
    seq_fail_counter = 0
    seq_success_counter = 0
    cluster_counter = 0
    seq_fail_path = save_dir + "failed_sequences.txt"

    # shuffle first so that we de-correlate cluster identities in parquet files
    # TODO: seed this
    clusters = list(cluster_dict.keys())
    rng = np.random.default_rng(seed=42)
    rng.shuffle(clusters)

    results = []

    with open(seq_fail_path, "w") as fail_file:
        for foldseek_cluster_id in clusters:
            sequences = []
            is_foldseek_representative = []
            is_af50_representative = []
            accessions = []
            members = cluster_dict[foldseek_cluster_id]
            cluster_counter += 1
            for member in members:
                # foldseek cluster should be contained in self but af50 should not (pure convention)
                try:
                    sequence = sequence_dict[foldseek_cluster_id]
                    sequences.append(sequence)
                    is_foldseek_representative.append(member == foldseek_cluster_id)
                    is_af50_representative.append(True)
                    accessions.append(member)
                    seq_success_counter += 1
                except:
                    seq_fail_counter += 1
                    fail_file.write(member + "\n")
                
                if not skip_af50 and member in af50_dict:
                    for af50_member in af50_dict[member]:
                        assert not af50_member == member
                        try:
                            sequence = sequence_dict[af50_member]
                            sequences.append(sequence)
                            is_foldseek_representative.append(False)
                            is_af50_representative.append(False)
                            seq_success_counter += 1
                            accessions.append(af50_member)
                        except:
                            seq_fail_counter += 1
                            fail_file.write(af50_member + "\n")

            res = {
                "sequences": sequences,
                "cluster_id": foldseek_cluster_id,
                "is_foldseek_representative": is_foldseek_representative,
                "accessions": accessions,
            }
            if not skip_af50:
                res["is_af50_representative"] = is_af50_representative
            results.append(res)
            if cluster_counter % 10000 == 0:
                print("\nProcessed", cluster_counter, "clusters")
                print("Number of failed sequences:", seq_fail_counter)
                print("Number of successful sequences:", seq_success_counter)
                save_clusters_to_parquet(pd.DataFrame(results), save_dir, cluster_counter)
                results = []

        if len(results) > 0:
            print("\nProcessed", cluster_counter, "clusters")
            print("Number of failed sequences:", seq_fail_counter)
            print("Number of successful sequences:", seq_success_counter)
            save_clusters_to_parquet(pd.DataFrame(results), save_dir, cluster_counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_path")  # use a subset of the clusters to test the script
    parser.add_argument("--skip_af50", action="store_true")
    args = parser.parse_args()

    # cluster_path = "/SAN/orengolab/cath_plm/ProFam/data/foldseek/1-AFDBClusters-entryId_repId_taxId.tsv"
    if args.skip_af50:
        print("Skipping af50 members in document creation")
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_v2/"
    else:
        save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50/"
    af50_path = "/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50/5-allmembers-repId-entryId-cluFlag-taxId.tsv"
    afdb_fasta_path = "/SAN/orengolab/cath_plm/ProFam/data/afdb/sequences.fasta"

    cluster_dict_pickle_path = os.path.join(save_dir, "foldseek_cluster_dict.pkl")
    af50_dict_pickle_path = os.path.join(save_dir, "af50_cluster_dict.pkl")
    sequence_dict_pickle_path = os.path.join(save_dir, "afdb_sequence_dict.pkl")

    if not os.path.exists(cluster_dict_pickle_path):
        print("Creating foldseek dataset")
        cluster_dict = make_cluster_dictionary(args.cluster_path)
        print("Number of clusters:", len(cluster_dict))
        print("Saving cluster dictionary")
        with open(cluster_dict_pickle_path, "wb") as f:
            pickle.dump(cluster_dict, f)
    else:
        print("Loading cluster dictionary")
        with open(cluster_dict_pickle_path, "rb") as f:
            cluster_dict = pickle.load(f)
        print("Number of clusters:", len(cluster_dict))

    if not os.path.exists(af50_dict_pickle_path):
        af50_dict = make_af50_dictionary(af50_path)
        with open(af50_dict_pickle_path, "wb") as f:
            pickle.dump(af50_dict, f)
    else:
        with open(af50_dict_pickle_path, "rb") as f:
            af50_dict = pickle.load(f)

    sequence_dict = make_sequence_dictionary(afdb_fasta_path)
    # with 128 GB memory, can't save sequence dict
    # with open(sequence_dict_pickle_path, "wb") as f:
    #     pickle.dump(sequence_dict, f)
    # else:
    #     with open(sequence_dict_pickle_path, "rb") as f:
    #         sequence_dict = pickle.load(f)

    create_foldseek_parquets(cluster_dict, af50_dict, sequence_dict, save_dir, skip_af50=args.skip_af50s)
