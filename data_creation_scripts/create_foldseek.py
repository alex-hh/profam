"""
Creates fasta files for the foldseek clusters
converts them into parquet files.

first create a dictionary for the clusters
then iterate through the file.
"""
import random
import sys
import sqlite3
import pickle
import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq



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

def get_sequence_from_profam_db(uniprot_id, cursor):
    cursor.execute('SELECT sequence FROM sequences WHERE sequence_id = ?', (uniprot_id,))
    result = cursor.fetchone()
    return result[0] if result else None


def fasta_to_parquet(save_dir, batch_id):
    fastas = glob.glob(save_dir + "*.fasta")
    results = []
    for fasta in fastas:
        cluster_id = os.path.splitext(os.path.basename(fasta))[0]
        with open(fasta, "r") as f:
            text = f.read()
        results.append({'text': text, "cluster_id": cluster_id, "num_sequences": len(text.split(">")) - 1})
        os.remove(fasta)
    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    output_file = f'{save_dir}/{batch_id}.parquet'
    pq.write_table(table, output_file)
    print(f"Saved batch {batch_id} to {output_file}")
    return output_file


def create_foldseek_fastas(db_path, cluster_dict, save_dir):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    seq_fail_counter = 0
    seq_success_counter = 0
    cluster_counter = 0
    seq_fail_path = save_dir + "failed_sequences.txt"

    # shuffle first so that we de-correlate cluster identities in parquet files
    clusters = list(cluster_dict.keys())
    random.shuffle(clusters)

    with open(seq_fail_path, "w") as fail_file:
        for cluster_id in clusters:
            members = cluster_dict[cluster_id]
            cluster_counter += 1
            with open(save_dir + cluster_id + ".fasta", "w") as f:
                for member in members:
                    sequence = get_sequence_from_profam_db(member, cursor)
                    if sequence:
                        f.write(f">{member}\n")
                        f.write(sequence + "\n")
                        seq_success_counter += 1
                    else:
                        seq_fail_counter += 1
                        fail_file.write(member + "\n")
            if cluster_counter % 10000 == 0:
                print("\nProcessed", cluster_counter, "clusters")
                print("Number of failed sequences:", seq_fail_counter)
                print("Number of successful sequences:", seq_success_counter)
                fasta_to_parquet(save_dir, cluster_counter)
    fasta_to_parquet(save_dir, cluster_counter)


if __name__ == "__main__":
    scratch_dir = sys.argv[1]
    cluster_path = "../data/foldseek/1-AFDBClusters-entryId_repId_taxId.tsv"
    uniprot_db_path = f"{scratch_dir}/profam_db/profam.db"
    assert os.path.exists(uniprot_db_path), "Profam database not found"
    save_dir = "../data/foldseek/"
    cluster_dict_pickle_path = os.path.join(save_dir, "foldseek_cluster_dict.pkl")

    if not os.path.exists(cluster_dict_pickle_path):
        print("Creating foldseek dataset")
        cluster_dict = make_cluster_dictionary(cluster_path)
        print("Number of clusters:", len(cluster_dict))
        print("Saving cluster dictionary")
        with open(save_dir + "foldseek_cluster_dict.pkl", "wb") as f:
            pickle.dump(cluster_dict, f)
    else:
        print("Loading cluster dictionary")
        with open(cluster_dict_pickle_path, "rb") as f:
            cluster_dict = pickle.load(f)
        print("Number of clusters:", len(cluster_dict))
    create_foldseek_fastas(uniprot_db_path, cluster_dict, save_dir)



