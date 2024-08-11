import pandas as pd
import os
"""
"""
ec_val_dir = "../data/ec/jsunn-y_care_EC_datasets/splits/task1"
c_30_50 = f"{ec_val_dir}/30-50_protein_test.csv"
c_50_70 = f"{ec_val_dir}/50-70_protein_test.csv"
c_70_90 = f"{ec_val_dir}/70-90_protein_test.csv"

val_tasks = {}

val_dfs = []

for cluster, c_path in zip(["30-50", "50-70", "70-90"], [c_30_50, c_50_70, c_70_90]):
    c = pd.read_csv(c_path)
    print(f"\nlen {cluster}:", len(c))
    print(f"Unique EC in {cluster}:", len(c["EC number"].unique()))
    for c_level in ["clusterRes30", "clusterRes50", "clusterRes70", "clusterRes90"]:
        for cid in c[c_level].unique():
            matched = c[c[c_level]==cid].copy()
            n_ec_in_clst = len(matched["EC number"].unique())
            if n_ec_in_clst > 1:
                print(f"Cluster {cid} in {cluster} has {n_ec_in_clst} ECs")
                matched["val_cluster_min_max"] = cluster
                matched["val_cluster_id"] = cid
                matched["val_cluster_level"] = c_level
                val_dfs.append(matched)
val_df = pd.concat(val_dfs)
val_df.to_csv("../data/ec/jsunn-y_care_EC_datasets/val_clustered_seqs_w_different_ec_nums.csv", index=False)

"""
We will be classifying 2 or 3 sequences from each val_cluster_id
corresponding to 2 or 3 different EC numbers
for each EC number we need to load the prompt and the evaluation sequences
"""


