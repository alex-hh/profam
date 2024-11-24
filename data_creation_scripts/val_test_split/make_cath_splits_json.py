"""
To build the JSON from scratch, the following files are needed:
- ESM_CATH_SPLIT: "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json"
- DOMAIN_CLASSIFICATIONS: "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"

If they don't exist, they will be downloaded by the script.

This script creates JSON files with the following format:
{
    "train": [list of CATH superfamily codes],
    "validation": [list of CATH superfamily codes],
    "test": [list of CATH superfamily codes]
}
"""

import json
import os
import pandas as pd
import urllib.request

# Define file paths
ESM_CATH_SPLIT_URL = "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json"
ESM_CATH_SPLIT_PATH = "data/val_test/esmif_splits.json"

DOMAIN_CLASSIFICATIONS_URL = "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
DOMAIN_CLASSIFICATIONS_PATH = "../data/cath/cath-domain-list.txt"  # Classification of domains

def download_external_files():
    if not os.path.exists(DOMAIN_CLASSIFICATIONS_PATH):
        print(f"Downloading domain classifications from {DOMAIN_CLASSIFICATIONS_URL}")
        os.makedirs(os.path.dirname(DOMAIN_CLASSIFICATIONS_PATH), exist_ok=True)
        urllib.request.urlretrieve(DOMAIN_CLASSIFICATIONS_URL, DOMAIN_CLASSIFICATIONS_PATH)
        assert os.path.exists(DOMAIN_CLASSIFICATIONS_PATH), "Failed to download domain classifications"

    if not os.path.exists(ESM_CATH_SPLIT_PATH):
        print(f"Downloading ESM CATH splits from {ESM_CATH_SPLIT_URL}")
        os.makedirs(os.path.dirname(ESM_CATH_SPLIT_PATH), exist_ok=True)
        urllib.request.urlretrieve(ESM_CATH_SPLIT_URL, ESM_CATH_SPLIT_PATH)
        assert os.path.exists(ESM_CATH_SPLIT_PATH), "Failed to download ESM CATH splits"

    print("Required external files are present.")

CLASSIFICATION_COLS = [
    "domain_name",
    "class",
    "architecture",
    "topology",
    "superfamily",
    "S35_cluster",
    "S60_cluster",
    "S95_cluster",
    "S100_cluster",
    "S100_index",
    "length",
    "resolution",
]

def domain_classification():
    """Parse the CATH domain classification file."""
    records = []
    with open(DOMAIN_CLASSIFICATIONS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                row = {col: v for col, v in zip(CLASSIFICATION_COLS, line.split())}
                records.append(row)

    df = pd.DataFrame.from_records(records)
    df["chain_id"] = df["domain_name"].str[:-2]
    df["CAT"] = df["class"] + "." + df["architecture"] + "." + df["topology"]
    df["CATH"] = df["CAT"] + "." + df["superfamily"]
    df["S35"] = df["CATH"] + "." + df["S35_cluster"]
    df["S60"] = df["S35"] + "." + df["S60_cluster"]
    return df

def chain_classification():
    """Aggregate domain classifications to chain-level classifications."""
    cath_chain_df_path = '../data/cath/cath_chain_topology_class.pkl'
    if os.path.exists(cath_chain_df_path):
        print(f"Loading chain classification from {cath_chain_df_path}")
        chain_df = pd.read_pickle(cath_chain_df_path)
    else:
        print(f"Creating chain classification df at {cath_chain_df_path}")
        dom_df = domain_classification()
        dom_df["has_class_4"] = dom_df["class"] == '4'
        dom_df["has_class_6"] = dom_df["class"] == '6'
        chain_df = dom_df.groupby("chain_id").agg(
            {
                'CAT': lambda x: tuple(x),
                'CATH': lambda x: tuple(x),
                'S35': lambda x: tuple(x),
                'S60': lambda x: tuple(x),
                'has_class_4': 'any',
                'has_class_6': 'any',
                'domain_name': "count",
            }
        ).reset_index().rename({"domain_name": "n_domains"}, axis=1)
        chain_df["n_CAT"] = chain_df["CAT"].apply(len)
        chain_df["n_CATH"] = chain_df["CATH"].apply(len)
        chain_df["n_S35"] = chain_df["S35"].apply(len)
        chain_df["n_S60"] = chain_df["S60"].apply(len)
        chain_df.to_pickle(cath_chain_df_path)

    chain_df["pdb_id"] = chain_df["chain_id"].apply(lambda x: x[:-1])
    chain_df["chain_code"] = chain_df["chain_id"].apply(lambda x: x[-1])
    chain_df["S60_comb"] = chain_df["S60"].apply(lambda x: tuple(sorted(x)))
    return chain_df

def make_splits_df(splits_file):
    chain_df = chain_classification()
    with open(splits_file, "r") as f:
        splits = json.load(f)
    assert len(splits.keys()) == 3, "Can't handle non-overlapping splits"

    chain2split = {c: s for s, chains in splits.items() for c in chains}
    chain_df["split"] = chain_df["chain_id"].apply(lambda x: chain2split.get(x, ""))
    return chain_df[chain_df["split"] != ""].reset_index(drop=True).copy()

def flatten_tuples(tuple_list):
    return sorted(list(set([item for sublist in tuple_list for item in sublist])))

def save_topology_superfamily_splits(chain_df, save_dir="data/val_test"):
    os.makedirs(save_dir, exist_ok=True)
    # Save the CAT and CATH codes associated with each split as JSON
    cath_splits = {}
    cat_splits = {}
    for split in ["train", "validation", "test"]:
        split_df = chain_df[chain_df["split"] == split]
        cath_splits[split] = flatten_tuples(split_df["CATH"].tolist())
        cat_splits[split] = flatten_tuples(split_df["CAT"].tolist())
    # Check that there are no overlaps between splits
    # Check CATH superfamily overlaps
    train_val_cath_overlap = len(set(cath_splits["train"]).intersection(set(cath_splits["validation"])))
    train_test_cath_overlap = len(set(cath_splits["train"]).intersection(set(cath_splits["test"])))
    val_test_cath_overlap = len(set(cath_splits["validation"]).intersection(set(cath_splits["test"])))
    print(f"CATH overlaps:")
    print(f"  train-val: {train_val_cath_overlap}")
    print(f"  train-test: {train_test_cath_overlap}")
    print(f"  val-test: {val_test_cath_overlap}")
    assert train_test_cath_overlap == 0, "Train and test CATH superfamilies must not overlap"
    assert val_test_cath_overlap == 0, "Val and test CATH superfamilies must not overlap"

    # Check CAT topology overlaps 
    train_val_cat_overlap = len(set(cat_splits["train"]).intersection(set(cat_splits["validation"])))
    train_test_cat_overlap = len(set(cat_splits["train"]).intersection(set(cat_splits["test"])))
    val_test_cat_overlap = len(set(cat_splits["validation"]).intersection(set(cat_splits["test"])))
    print(f"\nCAT topology overlaps:")
    print(f"  train-val: {train_val_cat_overlap}")
    print(f"  train-test: {train_test_cat_overlap}")
    print(f"  val-test: {val_test_cat_overlap}")
    assert train_test_cat_overlap == 0, "Train and test topologies must not overlap"
    assert val_test_cat_overlap == 0, "Val and test topologies must not overlap"

    with open(f"{save_dir}/superfamily_splits.json", "w") as f:
        json.dump(cath_splits, f, indent=4)
        print(f"Saved CATH superfamily splits to {save_dir}/superfamily_splits.json")
    with open(f"{save_dir}/topology_splits.json", "w") as f:
        json.dump(cat_splits, f, indent=4)
        print(f"Saved CAT topology splits to {save_dir}/topology_splits.json")

def make_cath_topology_split_json():
    print("Creating CATH topology and superfamily split JSON files")
    download_external_files()
    chain_df = make_splits_df(splits_file=ESM_CATH_SPLIT_PATH)
    save_topology_superfamily_splits(chain_df)

if __name__ == "__main__":
    make_cath_topology_split_json()

