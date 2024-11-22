"""
To build the json from scratch the following files are needed:
ESM_CATH_SPLIT = "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json"
DOMAIN_CLASSIFICATIONS = "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"



Creates a json file with the following format:
{
    "train": [list of CATH superfamily codes],
    "validation": [list of CATH superfamily codes],
    "test": [list of CATH superfamily codes]
}

As an intermediate step we need to create a dataframe
which maps the PDB IDs to CATH superfamily codes:

N.B. Domain splits are based on CATH classification in cath-domain-list.txt
CATH List File (CLF) Format 2.0
-------------------------------
This file format has an entry for each structural entry in CATH.
Column 1:  CATH domain name (seven characters)
Column 2:  Class number
Column 3:  Architecture number
Column 4:  Topology number
Column 5:  Homologous superfamily number
Column 6:  S35 sequence cluster number
Column 7:  S60 sequence cluster number
Column 8:  S95 sequence cluster number
Column 9:  S100 sequence cluster number
Column 10: S100 sequence count number
Column 11: Domain length
Column 12: Structure resolution (Angstroms)
           (999.000 for NMR structures and 1000.000 for obsolete PDB entries)

"""

import json
import os
import pandas as pd



ESM_CATH_SPLIT_URL = "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json"
ESM_CATH_SPLIT_PATH = "data/val_test/esmif_splits.json"
DOMAIN_CLASSIFICATIONS_URL = "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
DOMAIN_CLASSIFICATIONS_PATH = "../data/cath/cath-domain-list.txt"  # classification of domains

def download_external_files():
    if not os.path.exists(DOMAIN_CLASSIFICATIONS_PATH):
        os.makedirs(os.path.dirname(DOMAIN_CLASSIFICATIONS_PATH), exist_ok=True)
        os.system(f"wget {DOMAIN_CLASSIFICATIONS_URL}")
        assert os.path.exists(DOMAIN_CLASSIFICATIONS_PATH), "Failed to download domain classifications"
    if not os.path.exists(ESM_CATH_SPLIT_PATH):
        os.makedirs(os.path.dirname(ESM_CATH_SPLIT_PATH), exist_ok=True)
        os.system(f"wget {ESM_CATH_SPLIT_URL}")
        assert os.path.exists(ESM_CATH_SPLIT_PATH), "Failed to download ESMIF splits"
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
    """See 'how does the numbering in the CATH classification work?' on website:

    http://cathdb.info/wiki/doku/?id=faq

    and the README file for domain-list
    """
    records = []
    with open(DOMAIN_CLASSIFICATIONS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] != "#":
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
    cath_chain_df_path = '../data/cath/cath_chain_topology_class.pickle'
    if os.path.exists(cath_chain_df_path):
        chain_df = pd.read_pickle(cath_chain_df_path)

    else:
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
        chain_df["n_CAT"] = chain_df["CAT"].apply(lambda x: len(x))
        chain_df["n_CATH"] = chain_df["CATH"].apply(lambda x: len(x))
        chain_df["n_S35"] = chain_df["S35"].apply(lambda x: len(x))
        chain_df["n_S60"] = chain_df["S60"].apply(lambda x: len(x))
        chain_df.to_pickle(cath_chain_df_path)

    chain_df["pdb_id"] = chain_df["chain_id"].apply(lambda x: x[:-1])
    chain_df["chain_code"] = chain_df["chain_id"].apply(lambda x: x[-1])
    # note storing s60 as sorted in GENERAL is risky because it breaks 
    # identification of individual domains with clusters.
    chain_df["S60_comb"] = chain_df["S60"].apply(lambda x: tuple(sorted(x)))
    return chain_df


def make_splits_df(splits_file):
    chain_df = chain_classification()
    with open(splits_file, "r") as f:
        splits = json.load(f)
    assert len(splits.keys()) == 3, "Can't handle non-overlapping splits"

    chain2split = {c: s for s, chains in splits.items() for c in chains}
    chain_df["split"] = chain_df["chain_id"].apply(lambda x: chain2split.get(x, ""))
    return chain_df[chain_df["split"]!=""].reset_index(drop=True).copy()


def flatten_tuples(tuple_list):
    return sorted(list(set([item for sublist in tuple_list for item in sublist])))

def save_topology_superfamily_splits(chain_df, save_dir="data/val_test"):
    os.makedirs(save_dir, exist_ok=True)
    # save the CAT and CATH codes associated with each split as json
    cath_splits = {}
    cat_splits = {}
    for split in ["train", "validation", "test"]:
        split_df = chain_df[chain_df["split"] == split]
        cath_splits[split] = flatten_tuples(split_df["CATH"].tolist())
        cat_splits[split] = flatten_tuples(split_df["CAT"].tolist())
    # check that there are no overlaps between splits
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
        json.dump(cath_splits, f)
        print(f"Saved CATH superfamily splits to {save_dir}/superfamily_splits.json")
    with open(f"{save_dir}/topology_splits.json", "w") as f:
        json.dump(cat_splits, f)
        print(f"Saved CAT topology splits to {save_dir}/topology_splits.json")

def make_cath_topology_split_json():
    download_external_files()
    chain_df = make_splits_df(splits_file=ESM_CATH_SPLIT_PATH)
    print(chain_df.head())
    save_topology_superfamily_splits(chain_df)

if __name__ == "__main__":
    make_cath_topology_split_json()

