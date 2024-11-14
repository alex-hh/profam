"""
Load / Process CATH metadata.
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
Numbers within each column are relative to the higher categories.
e.g. topology number 1 is relative to the C and A codes,
S35 sequence cluster is relative to CATH codes.
We *might* want to check that no chains in this set contain domains outside of the
nonredundant set on which redundancy control is based.
If it's the same pipeline as Ingraham, we take CHAINS to which non-redundant domains
are annotated, then annotate these chains with CATH nodes. This produces a dataset of
CATH nodes. The split procedure occurs on CATH nodes and is random 80/10/10.
Since each chain can contain multiple CAT codes, we first removed any redundant entries 
from train and then from validation. What does this mean?
Finally, we removed any chains from the test set that had CAT overlap with train and 
removed chains from the validation set with CAT overlap to train or test. 
Facebook description is clearer perhaps:
As each chain may be classified with more than one topology codes, we further removed 
chains with topology codes spanning different splits, so that there is no overlap in 
topology codes between train, validation, and test. This results in 16,153 chains in 
the train split, 1457 chains in the validation split, and 1797 chains in the test split.
https://github.com/jingraham/neurips19-graph-protein-design/blob/master/data/build_chain_dataset.py
"""
from dataclasses import dataclass
import json
import os
import pandas as pd



ESM_CATH_SPLIT = "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json"
DOMAIN_BOUNDARIES = "data/cath/cath-domain-boundaries.txt"   # annotations of domains to chains
DOMAIN_CLASSIFICATIONS = "../data/cath/cath-domain-list.txt"  # classification of domains
CATH_PDB_URL = "http://www.cathdb.info/version/v4_3_0/api/rest/id/{}.pdb"


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
    with open(DOMAIN_CLASSIFICATIONS, "r") as f:
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
    cath_chain_df_path = 'data/cath_chain_topology_class.pickle'
    if os.path.exists(cath_chain_df_path):
        chain_df = pd.read_pickle(cath_chain_df_path)
        if 'has_class_4' not in chain_df.columns:
            os.remove(cath_chain_df_path)
            return chain_classification()
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


def esmif_chain_splits_df(split_json_path="data/val_test/esmif_splits.json"):
    """Load a dataframe containing chain ids, topology information and split information
    for chains within the ESM-IF CATH splits.
    """
    return make_splits_df(split_json_path)

def flatten_tuples(tuple_list):
    return sorted(list(set([item for sublist in tuple_list for item in sublist])))

def save_topology_superfamily_splits(chain_df):
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
    with open("data/val_test/superfamily_splits.json", "w") as f:
        json.dump(cath_splits, f)
    with open("data/val_test/topology_splits.json", "w") as f:
        json.dump(cat_splits, f)

if __name__ == "__main__":
    chain_df = esmif_chain_splits_df()
    print(chain_df.head())
    save_topology_superfamily_splits(chain_df)

