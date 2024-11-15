import json
import numpy as np
import os
import glob

import pandas as pd

"""
Ultimate goal is to create train / val / test splits for:

TED
FunFams
Foldseek

this file creates a json for each of these datasets mapping fam_ids
to train, validation and test sets

strategy is to first take all test IDs and remove them from 
training and validation sets.
"""


def make_ted_json_from_cath_tsv(ted_cath_asignment_path: str):
    # cluster_reps = set()
    # ted_ids = set()
    # up_ids = set()
    output_path = ted_cath_asignment_path.replace(".tsv", ".json")
    if not os.path.exists(output_path):
        cath_assignments = {}
        counter = 0
        with open(ted_cath_asignment_path, "r") as f:
            for line in f:
                split_line = line.strip().split("\t")
                cath_code = split_line[13]
                ted_id = split_line[0]
                if cath_code not in cath_assignments:
                    cath_assignments[cath_code] = []
                cath_assignments[cath_code].append(ted_id)
                counter += 1
                if counter % 1000000 == 0:
                    print(f"Processed {counter} lines")
        with open(output_path, "w") as f:
            json.dump(cath_assignments, f, indent=4)
    else:
        print(f"File {output_path} already exists")

def report_ted_json_stats(ted_json: dict):
    print(f"Number of cath codes: {len(ted_json)}")
    print(f"Number of sequences: {sum([len(v) for v in ted_json.values()])}")
    print(f"Mean len per cath code: {np.mean([len(v) for v in ted_json.values()])}")
    print(f"Median len per cath code: {np.median([len(v) for v in ted_json.values()])}")
    print(f"Max len per cath code: {max([len(v) for v in ted_json.values()])}")
    print(f"Min len per cath code: {min([len(v) for v in ted_json.values()])}")

def reformat_to_up_ids(ted_json: dict):
    return {k: [upid.split("-")[1] for upid in v] for k, v in ted_json.items()}


def make_accessions_split(ted_json: dict, splits: dict):
    """
    Which uniprot accessions are assigned
    to train / validation / test based on the 
    topology splits.
    """
    split_to_accessions = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for cath_code, ted_ids in ted_json.items():
        up_ids = [t.split("-")[1] for t in ted_ids]
        if cath_code in splits["test"]:
            split_to_accessions["test"].update(up_ids)
        elif cath_code in splits["validation"]:
            split_to_accessions["validation"].update(up_ids)
        else:
            split_to_accessions["train"].update(up_ids)
    split_to_accessions["validation"] = split_to_accessions["validation"] - split_to_accessions["test"]
    split_to_accessions["train"] = split_to_accessions["train"] - split_to_accessions["test"] - split_to_accessions["validation"]
    print(f"Finished making accessions split")
    return split_to_accessions

def make_foldseek_json(accessions_split: dict, fseek_parquet_dir: str):
    """
    creates the json file which assigns each
    family (foldseek afdb cluster) to train/val/test
    if a single ID is in test 
    """
    fseek_splits = {
        "train": [],
        "validation": [],
        "test": [],
    }
    parq_paths = glob.glob(f"{fseek_parquet_dir}/*.parquet")
    overlaps = {}
    for parq_path in parq_paths:
        print(f"Processing {parq_path}")
        df = pd.read_parquet(parq_path)
        for i, row in df.iterrows():
            fam_id = row["fam_id"]
            accessions = set(row["accessions"])
            if accessions & split_to_accessions["test"]:
                fseek_splits["test"].append(fam_id)
            elif accessions & split_to_accessions["validation"]:
                fseek_splits["validation"].append(fam_id)
            else:
                fseek_splits["train"].append(fam_id)
            if i % 100 == 0:
                print(f"Processed {i} families")
    return fseek_splits

def report_foldseek_json_stats(fseek_json: dict):
    for split, fams in fseek_json.items():
        print(f"{split}: {len(fams)}")


if __name__ == "__main__":
    splits = json.load(open("data/val_test/superfamily_splits.json", "r"))
    ted_tsv_path = "../data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv"
    ted_json_path = ted_tsv_path.replace(".tsv", ".json")
    make_ted_json_from_cath_tsv(ted_tsv_path)
    ted_json = json.load(open(ted_json_path, "r"))
    report_ted_json_stats(ted_json)
    split_to_accessions = make_accessions_split(ted_json, splits)
    fseek_splits = make_foldseek_json(
        accessions_split=split_to_accessions,
        fseek_parquet_dir="../data/foldseek_af50",
    )
    report_foldseek_json_stats(fseek_splits)
    json.dump(fseek_splits, open("data/val_test/foldseek_cath_topology_splits.json", "w"), indent=4)



