import json
import numpy as np
import os


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
                ted_id_clust_rep, ted_id, cath_code, assignment_type = line.strip().split("\t")
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


def make_foldseek_json(splits, ted_json, foldseek_json_path):
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
    fseek_members
    foldseek = json.load(open(foldseek_json_path, "r"))
    for fam, entries in foldseek.items():

    


if __name__ == "__main__":
    splits = json.load(open("data/val_test/superfamily_splits.json", "r"))
    make_ted_json_from_cath_tsv("../data/ted/ted_324m_seq_clustering.cathlabels.tsv")
    ted_json = json.load(open("../data/ted/ted_324m_seq_clustering.cathlabels.json", "r"))
    report_ted_json_stats(ted_json)
    make_foldseek_json(
        splits=splits,
        ted_json=ted_json,
        foldseek_json_path="",
    )



