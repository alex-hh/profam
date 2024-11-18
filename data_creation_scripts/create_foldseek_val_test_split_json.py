import json
import numpy as np
import os
import glob
import argparse
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


def make_split_to_accessions(ted_json: dict, splits: dict, split_to_accession_savepath: str):
    """
    Which uniprot accessions are assigned
    to train / validation / test based on the 
    topology splits.
    """
    if os.path.exists(split_to_accession_savepath):
        print(f"Loading existing accessions split from {split_to_accession_savepath}")
        return json.load(open(split_to_accession_savepath, "r"))
    split_to_accessions = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    total_cath_codes = len(ted_json)
    for i, (cath_code, ted_ids) in enumerate(ted_json.items()):
        up_ids = [t.split("-")[1] for t in ted_ids]
        if cath_code in splits["test"]:
            split_to_accessions["test"].update(up_ids)
        elif cath_code in splits["validation"]:
            split_to_accessions["validation"].update(up_ids)
        else:
            split_to_accessions["train"].update(up_ids)
        if i % 100 == 0:
            print(f"Processed {i}/{total_cath_codes} cath codes")
    split_to_accessions["validation"] = split_to_accessions["validation"] - split_to_accessions["test"]
    split_to_accessions["train"] = split_to_accessions["train"] - split_to_accessions["test"] - split_to_accessions["validation"]
    split_to_accessions["test"] = sorted(list(split_to_accessions["test"]))
    split_to_accessions["validation"] = sorted(list(split_to_accessions["validation"]))
    split_to_accessions["train"] = sorted(list(split_to_accessions["train"]))
    print(f"Writing accessions split to {split_to_accession_savepath}")
    with open(split_to_accession_savepath, "w") as f:
        json.dump(split_to_accessions, f, indent=4)
    return split_to_accessions

def make_foldseek_json(split_to_accessions: dict, fseek_parquet_dir: str, task_index: int=0, num_tasks: int=1):
    """
    creates the json file which assigns each
    family (foldseek afdb cluster) to train/val/test
    if a single ID is in test.
    Processes a subset of parquet files based on task_index and num_tasks.
    """
    if num_tasks > 1:
        raise NotImplementedError("Multi-tasking not implemented yet")
    fseek_splits = {
        "train": [],
        "validation": [], 
        "test": [],
    }
    split_to_accessions = {k: set(v) for k, v in split_to_accessions.items()}
    parq_paths = sorted(glob.glob(f"{fseek_parquet_dir}/*.parquet"))
    
    # Calculate chunk for this task
    chunk_size = len(parq_paths) // num_tasks
    start_idx = task_index * chunk_size
    end_idx = start_idx + chunk_size if task_index < num_tasks - 1 else len(parq_paths)
    task_parq_paths = parq_paths[start_idx:end_idx]
    
    print(f"Task {task_index+1}/{num_tasks}: Processing {len(task_parq_paths)} files")
    
    for parq_path in task_parq_paths:
        print(f"Processing {parq_path}")
        df = pd.read_parquet(parq_path)
        print(f"df.shape: {df.shape}")
        for i, row in df.iterrows():
            fam_id = row["fam_id"]
            accessions = set(row["accessions"])
            if accessions & split_to_accessions["test"]:
                fseek_splits["test"].append(fam_id)
            elif accessions & split_to_accessions["validation"]:
                fseek_splits["validation"].append(fam_id)
            else:
                fseek_splits["train"].append(fam_id)
            if i % 1000 == 0:
                print(f"Processed {i} families")
    return fseek_splits

def report_foldseek_json_stats(fseek_json: dict):
    for split, fams in fseek_json.items():
        print(f"{split}: {len(fams)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=0, help="Index of current task (0-based)")
    parser.add_argument("--num_tasks", type=int, default=1, help="Total number of tasks")
    args = parser.parse_args()
    splits = json.load(open("data/val_test/superfamily_splits.json", "r"))
    ted_tsv_path = "../data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv"
    split_to_accession_savepath = "../data/ted/ted_esmif_accessions_split.json"
    ted_json_path = ted_tsv_path.replace(".tsv", ".json")
    make_ted_json_from_cath_tsv(ted_tsv_path)
    ted_json = json.load(open(ted_json_path, "r"))
    report_ted_json_stats(ted_json)
    split_to_accessions = make_split_to_accessions(ted_json, splits, split_to_accession_savepath)
    fseek_splits = make_foldseek_json(
        split_to_accessions=split_to_accessions,
        fseek_parquet_dir="../data/foldseek_af50",
    )
    report_foldseek_json_stats(fseek_splits)
    json.dump(fseek_splits, open("data/val_test/foldseek_cath_topology_splits.json", "w"), indent=4)



