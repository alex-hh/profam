import json
import numpy as np
import os
import glob
import argparse
import pandas as pd
from data_creation_scripts.val_test_split.make_cath_splits_json import make_cath_topology_split_json

"""
Uses the CATH superfamily splits JSON to create the FoldSeek
train/val/test splits JSON: using the TED assignments to map
UniProt accessions to superfamily IDs.
"""

# requried external file for assigning uniprot IDs to CATH:
TED_CATH_ASSIGNMENTS_URL = "https://zenodo.org/records/13908086/files/ted_365m.domain_summary.cath.globularity.taxid.tsv.gz?download=1"

def make_ted_json_from_cath_tsv(ted_cath_assignment_path: str):
    if not os.path.exists(ted_cath_assignment_path):
        print(f"TED CATH assignment file not found at {ted_cath_assignment_path}")
        print("Downloading TED CATH assignments (20GB) (this may take a while)...")
        urllib.request.urlretrieve(TED_CATH_ASSIGNMENTS_URL, ted_cath_assignment_path)
        assert os.path.exists(ted_cath_assignment_path), "Failed to download TED CATH assignments"

    print(f"Creating TED JSON from {ted_cath_assignment_path}")
    output_path = ted_cath_assignment_path.replace(".tsv", ".json")
    if not os.path.exists(output_path):
        cath_assignments = {}
        counter = 0
        with open(ted_cath_assignment_path, "r") as f:
            for line in f:
                split_line = line.strip().split("\t")
                ted_id = split_line[0]
                cath_code = split_line[13]
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
    print(f"Mean sequences per CATH code: {np.mean([len(v) for v in ted_json.values()]):.2f}")
    print(f"Median sequences per CATH code: {np.median([len(v) for v in ted_json.values()]):.2f}")
    print(f"Max sequences per CATH code: {max([len(v) for v in ted_json.values()])}")
    print(f"Min sequences per CATH code: {min([len(v) for v in ted_json.values()])}")

def make_split_to_accessions(ted_json: dict, splits: dict, split_to_accession_savepath: str):
    """
    Determines which UniProt accessions are assigned
    to train/validation/test based on the topology splits.
    """
    if os.path.exists(split_to_accession_savepath):
        print(f"Loading existing accessions split from {split_to_accession_savepath}")
        with open(split_to_accession_savepath, "r") as f:
            return json.load(f)
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
            print(f"Processed {i}/{total_cath_codes} CATH codes")
    split_to_accessions["validation"] -= split_to_accessions["test"]
    split_to_accessions["train"] -= split_to_accessions["test"] | split_to_accessions["validation"]
    split_to_accessions = {k: sorted(list(v)) for k, v in split_to_accessions.items()}

    print(f"Writing accessions split to {split_to_accession_savepath}")
    with open(split_to_accession_savepath, "w") as f:
        json.dump(split_to_accessions, f, indent=4)
    return split_to_accessions

def make_foldseek_json(split_to_accessions: dict, fseek_parquet_dir: str):
    """
    Creates the JSON file which assigns each
    family (FoldSeek AFDB cluster) to train/val/test,
    if a single ID is in test.
    """
    fseek_splits = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_to_accessions = {k: set(v) for k, v in split_to_accessions.items()}
    parq_paths = sorted(glob.glob(os.path.join(fseek_parquet_dir, "*.parquet")))
    
    for parq_path in parq_paths:
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
        print(f"{split}: {len(fams)} families")

def create_foldseek_split_json(
    foldseek_split_json_path="data/val_test/foldseek_cath_topology_splits.json",
    superfamily_splits_json_path="data/val_test/superfamily_splits.json",
    ted_tsv_path="../data/ted/ted_365m.domain_summary.cath.globularity.taxid.tsv",
    split_to_accession_savepath="../data/ted/ted_esmif_accessions_split.json",
    fseek_parquet_dir="../data/foldseek_af50",
):
    print(f"Creating FoldSeek split JSON from {superfamily_splits_json_path}")
    if not os.path.exists(superfamily_splits_json_path):
        print(f"Superfamily splits JSON not found at {superfamily_splits_json_path}, creating it.")
        make_cath_topology_split_json()
    splits = json.load(open(superfamily_splits_json_path, "r"))

    ted_json_path = ted_tsv_path.replace(".tsv", ".json")
    make_ted_json_from_cath_tsv(ted_tsv_path)
    ted_json = json.load(open(ted_json_path, "r"))
    report_ted_json_stats(ted_json)
    split_to_accessions = make_split_to_accessions(ted_json, splits, split_to_accession_savepath)
    fseek_splits = make_foldseek_json(
        split_to_accessions=split_to_accessions,
        fseek_parquet_dir=fseek_parquet_dir,
    )
    report_foldseek_json_stats(fseek_splits)
    os.makedirs(os.path.dirname(foldseek_split_json_path), exist_ok=True)
    with open(foldseek_split_json_path, "w") as f:
        json.dump(fseek_splits, f, indent=4)
    print(f"FoldSeek split JSON saved to {foldseek_split_json_path}")
    return fseek_splits

if __name__ == "__main__":
    create_foldseek_split_json()
