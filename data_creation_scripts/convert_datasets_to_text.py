import argparse
import glob
import os
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np

"""
Created by Jude Wells 2025-05-05

Iterates over the parquet files and converts them into plain text format
which can be used with the bionemo textMemMap dataset

As I understand it we would have files named like so (S50 refers to non-reundant at 50% SI):
TED_S50_train_00_000.sequences / TED_S50_val_00_000.sequences / TED_S50_test_00_000.sequences  and also TED_S50_train_00_000.coords / TED_S50_val_00_000.coords / TED_S50_test_00_000.coords
Each of those files can contain entries for multiple families but where possible we try to avoid families being split across multiple files (except for in the case of very large families)
the .sequences files would look like a fasta where we store the aligned sequence (aligned with the rest of the family):
>Q5VSL9.TED.001/0.3.11.33.2.51.100.101.67
--GCAMM-MMMM--FEQ-
>Q5VSL9.TED.002/0.3.11.33.2.51.100.101.68
--GCAMM-MMMM--FEQ-
corresponding coordinate files
>Q5VSL9.TED.001/0.3.11.33.2.51.100.101.67
N:
1.001,2.002,3.002... all the way up to 3 * chain_length
CA:
1.001,2.002,3.002... all the way up to 3 * chain_length
C:
1.001,2.002,3.002... all the way up to 3 * chain_length
O:
1.001,2.002,3.002... all the way up to 3 * chain_length
plddts:
65.4, 65.4, 65.4... all the way up to chain_length
>Q5VSL9.TED.002/0.3.11.33.2.51.100.101.68
N:
1.001,2.002,3.002... all the way up to 3 * chain_length
C:
1.001,2.002,3.002... all the way up to 3 * chain_length
CA:
1.001,2.002,3.002... all the way up to 3 * chain_length
plddts:
65.4, 65.4, 65.4... all the way up to chain_length
Then we also have mappings files:
TED_S50_1.mapping, TED_S50_2.mapping...
These will contain family identifier followed by list of filenames and indexes
>1.10.11.12
TED_S50_train_00_000:0,1,2,3,4,5,
TED_S50_train_00_001:,2,3,4 (assuming this is a large family split over multiple files)
>1.1.3.10
TED_S50_train_00_000:99,100,101
...
For single sequence datasets which I think you refer to as "samples datasets" we don't need the mappings file as we can just sample files and indexes directly.
We will aim to keep the file size around 100MB
"""

datasets_to_convert = [
    {
        "name": "TEDS100",
        "parquet_dir": "../data/ted/s100_parquets/train_test_split_v2",
        "output_dir": "../data/ted/s100_text/train_test_split_v2"
    },
    {
        "name": "FunFamsS50",
        "parquet_dir": "../data/funfams/s50_parquets/train_test_split_v2"
    },
    {
        "name": "foldseek_s50_seq_only",
        "parquet_dir": "../data/foldseek/foldseek_s50_seq_only/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_s50_seq_only_text/train_test_split_v2"
    },
    {
        "name": "foldseek_s50_struct",
        "parquet_dir": "../data/foldseek/foldseek_s50_struct/train_val_test_split_v2/",
        "output_dir": "../data/foldseek/foldseek_s50_struct_text/train_test_split_v2"
    },
    {
        "name": "foldseek_s100_struct",
        "parquet_dir": "../data/foldseek/foldseek_s100_struct/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_s100_struct_text/train_test_split_v2"
    },
    {
        "name": "foldseek_reps_single",
        "parquet_dir": "../data/foldseek/foldseek_reps_single/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_reps_single_text/train_test_split_v2"
    },
    # {
    #     "name": "afdb_s50_single",
    #     "parquet_pattern": "../data/afdb_s50_single/*.parquet",
    #     "filtered_parquet_dir": "../data/afdb/afdb_s50_single_parquets/train_test_split_v2",
    #     "parquet_dir": "../data/afdb/afdb_s50_single_parquets/train_test_split_v2",
    # },
    {
        "name": "ted_s50_hq",
        "parquet_dir": "../data/ted/s50_parquets/train_test_split_v2"
    },

    {
        "name": "FunFamsS100",
        "parquet_dir": "../data/funfams/s100_noali_parquets/train_test_split_v2",
        "output_dir": "../data/funfams/s100_noali_text/train_test_split_v2"
    },
    {
        "name": "Foldseek_s100",
        "parquet_dir": "../data/foldseek/foldseek_s100_raw/train_test_split_v2",
        "output_dir": "../data/foldseek/foldseek_s100_raw_text/train_test_split_v2"
    },

    {
        "name": "Pfam",
        "parquet_dir": "../data/pfam/train_test_split_parquets/train_test_split_v2",
    },
    {
        "name": "OpenFold_clustered",
        "parquet_dir": "../data/openfold/uniclust30_clustered_shuffled_final/train_test_split_v2",
        "output_dir": "../data/openfold/uniclust30_clustered_shuffled_final_text/train_test_split_v2"
    },
    {
        "name": "Uniref90",
        "parquet_dir": "../data/uniref/uniref90_parquets_shuffled/train_test_split_v2",
    },
    # {
    #     "name": "afdb_s50_single",
    #     "parquet_dir": "../data/afdb/afdb_s50_single_parquets/train_test_split_v2",
    # },
]

def _cluster_level_from_col(col: str) -> float:
    """Extract the clustering level as a float from a column name like 'cluster_ids_0_95'."""
    level_str = col.replace("cluster_ids_", "")
    return float(level_str.replace("_", "."))


def _col_from_cluster_level(level: float) -> str:
    """Return column name corresponding to clustering level."""
    s = ("%.3f" % level).rstrip("0").rstrip(".")  # ensure e.g. 0.2 not 0.200
    s = s.replace(".", "_")
    return f"cluster_ids_{s}"


def _generate_groups(row: pd.Series, seq_count: int, min_lvl: float, cluster_cols_map: Dict[float, str]):
    """Return list of lists of sequence indices that satisfy min_lvl criteria.

    A group must contain at least 2 sequences (non-singleton).
    We group by min_lvl column to ensure all sequences in a group have at least min_lvl sequence identity.
    """
    indices_all = list(range(seq_count))
    def _cluster_ids(level: float):
        col = cluster_cols_map[level]
        return row[col]

    groups: List[List[int]] = []
    cluster_ids = _cluster_ids(min_lvl)
    unique_ids = np.unique(cluster_ids)
    for cid in unique_ids:
        inds = [i for i, c in enumerate(cluster_ids) if c == cid]
        if len(inds) > 1:
            # Add safety check
            if max(inds) >= seq_count:
                print(f"Warning: Found invalid index {max(inds)} in cluster {cid} for level {min_lvl}. Max valid index is {seq_count-1}")
                continue
            groups.append(inds)
    return groups


def write_rows_to_text(df: pd.DataFrame, output_dir: str):
    """Write sequences, coords (if present) and mapping files for a given dataframe.

    Args:
        df: DataFrame with rows containing sequence and optional structure information.
        output_dir: Directory where .sequences/.coords/.mapping files will be written.
    """
    basename_prefix = os.path.splitext(os.path.basename(output_dir))[0]

    
    seq_col = "msta_seqs" if "msta_seqs" in df.columns else "sequences"
    has_coords = all(col in df.columns for col in ["N", "CA", "C", "O"]) and "plddts" in df.columns

    
    cluster_cols = [c for c in df.columns if c.startswith("cluster_ids_")]
    levels = sorted([_cluster_level_from_col(c) for c in cluster_cols])
    cluster_cols_map = {
        l: _col_from_cluster_level(l) for l in levels
    }

    # file handles
    seq_file_path = f"{output_dir}.sequences"
    seq_fh = open(seq_file_path, "w")
    coords_fh = None
    if has_coords:
        coords_file_path = f"{output_dir}.coords"
        coords_fh = open(coords_file_path, "w")

    mapping_dict: Dict[str, List[Tuple[str, str]]] = {f"min{level}": [] for level in levels}
    mapping_dict["min0"] = []

    seq_global_index = 0  # across all rows in this file

    for row_idx, row in df.iterrows():
        
        cluster_id_maps = {}
        for level in levels:
            col = cluster_cols_map[level]
            unique_ids = sorted(list(set(row[col])))
            cluster_id_maps[level] = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
        sequences = row[seq_col]
        accessions = row["accessions"]
        assert len(sequences) == len(accessions)
        seq_count = len(sequences)
        
        
        indices_for_row = []
        for i in range(seq_count):
            accession = accessions[i]
            seq = sequences[i]

            cluster_indices = []
            for level in levels:
                old_cluster_id = row[cluster_cols_map[level]][i]
                new_cluster_id = cluster_id_maps[level][old_cluster_id]
                cluster_indices.append(str(new_cluster_id))
            extended_id = f"{accession}/{'.'.join(cluster_indices)}"
            
            seq_fh.write(f">{extended_id}\n{seq}\n")
            if has_coords:
                coords_fh.write(f">{extended_id}\n")
                for atom in ["N", "CA", "C", "O"]:
                    coords_fh.write(f"{atom}:\n")
                    coords = np.asarray(row[atom][i]).ravel()
                    coords_str = ",".join([f"{x:.3f}" for x in coords])
                    coords_fh.write(f"{coords_str}\n")
                coords_fh.write("plddts:\n")
                plddts_arr = np.asarray(row["plddts"][i]).ravel()
                plddt_str = ",".join([f"{x:.1f}" for x in plddts_arr])
                coords_fh.write(f"{plddt_str}\n")
            indices_for_row.append(seq_global_index)
            seq_global_index += 1

        if seq_count >= 1:
            inds_str = ",".join(map(str, indices_for_row))
            seq_file_name = os.path.basename(seq_file_path)
            mapping_line = f"{seq_file_name}:{inds_str}"
            mapping_dict["min0"].append((str(row["fam_id"]), mapping_line))
        else:
            # This handles:
            # 1. No "fam_id" column in the row.
            # 2. "fam_id" is present but is NaN for this row.
            # 3. "fam_id" is present, but seq_count is 1 (singleton family or single sequence row).
            # In these cases, map each sequence individually using its accession as the key.
            for i, global_idx in enumerate(indices_for_row):
                seq_file_name = os.path.basename(seq_file_path)
                mapping_line = f"{seq_file_name}:{global_idx}"
                individual_key = accessions[i] # Use accession as the "family" key
                mapping_dict["min0"].append((individual_key, mapping_line))

        for min_lvl in levels:
            groups = _generate_groups(row, seq_count, min_lvl, cluster_cols_map)
            for g_ix, inds in enumerate(groups):
                # Add safety check
                if max(inds) >= len(indices_for_row):
                    print(f"Warning: Skipping invalid group in row {row_idx} for level {min_lvl}")
                    print(f"Group indices: {inds}")
                    print(f"Available indices: {len(indices_for_row)}")
                    continue
                global_inds = [indices_for_row[i] for i in inds]
                inds_str = ",".join(map(str, global_inds))
                seq_file_name = os.path.basename(seq_file_path)
                mapping_line = f"{seq_file_name}:{inds_str}"
                mapping_dict[f"min{min_lvl}"].append((f'{row["fam_id"]}.{g_ix}', mapping_line))

    seq_fh.close()
    if coords_fh is not None:
        coords_fh.close()

    for min_lvl, entries in mapping_dict.items():
        if not entries:
            continue  # nothing to write
        mapping_file_path = os.path.join(os.path.dirname(seq_file_path), f"{basename_prefix}_{min_lvl}.mapping")
        with open(mapping_file_path, "w") as mf:
            current_fam = None
            for  i, (fam_id, mapping_line) in enumerate(entries):
                if fam_id != current_fam:
                    if i != 0:
                        mf.write("\n")
                    mf.write(f">{fam_id}\n")
                    current_fam = fam_id
                mf.write(f"{mapping_line}")


def convert_parquet_to_text(parquet_file: str, output_dir: str):
    """Convert a single parquet file into sequence/coords text files and mapping files.

    Args:
        parquet_file: Path to parquet file
        output_dir: Directory to write outputs (without file extension). Will create if needed.
    """
    base_name = os.path.splitext(os.path.basename(parquet_file))[0]
    file_output_prefix = os.path.join(output_dir, base_name)

    # Check if output files already exist
    sequences_file_path = f"{file_output_prefix}.sequences"
    # Assuming min0 is the primary mapping file to check for completion
    mapping_file_path = f"{file_output_prefix}_min0.mapping"

    if os.path.exists(sequences_file_path) and os.path.exists(mapping_file_path):
        print(f"Output files for {parquet_file} already exist. Skipping.")
        return

    os.makedirs(os.path.dirname(file_output_prefix), exist_ok=True)

    print(f"Reading parquet {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows from parquet.")

    write_rows_to_text(df, file_output_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=-1)
    args = parser.parse_args()
    if args.task_index != -1:
        datasets_to_convert = [datasets_to_convert[args.task_index]]

    for dataset in datasets_to_convert:
        print(f"Converting {dataset['name']} to text")
        
        for subdir in ["train_filtered", "val_filtered", "test_filtered", "train", "val", "test"]:
            parquet_dir = os.path.join(dataset['parquet_dir'], subdir)
            for parquet_file in glob.glob(os.path.join(parquet_dir, "*.parquet")):
                if "output_dir" in dataset:
                    output_dir = os.path.join(dataset["output_dir"], subdir)
                elif "parquets" in parquet_dir:
                    output_dir = os.path.join(parquet_dir.replace("parquets", "text"))
                else:
                    raise ValueError(f"No output directory specified for {dataset['name']}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                convert_parquet_to_text(parquet_file, output_dir)