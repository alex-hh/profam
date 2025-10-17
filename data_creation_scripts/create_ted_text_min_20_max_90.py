"""
Create S20-only mapping and sequences with one representative per S90 cluster inside each S20 cluster.

Behavior:
- Read TED parquet files containing arrays per row: 'sequences', 'accessions', 'cluster_ids_0_2' (S20) and 'cluster_ids_0_9' (S90).
- For each row (family), within each S20 cluster, keep at most one sequence for each S90 label (first occurrence).
- Write only the filtered sequences to a single .sequences file per parquet.
- Write a single mapping file that groups sequences by S20 clusters: <basename>_min20_max90.mapping.
- If N/CA/C/O/plddts arrays exist, also write a .coords file for the filtered sequences.
"""

import glob
import argparse
import os
from typing import Dict, Iterable, List, Tuple, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd


def _ensure_output_dir(path_prefix: str) -> None:
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)


def _col_exists(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def _remap_ids_preserve_order(ids: Iterable) -> Dict:
    """Map arbitrary ids to contiguous integers in first-seen order."""
    mapping: Dict = {}
    next_id = 0
    for x in ids:
        if x not in mapping:
            mapping[x] = next_id
            next_id += 1
    return mapping


def _filtered_indices_by_s20_s90(row: pd.Series, excluded_accessions: Optional[Set[str]] = None) -> List[int]:
    """Select indices keeping at most one representative per S90 within each S20 cluster.

    If excluded_accessions is provided, skip any index whose accession is in the set. This ensures
    that we still choose the first non-excluded representative for each S90 within each S20.
    """
    s20 = row["cluster_ids_0_2"]
    s90 = row["cluster_ids_0_9"]
    accessions = row["accessions"]
    n = len(row["sequences"]) if "sequences" in row else len(row["msta_seqs"])  # type: ignore

    seen_by_s20: Dict[object, set] = {}
    selected: List[int] = []
    for i in range(n):
        if excluded_accessions is not None and accessions[i] in excluded_accessions:
            continue
        k20 = s20[i]
        k90 = s90[i]
        if k20 not in seen_by_s20:
            seen_by_s20[k20] = set()
        if k90 in seen_by_s20[k20]:
            continue  # already have a rep of this S90 within this S20
        seen_by_s20[k20].add(k90)
        selected.append(i)
    return selected


def write_rows_to_text_s20(df: pd.DataFrame, output_prefix: str, excluded_accessions: Optional[Set[str]] = None) -> None:
    """Write filtered .sequences/.coords and a single S20 mapping file.

    The mapping file name is '<output_prefix>_min0.2.mapping' and contains one entry per S20 group,
    with indices referencing the filtered sequences written to '<output_prefix>.sequences'.
    """

    basename_prefix = os.path.splitext(os.path.basename(output_prefix))[0]

    seq_col = "sequences"
    has_coords = all(_col_exists(df, c) for c in ["N", "CA", "C", "O"]) and _col_exists(df, "plddts")

    seq_path = f"{output_prefix}.sequences"
    coords_path = f"{output_prefix}.coords" if has_coords else None

    # Lazily open files on first write to avoid creating empty outputs
    seq_fh = None
    coords_fh = None

    # Collect mapping entries as (fam_or_group_id, mapping_line)
    mapping_entries: List[Tuple[str, str]] = []

    seq_global_index = 0

    for _, row in df.iterrows():
        # Basic columns
        fam_id = str(row["fam_id"]) if _col_exists(df, "fam_id") else "NA"
        sequences = row[seq_col]
        accessions = row["accessions"]
        s20 = row["cluster_ids_0_2"]
        s90 = row["cluster_ids_0_9"]

        if not (len(sequences) == len(accessions) == len(s20) == len(s90)):
            print(f"Warning: {fam_id} has different lengths: {len(sequences)} != {len(accessions)} != {len(s20)} != {len(s90)}")
            continue

        # Choose filtered indices: one per S90 within each S20 (skipping excluded accessions)
        selected = _filtered_indices_by_s20_s90(row, excluded_accessions)
        if not selected:
            continue

        # Keep only S20 groups that have at least 2 selected members (i.e., >=2 distinct S90 reps)
        selected_by_s20: Dict[object, List[int]] = {}
        for i in selected:
            selected_by_s20.setdefault(s20[i], []).append(i)
        eligible_s20_ids = {k for k, inds in selected_by_s20.items() if len(inds) >= 2}
        filtered_selected = [i for i in selected if s20[i] in eligible_s20_ids]
        if not filtered_selected:
            continue

        # Build per-level remapping for readable headers (only S20 and S90)
        s20_map = _remap_ids_preserve_order(s20[i] for i in filtered_selected)
        s90_map = _remap_ids_preserve_order(s90[i] for i in filtered_selected)

        # Track global indices for the filtered entries of this row
        global_indices_for_row: List[int] = []
        s20_of_selected: List[object] = []

        for i in filtered_selected:
            accession = accessions[i]
            seq = sequences[i]

            s20_new = s20_map[s20[i]]
            s90_new = s90_map[s90[i]]
            extended_id = f"{accession}/{s20_new}.{s90_new}"

            if seq_fh is None:
                seq_fh = open(seq_path, "w")
                if has_coords and coords_path is not None:
                    coords_fh = open(coords_path, "w")

            seq_fh.write(f">{extended_id}\n{seq}\n")

            if coords_fh is not None:
                coords_fh.write(f">{extended_id}\n")
                for atom in ["N", "CA", "C", "O"]:
                    coords_fh.write(f"{atom}:\n")
                    coords = np.asarray(row[atom][i]).ravel()
                    coords_str = ",".join(f"{x:.3f}" for x in coords)
                    coords_fh.write(f"{coords_str}\n")
                coords_fh.write("plddts:\n")
                plddts_arr = np.asarray(row["plddts"][i]).ravel()
                plddt_str = ",".join(f"{x:.1f}" for x in plddts_arr)
                coords_fh.write(f"{plddt_str}\n")

            global_indices_for_row.append(seq_global_index)
            s20_of_selected.append(s20[i])
            seq_global_index += 1

        # Build S20 groups and add mapping entries (use fam_id.g_ix)
        # Group indices by original S20 id keeping order of first appearance
        s20_order = list(_remap_ids_preserve_order(s20_of_selected).keys())
        s20_to_group_ix = {s20_id: ix for ix, s20_id in enumerate(s20_order)}

        groups: Dict[int, List[int]] = {}
        for local_pos, s20_id in enumerate(s20_of_selected):
            gix = s20_to_group_ix[s20_id]
            groups.setdefault(gix, []).append(global_indices_for_row[local_pos])

        seq_file_name = os.path.basename(seq_path)
        for gix in sorted(groups.keys()):
            inds = groups[gix]
            inds_str = ",".join(str(x) for x in inds)
            mapping_line = f"{seq_file_name}:{inds_str}"
            mapping_entries.append((f"{fam_id}.{gix}", mapping_line))

    if seq_fh is not None:
        seq_fh.close()
    if coords_fh is not None:
        coords_fh.close()

    # Write single S20 mapping file
    # Only write mapping file if there is at least one entry
    if mapping_entries:
        mapping_path = os.path.join(os.path.dirname(seq_path), f"{basename_prefix}_min20_max90.mapping")
        with open(mapping_path, "w") as mf:
            current_key = None
            for i, (key, mline) in enumerate(mapping_entries):
                if key != current_key:
                    if i != 0:
                        mf.write("\n")
                    mf.write(f">{key}\n")
                    current_key = key
                mf.write(mline)


def convert_parquet_to_text_s20(parquet_file: str, output_dir: str, excluded_accessions: Optional[Set[str]] = None) -> None:
    base = os.path.splitext(os.path.basename(parquet_file))[0]
    out_prefix = os.path.join(output_dir, base)
    _ensure_output_dir(out_prefix)

    seq_fp = f"{out_prefix}.sequences"
    map_fp = f"{out_prefix}_min20_max90.mapping"
    if os.path.exists(seq_fp) and os.path.exists(map_fp):
        print(f"Skipping existing outputs for {parquet_file}")
        return

    print(f"Reading parquet: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows")
    write_rows_to_text_s20(df, out_prefix, excluded_accessions)


def _load_excluded_accessions(list_path: Optional[str]) -> Optional[Set[str]]:
    """Load a set of accession strings to exclude. Returns None if list_path is None.

    The file is expected to contain one accession per line. Empty lines and whitespace are ignored.
    """
    excluded: Set[str] = set()
    with open(list_path, "r") as fh:
        for line in fh:
            acc = line.strip()
            if acc:
                excluded.add(acc)
    return excluded


def _write_readme(output_dir: str, args: argparse.Namespace, exclusion_list_path: Optional[str], excluded_count: Optional[int]) -> None:
    """Write a README.md into output_dir describing how the dataset was created.

    If README.md already exists, do nothing.
    """
    readme_path = os.path.join(output_dir, "README.md")
    if os.path.exists(readme_path):
        return
    timestamp = datetime.now().isoformat(timespec="seconds")
    script_path = os.path.relpath(__file__, start=os.getcwd()) if os.path.exists(__file__) else __file__
    lines: List[str] = []
    lines.append(f"Dataset created by {script_path} on {timestamp}\n")
    lines.append("Inputs/parameters:\n")
    lines.append(f"- Parquet glob pattern: {args.pattern}\n")
    if args.task_index is not None and args.num_tasks is not None:
        lines.append(f"- Task partition: task_index={args.task_index}, num_tasks={args.num_tasks}\n")
    lines.append(f"- Output directory: {output_dir}\n")
    if exclusion_list_path is not None:
        excl_count_str = "unknown" if excluded_count is None else str(excluded_count)
        lines.append(f"- Excluded accessions list: {exclusion_list_path} (loaded {excl_count_str})\n")
    else:
        lines.append("- Excluded accessions list: none\n")
    lines.append("\nMethod:\n")
    lines.append("- For each family row, within each S20 cluster, keep at most one sequence for each S90 label (first non-excluded occurrence).\n")
    lines.append("- Only S20 groups with at least two selected members (>=2 distinct S90 representatives) are retained.\n")
    lines.append("- Sequences are written to '<basename>.sequences' with headers in the form 'accession/s20.s90'.\n")
    lines.append("- If present, per-residue coordinates (N, CA, C, O) and pLDDTs are written to '<basename>.coords'.\n")
    lines.append("- A mapping file '<basename>_min20_max90.mapping' groups indices by S20, referencing the .sequences file.\n")
    lines.append("\nReproducibility:\n")
    lines.append("- This README was auto-generated by the script at runtime to document dataset provenance.\n")
    with open(readme_path, "w") as rh:
        rh.write("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=None, help="Task index (0-based) for partitioned processing")
    parser.add_argument("--num_tasks", type=int, default=None, help="Total number of tasks to partition files")
    parser.add_argument("--pattern", type=str, default="../data/ted/s100_parquets/train_test_split_v3_debug/train_filtered/train_train*.parquet", help="Glob pattern for parquet files")
    parser.add_argument("--output_dir", type=str, default="../data/ted/s100_text_min_20_max_90/train_test_split_v2/train_filtered", help="Output directory root")
    parser.add_argument("--exclude_accessions", type=str, default="../data/ted/ted_discontinuous_domains.list", help="Path to a file listing accessions to exclude (one per line)")
    args = parser.parse_args()

    parquet_path_pattern = args.pattern
    output_dir = args.output_dir

    parquet_files = sorted(glob.glob(parquet_path_pattern))
    total_files = len(parquet_files)
    print(f"Found {total_files} parquet files matching pattern")

    # Load exclusions if provided
    excluded_accessions = _load_excluded_accessions(args.exclude_accessions)
    if excluded_accessions is not None:
        print(f"Loaded {len(excluded_accessions)} excluded accessions from {args.exclude_accessions}")

    # Validate task partitioning args
    if (args.task_index is None) ^ (args.num_tasks is None):
        raise ValueError("Both --task_index and --num_tasks must be provided together, or neither.")

    if args.task_index is not None and args.num_tasks is not None:
        if args.num_tasks <= 0:
            raise ValueError("--num_tasks must be > 0")
        if not (0 <= args.task_index < args.num_tasks):
            raise ValueError("--task_index must satisfy 0 <= task_index < num_tasks")

        k = args.task_index
        T = args.num_tasks
        start_idx = (k * total_files) // T
        end_idx = ((k + 1) * total_files) // T
        parquet_files = parquet_files[start_idx:end_idx]
        print(f"Task {k}/{T}: processing files [{start_idx}:{end_idx}) -> {len(parquet_files)} files")

    for pq in parquet_files:
        subdir = os.path.basename(os.path.dirname(pq))
        # If the provided output_dir already points to the exact subdir, reuse it; otherwise create a subdir under it
        out_root = output_dir if os.path.basename(output_dir) == subdir else os.path.join(output_dir, subdir)
        if not os.path.exists(out_root):
            os.makedirs(out_root, exist_ok=True)
        # Write a README once per output directory
        _write_readme(out_root, args, args.exclude_accessions, None if excluded_accessions is None else len(excluded_accessions))
        convert_parquet_to_text_s20(pq, out_root, excluded_accessions)