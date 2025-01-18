import glob
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Configure logging to display time, level and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Datasets definition (same as original)
DATASETS = {
    # 1: {
    #     "name": "ted_s100",
    #     "parent_dir": "../data/ted/s100_parquets",
    #     "split_dir": "../data/ted/s100_parquets/train_val_test_split",
    # },
    # 2: {
    #     "name": "ted_s50",
    #     "parent_dir": "../data/ted/s50_parquets",
    #     "split_dir": "../data/ted/s50_parquets/train_val_test_split",
    # },
    # 3: {
    #     "name": "funfam_s100_noali",
    #     "parent_dir": "../data/funfams/s100_noali_parquets",
    #     "split_dir": "../data/funfams/s100_noali_parquets/train_val_test_split",
    # },
    # 4: {
    #     "name": "funfam_s50",
    #     "parent_dir": "../data/funfams/s50_parquets",
    #     "split_dir": "../data/funfams/s50_parquets/train_val_test_split",
    # },
    # 5: {
    #     "name": "foldseek_s100_raw",
    #     "parent_dir": "../data/foldseek/foldseek_s100_raw",
    #     "split_dir": "../data/foldseek/foldseek_s100_raw/train_val_test_split",
    # },
    # 6: {
    #     "name": "afdb_s50_single",
    #     "parent_dir": "../data/afdb_s50_single",
    #     "split_dir": "../data/afdb_s50_single/train_val_test_split",
    # },
    7: {
        "name": "foldseek_s100_struct",
        "parent_dir": "../data/foldseek/foldseek_s100_struct",
        "split_dir": "../data/foldseek/foldseek_s100_struct/train_val_test_split",
    },
    8: {
        "name": "foldseek_reps_single",
        "parent_dir": "../data/foldseek/foldseek_reps_single",
        "split_dir": "../data/foldseek/foldseek_reps_single/train_val_test_split",
    },
    9: {
        "name": "foldseek_s50_struct",
        "parent_dir": "../data/foldseek/foldseek_s50_struct",
        "split_dir": "../data/foldseek/foldseek_s50_struct/train_val_test_split",
    },
}


def _count_rows_in_parquet(parquet_file):
    """Helper function to read a parquet file and return its row count."""
    try:
        df_parent = pd.read_parquet(parquet_file)
        return len(df_parent)
    except Exception as e:
        logging.error(f"Failed to read {parquet_file}: {e}")
        return 0


def _process_split_parquet_file(parquet_file, dataset_id, dataset_name, split):
    """
    Reads a single split parquet file and returns accumulated stats:
    (num_rows,
     invalid_fam_id_count,
     invalid_acc_seq_count,
     empty_acc_seq_count,
     acc_len_lt_2_count,
     min_seq_len,
     updated_accessions_set)
    """
    parquet_stats = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "split": split,
        "parquet_file": parquet_file,
        "num_rows": 0,
        "rows_with_invalid_fam_id": 0,
        "rows_with_invalid_accessions_or_sequences": 0,
        "count_accessions_len_less_than_2": 0,
        "rows_with_empty_accessions_or_sequences": 0,
        # "min_sequence_length": None,
        "updated_accessions_set": set(),
    }

    try:
        df_split = pd.read_parquet(parquet_file)
    except Exception as e:
        logging.error(f"Failed to read {parquet_file}: {e}")
        return parquet_stats

    parquet_stats["num_rows"] = len(df_split)

    for idx, row in df_split.iterrows():
        # Check fam_id
        if not row.get("fam_id"):
            parquet_stats["rows_with_invalid_fam_id"] += 1

        accessions = row.get("accessions")
        sequences = row.get("sequences")

        # Check None or array type
        if accessions is None or sequences is None:
            parquet_stats["rows_with_invalid_accessions_or_sequences"] += 1
            continue
        if not isinstance(accessions, np.ndarray) or not isinstance(
            sequences, np.ndarray
        ):
            parquet_stats["rows_with_invalid_accessions_or_sequences"] += 1
            continue

        # Check lengths
        if len(accessions) != len(sequences):
            parquet_stats["rows_with_invalid_accessions_or_sequences"] += 1
            continue

        # Check empty arrays
        if len(accessions) == 0 or len(sequences) == 0:
            parquet_stats["rows_with_empty_accessions_or_sequences"] += 1

        # # Minimum sequence length
        # seq_lengths = [len(seq) for seq in sequences if isinstance(seq, str)]
        # if seq_lengths:
        #     min_length = min(seq_lengths)
        #     if (
        #         parquet_stats["min_sequence_length"] is None
        #         or min_length < parquet_stats["min_sequence_length"]
        #     ):
        #         parquet_stats["min_sequence_length"] = min_length

        # Count short accessions
        if len(accessions) < 2:
            parquet_stats["count_accessions_len_less_than_2"] += 1

        # # Collect accessions
        # parquet_stats["updated_accessions_set"].update(accessions)

    return parquet_stats


def validate_parquets_parallel(
    output_dir="validate_cath_split_parquets_results", max_workers=64
):
    """
    Validate parquets by parallelizing at the parquet-file level.
    1) Parallel read for parent parquets to count rows.
    2) For each split (train/val/test), parallel read and process each parquet.
    3) Collect dataset-level stats, save partial CSV, then combine everything.
    """
    os.makedirs(output_dir, exist_ok=True)

    # We'll collect final results for all datasets
    all_dataset_results = []
    all_parquet_file_results = []

    for dataset_id, dataset_info in DATASETS.items():
        dataset_name = dataset_info["name"]
        parent_dir = dataset_info["parent_dir"]
        split_dir = dataset_info["split_dir"]

        logging.info(f"Validating dataset {dataset_id}: {dataset_name}")

        # ------------------------------------------------
        # 1) Parallel read for parent parquet files to count total parent rows
        # ------------------------------------------------
        parent_parquet_files = glob.glob(os.path.join(parent_dir, "*.parquet"))
        logging.info(
            f"Found {len(parent_parquet_files)} parent parquet files in directory {parent_dir}"
        )

        total_parent_rows = 0
        # Use thread pool for counting rows
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_count_rows_in_parquet, pf): pf
                for pf in parent_parquet_files
            }
            for future in as_completed(futures):
                total_parent_rows += future.result()

        logging.info(
            f"Total rows in parent parquets for {dataset_name}: {total_parent_rows}"
        )

        # Initialize overall results for this dataset
        dataset_stats = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "total_parent_rows": total_parent_rows,
            "total_train_rows": 0,
            "total_val_rows": 0,
            "total_test_rows": 0,
            "rows_with_invalid_fam_id": 0,
            "rows_with_invalid_accessions_or_sequences": 0,
            "rows_with_empty_accessions_or_sequences": 0,
            "count_accessions_len_less_than_2": 0,
            # "min_sequence_length": None,
        }

        # We will accumulate all parquet-file-level stats for this dataset
        dataset_parquet_file_stats = []

        # For overlap tracking
        split_accessions = {"train": set(), "val": set(), "test": set()}

        # ------------------------------------------------
        # 2) For each split, parallel read/process each parquet file
        # ------------------------------------------------
        for split in ["train", "val", "test"]:
            split_dir_split = os.path.join(split_dir, split)
            split_parquet_files = glob.glob(os.path.join(split_dir_split, "*.parquet"))
            logging.info(
                f"Processing '{split}' split for {dataset_name}. "
                f"Found {len(split_parquet_files)} parquet files."
            )

            # We'll capture sums to update dataset_stats
            total_split_rows_split = 0
            invalid_fam_id = 0
            invalid_acc_seq = 0
            empty_acc_seq = 0
            acc_len_lt_2 = 0
            min_seq_length_for_split = None

            # Parallel processing of each split parquet
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_parquet = {
                    executor.submit(
                        _process_split_parquet_file, pf, dataset_id, dataset_name, split
                    ): pf
                    for pf in split_parquet_files
                }

                for fut in as_completed(future_to_parquet):
                    pf_stats = fut.result()
                    dataset_parquet_file_stats.append(pf_stats)

                    total_split_rows_split += pf_stats["num_rows"]
                    invalid_fam_id += pf_stats["rows_with_invalid_fam_id"]
                    invalid_acc_seq += pf_stats[
                        "rows_with_invalid_accessions_or_sequences"
                    ]
                    empty_acc_seq += pf_stats["rows_with_empty_accessions_or_sequences"]
                    acc_len_lt_2 += pf_stats["count_accessions_len_less_than_2"]
                    # if pf_stats["min_sequence_length"] is not None:
                    #     if (
                    #         min_seq_length_for_split is None
                    #         or pf_stats["min_sequence_length"] < min_seq_length_for_split
                    #     ):
                    #         min_seq_length_for_split = pf_stats["min_sequence_length"]

                    # Update overlap set
                    split_accessions[split].update(pf_stats["updated_accessions_set"])

            # Update dataset-level stats
            dataset_stats[f"total_{split}_rows"] = total_split_rows_split
            dataset_stats["rows_with_invalid_fam_id"] += invalid_fam_id
            dataset_stats[
                "rows_with_invalid_accessions_or_sequences"
            ] += invalid_acc_seq
            dataset_stats["rows_with_empty_accessions_or_sequences"] += empty_acc_seq
            dataset_stats["count_accessions_len_less_than_2"] += acc_len_lt_2

            # if (
            #     min_seq_length_for_split is not None
            #     and (dataset_stats["min_sequence_length"] is None
            #          or min_seq_length_for_split < dataset_stats["min_sequence_length"])
            # ):
            #     dataset_stats["min_sequence_length"] = min_seq_length_for_split

        # ------------------------------------------------
        # 3) Compute totals and overlap + warnings
        # ------------------------------------------------
        total_split_rows = (
            dataset_stats["total_train_rows"]
            + dataset_stats["total_val_rows"]
            + dataset_stats["total_test_rows"]
        )
        dataset_stats["total_split_rows"] = total_split_rows

        # Check mismatch parent vs. split row counts
        if total_parent_rows != total_split_rows:
            logging.warning(
                f"Total rows in splits ({total_split_rows}) do not match parent total rows "
                f"({total_parent_rows}) for dataset '{dataset_name}'"
            )

        # # Overlap checks
        # train_accessions = split_accessions["train"]
        # val_accessions = split_accessions["val"]
        # test_accessions = split_accessions["test"]
        #
        # overlap_counts = {
        #     "train_test": len(train_accessions.intersection(test_accessions)),
        #     "train_val": len(train_accessions.intersection(val_accessions)),
        #     "val_test": len(val_accessions.intersection(test_accessions)),
        # }
        # dataset_stats.update(overlap_counts)
        #
        # logging.info(f"Overlap counts for dataset '{dataset_name}': {overlap_counts}")

        # ------------------------------------------------
        # 4) Save partial CSV results for this dataset
        # ------------------------------------------------
        pd.DataFrame([dataset_stats]).to_csv(
            os.path.join(
                output_dir, f"validation_results_by_dataset_{dataset_name}.csv"
            ),
            index=False,
        )
        pd.DataFrame(dataset_parquet_file_stats).to_csv(
            os.path.join(
                output_dir, f"validation_results_by_parquet_file_{dataset_name}.csv"
            ),
            index=False,
        )

        # Accumulate for final combined CSV
        all_dataset_results.append(dataset_stats)
        all_parquet_file_results.extend(dataset_parquet_file_stats)

    # ------------------------------------------------
    # Combine everything at the end
    # ------------------------------------------------
    dataset_results_df = pd.DataFrame(all_dataset_results)
    parquet_file_results_df = pd.DataFrame(all_parquet_file_results)

    dataset_results_df.to_csv(
        os.path.join(output_dir, "validation_results_by_dataset.csv"), index=False
    )
    parquet_file_results_df.to_csv(
        os.path.join(output_dir, "validation_results_by_parquet_file.csv"), index=False
    )

    logging.info("Parallel validation completed (file-level parallelization).")


if __name__ == "__main__":
    validate_parquets_parallel()
