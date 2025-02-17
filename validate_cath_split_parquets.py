import glob
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

# Configure logging to display time, level and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ROOT_PATH = "/SAN/orengolab/cath_plm/ProFam"


def calculate_total_rows():
    total_parent_rows = 0
    for parquet_file in parent_parquet_files:
        logging.info(f"Reading parent parquet file: {parquet_file}")
        try:
            df = pd.read_parquet(parquet_file)
        except:
            df = pl.read_parquet(parquet_file).to_pandas()
        total_parent_rows += len(df)

    logging.info(f"Total rows in parent parquets: {total_parent_rows}")


def validate_parquets():
    # Paths to datasets
    # Dataset 1 to 9
    datasets = {
        1: {
            "name": "ted_s100",
            "parent_dir": f"{ROOT_PATH}/data/ted/s100_parquets",
            "split_dir": f"{ROOT_PATH}/data/ted/s100_parquets/train_val_test_split_for_debug",
        },
    }

    # datasets = {
    #     1: {
    #         "name": "ted_s100",
    #         "parent_dir": f"{ROOT_PATH}/data/ted/s100_parquets",
    #         "split_dir": f"{ROOT_PATH}/data/ted/s100_parquets/train_val_test_split",
    #     },
    #     2: {
    #         "name": "ted_s50",
    #         "parent_dir": f"{ROOT_PATH}/data/ted/s50_parquets",
    #         "split_dir": f"{ROOT_PATH}/data/ted/s50_parquets/train_val_test_split",
    #     },
    #     3: {
    #         "name": "funfam_s100_noali",
    #         "parent_dir": f"{ROOT_PATH}/data/funfams/s100_noali_parquets",
    #         "split_dir": f"{ROOT_PATH}/data/funfams/s100_noali_parquets/train_val_test_split",
    #     },
    #     4: {
    #         "name": "funfam_s50",
    #         "parent_dir": f"{ROOT_PATH}/data/funfams/s50_parquets",
    #         "split_dir": f"{ROOT_PATH}/data/funfams/s50_parquets/train_val_test_split",
    #     },
    #     5: {
    #         "name": "foldseek_s100_raw",
    #         "parent_dir": f"{ROOT_PATH}/data/foldseek/foldseek_s100_raw",
    #         "split_dir": f"{ROOT_PATH}/data/foldseek/foldseek_s100_raw/train_val_test_split",
    #     },
    #     6: {
    #         "name": "afdb_s50_single",
    #         "parent_dir": f"{ROOT_PATH}/data/afdb_s50_single",
    #         "split_dir": f"{ROOT_PATH}/data/afdb_s50_single/train_val_test_split",
    #     },
    #     7: {
    #         "name": "foldseek_s100_struct",
    #         "parent_dir": f"{ROOT_PATH}/data/foldseek/foldseek_s100_struct",
    #         "split_dir": f"{ROOT_PATH}/data/foldseek/foldseek_s100_struct/train_val_test_split",
    #     },
    #     8: {
    #         "name": "foldseek_reps_single",
    #         "parent_dir": f"{ROOT_PATH}/data/foldseek/foldseek_reps_single",
    #         "split_dir": f"{ROOT_PATH}/data/foldseek/foldseek_reps_single/train_val_test_split",
    #     },
    #     9: {
    #         "name": "foldseek_s50_struct",
    #         "parent_dir": f"{ROOT_PATH}/data/foldseek/foldseek_s50_struct",
    #         "split_dir": f"{ROOT_PATH}/data/foldseek/foldseek_s50_struct/train_val_test_split",
    #     },
    # }
    # breakpoint()
    # Initialize overall results
    dataset_results = []
    parquet_file_results = []

    for dataset_id, dataset_info in datasets.items():
        dataset_name = dataset_info["name"]
        parent_dir = dataset_info["parent_dir"]
        split_dir = dataset_info["split_dir"]

        logging.info(f"Validating dataset {dataset_id}: {dataset_name}")

        # Load parent parquets and count total number of rows
        parent_parquet_files = glob.glob(os.path.join(parent_dir, "*.parquet"))
        logging.info(
            f"Found {len(parent_parquet_files)} parent parquet files in directory {parent_dir}"
        )

        total_parent_rows = 0
        for parquet_file in parent_parquet_files:
            logging.info(f"Reading parent parquet file: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            total_parent_rows += len(df)

        logging.info(f"Total rows in parent parquets: {total_parent_rows}")

        # Initialize counters
        total_split_rows = 0
        split_accessions = {"train": set(), "val": set(), "test": set()}
        overlap_counts = {"train_test": 0, "train_val": 0, "val_test": 0}

        splits = ["train", "val", "test"]
        dataset_stats = {"dataset_id": dataset_id, "dataset_name": dataset_name}
        dataset_stats.update(
            {
                "total_parent_rows": total_parent_rows,
                "total_train_rows": 0,
                "total_val_rows": 0,
                "total_test_rows": 0,
                "rows_with_invalid_fam_id": 0,
                "rows_with_invalid_accessions_or_sequences": 0,
                "count_accessions_len_less_than_2": 0,
                "rows_with_empty_accessions_or_sequences": 0,
                "min_sequence_length": None,  # To be updated
            }
        )

        for split in splits:
            split_dir_split = os.path.join(split_dir, split)
            split_parquet_files = glob.glob(os.path.join(split_dir_split, "*.parquet"))
            logging.info(
                f"Processing '{split}' split. Found {len(split_parquet_files)} parquet files in directory {split_dir_split}"
            )

            total_split_rows_split = 0

            invalid_fam_id = 0
            invalid_accessions_or_sequences = 0
            accessions_len_less_than_2 = 0
            empty_accessions_or_sequences = 0
            min_seq_length = None

            for parquet_file in split_parquet_files:
                logging.info(f"Reading split parquet file: {parquet_file}")
                try:
                    df = pd.read_parquet(parquet_file)
                except:
                    breakpoint()
                    df = pl.read_parquet(parquet_file).to_pandas()

                total_split_rows_split += len(df)
                parquet_file_stats = {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "split": split,
                    "parquet_file": parquet_file,
                    "num_rows": len(df),
                    "rows_with_invalid_fam_id": 0,
                    "rows_with_invalid_accessions_or_sequences": 0,
                    "count_accessions_len_less_than_2": 0,
                    "rows_with_empty_accessions_or_sequences": 0,
                    "min_sequence_length": None,  # To be updated
                }

                for idx, row in df.iterrows():
                    # Check fam_id is non-empty
                    if not row.get("fam_id"):
                        invalid_fam_id += 1
                        parquet_file_stats["rows_with_invalid_fam_id"] += 1

                    # Check accessions and sequences are non-empty
                    accessions = row.get("accessions")
                    sequences = row.get("sequences")
                    if accessions is None or sequences is None:
                        invalid_accessions_or_sequences += 1
                        parquet_file_stats[
                            "rows_with_invalid_accessions_or_sequences"
                        ] += 1
                        continue

                    # Check they are numpy arrays
                    if not isinstance(accessions, np.ndarray) or not isinstance(
                        sequences, np.ndarray
                    ):
                        invalid_accessions_or_sequences += 1
                        parquet_file_stats[
                            "rows_with_invalid_accessions_or_sequences"
                        ] += 1
                        continue

                    # Check lengths are equal
                    if len(accessions) != len(sequences):
                        invalid_accessions_or_sequences += 1
                        parquet_file_stats[
                            "rows_with_invalid_accessions_or_sequences"
                        ] += 1
                        continue

                    # Accessions and sequences should not be empty arrays
                    if len(accessions) == 0 or len(sequences) == 0:
                        empty_accessions_or_sequences += 1
                        parquet_file_stats[
                            "rows_with_empty_accessions_or_sequences"
                        ] += 1

                    # Min length of sequence strings in sequences array should be at least 5
                    seq_lengths = [
                        len(seq) for seq in sequences if isinstance(seq, str)
                    ]
                    if seq_lengths:
                        min_length = min(seq_lengths)
                        if (
                            parquet_file_stats["min_sequence_length"] is None
                            or min_length < parquet_file_stats["min_sequence_length"]
                        ):
                            parquet_file_stats["min_sequence_length"] = min_length
                        if (
                            dataset_stats["min_sequence_length"] is None
                            or min_length < dataset_stats["min_sequence_length"]
                        ):
                            dataset_stats["min_sequence_length"] = min_length
                    else:
                        min_length = None

                    # Count when length of accessions array is less than 2
                    if len(accessions) < 2:
                        accessions_len_less_than_2 += 1
                        parquet_file_stats["count_accessions_len_less_than_2"] += 1

                    # Collect accessions for overlap check
                    split_accessions[split].update(accessions)

                parquet_file_results.append(parquet_file_stats)

            dataset_stats["total_" + split + "_rows"] = total_split_rows_split
            dataset_stats["rows_with_invalid_fam_id"] += invalid_fam_id
            dataset_stats[
                "rows_with_invalid_accessions_or_sequences"
            ] += invalid_accessions_or_sequences
            dataset_stats[
                "rows_with_empty_accessions_or_sequences"
            ] += empty_accessions_or_sequences
            dataset_stats[
                "count_accessions_len_less_than_2"
            ] += accessions_len_less_than_2
            if dataset_stats["min_sequence_length"] is None or (
                parquet_file_stats["min_sequence_length"] is not None
                and parquet_file_stats["min_sequence_length"]
                < dataset_stats["min_sequence_length"]
            ):
                dataset_stats["min_sequence_length"] = parquet_file_stats[
                    "min_sequence_length"
                ]

            logging.info(
                f"Completed processing '{split}' split for dataset '{dataset_name}'. Total rows: {total_split_rows_split}"
            )

        # Total split rows
        total_split_rows = (
            dataset_stats["total_train_rows"]
            + dataset_stats["total_val_rows"]
            + dataset_stats["total_test_rows"]
        )
        dataset_stats["total_split_rows"] = total_split_rows

        # Check total rows
        if total_parent_rows != total_split_rows:
            logging.warning(
                f"Total rows in splits ({total_split_rows}) do not match parent total rows ({total_parent_rows}) for dataset '{dataset_name}'"
            )

        # Count accessions that occur in both splits
        train_accessions = split_accessions["train"]
        val_accessions = split_accessions["val"]
        test_accessions = split_accessions["test"]

        train_test_overlap = train_accessions & test_accessions
        train_val_overlap = train_accessions & val_accessions
        val_test_overlap = val_accessions & test_accessions

        overlap_counts["train_test"] = len(train_test_overlap)
        overlap_counts["train_val"] = len(train_val_overlap)
        overlap_counts["val_test"] = len(val_test_overlap)

        dataset_stats.update(overlap_counts)

        logging.info(f"Overlap counts for dataset '{dataset_name}': {overlap_counts}")

        dataset_results.append(dataset_stats)

    # Convert results to DataFrames and save as CSV
    logging.info(
        "Converting dataset results to DataFrame and saving to 'validation_results_by_dataset.csv'"
    )
    dataset_results_df = pd.DataFrame(dataset_results)
    dataset_results_df.to_csv("validation_results_by_dataset.csv", index=False)

    logging.info(
        "Converting parquet file results to DataFrame and saving to 'validation_results_by_parquet_file.csv'"
    )
    parquet_file_results_df = pd.DataFrame(parquet_file_results)
    parquet_file_results_df.to_csv(
        "validation_results_by_parquet_file.csv", index=False
    )

    logging.info("Validation completed successfully.")


if __name__ == "__main__":
    validate_parquets()
