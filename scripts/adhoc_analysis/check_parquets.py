import glob
import os

import pandas as pd
from tqdm import tqdm


def check_parquet(parq_path):
    try:
        df = pd.read_parquet(parq_path)
        failed = False

        required_columns = ["sequences", "accessions", "fam_id"]
        for col in required_columns:
            if col not in df.columns:
                print(f"{parq_path} does not have {col} column")
                failed = True

        if not failed:
            for idx, row in df.iterrows():
                sequences = row["sequences"]
                accessions = row["accessions"]

                if len(sequences) != len(accessions):
                    print(
                        f"{parq_path} has different number of sequences and accessions in row {idx}"
                    )
                    failed = True
                    break

                for seq in sequences:
                    if not isinstance(seq, str) or len(seq) < 3:
                        print(f"{parq_path} has invalid sequence in row {idx}: {seq}")
                        failed = True
                        break

                if failed:
                    break

                if len(sequences) == 0:
                    print(f"{parq_path} has empty sequences in row {idx}")
                    failed = True
                    break

                if len(accessions) == 0:
                    print(f"{parq_path} has empty accessions in row {idx}")
                    failed = True
                    break

        return failed
    except Exception as e:
        print(f"Error processing {parq_path}: {str(e)}")
        return True


# Main script
if __name__ == "__main__":
    glob_patterns = [
        ("../foldseek_struct/*.parquet", "Foldseek struct"),
        ("../foldseek_representatives/*.parquets", "Foldseek representatives"),
        ("../data/pfam/train_test_split_parquets_v2/**/*.parquet", "Pfam"),
        ("../data/GO_MF/mfparquets/*.parquet", "GO MF"),
        ("../data/funfams/parquets/*.parquet", "Funfams"),
    ]

    parquet_filepaths = []
    for pattern, name in glob_patterns:
        files = glob.glob(pattern, recursive=True)
        print(f"Found {len(files)} parquet files for {name}")
        parquet_filepaths.extend(files)

    # ted_parquets = None # currently TED is fasta format
    # gene3d = None # currently gene3d is fasta format
    # ec = None # currently ec is fasta format

    fail_counter = 0
    print(f"Found {len(parquet_filepaths)} parquet files in total")
    for parquet_file in tqdm(parquet_filepaths):
        failed = check_parquet(parquet_file)
        if failed:
            fail_counter += 1
    print(f"Found {fail_counter} failed parquets")
