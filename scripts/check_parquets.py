"""
Iterate through all the parquet files
and make sure that there is some minimum
text in the text column
"""
import glob
import os

import pyarrow.parquet as pq
from tqdm import tqdm
import pyarrow.compute as pc


def check_parquet(parq_path):
    try:
        table = pq.read_table(parq_path)
        failed = False

        required_columns = ["sequences", "accessions", "fam_id"]
        for col in required_columns:
            if col not in table.column_names:
                print(f"{parq_path} does not have {col} column")
                failed = True

        if not failed:
            seq_lengths = pc.utf8_length(table["sequences"])
            acc_lengths = pc.utf8_length(table["accessions"])

            if pc.any(seq_lengths == 0):
                print(f"{parq_path} has empty sequences")
                failed = True
            if pc.any(acc_lengths == 0):
                print(f"{parq_path} has empty accessions")
                failed = True
            if not pc.all(seq_lengths == acc_lengths):
                print(f"{parq_path} has different number of sequences and accessions")
                failed = True

        return failed
    except Exception as e:
        print(f"Error processing {parq_path}: {str(e)}")
        return True


if __name__ == "__main__":
    parquet_dirs = ["/SAN/orengolab/cath_plm/ProFam/data/GO_MF/mfparquets"]
    fail_counter = 0
    for d in parquet_dirs:
        parquet_files = glob.glob(os.path.join(d, "*.parquet"))
        print(f"Found {len(parquet_files)} parquet files in {d}")
        for parquet_file in tqdm(parquet_files):
            failed = check_parquet(parquet_file)
            if failed:
                fail_counter += 1
    print(f"Found {fail_counter} failed parquets")
