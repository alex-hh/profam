import glob
import os

import pyarrow.parquet as pq
from tqdm import tqdm


def check_parquet(parq_path):
    try:
        table = pq.read_table(parq_path)
        df = table.to_pandas()
        failed = False

        required_columns = ["sequences", "accessions", "fam_id"]
        for col in required_columns:
            if col not in df.columns:
                print(f"{parq_path} does not have {col} column")
                failed = True

        if not failed:
            seq_lens = df["sequences"].str.len()
            accessions_lens = df["accessions"].str.len()

            if (seq_lens == 0).any():
                print(f"{parq_path} has empty sequences")
                failed = True
            if (accessions_lens == 0).any():
                print(f"{parq_path} has empty accessions")
                failed = True
            if not (seq_lens == accessions_lens).all():
                print(f"{parq_path} has different number of sequences and accessions")
                failed = True

        return failed
    except Exception as e:
        print(f"Error processing {parq_path}: {str(e)}")
        return True


# Main script remains the same
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
