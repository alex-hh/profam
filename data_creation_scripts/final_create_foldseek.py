import glob
import os
from data_creation_scripts.create_foldseek import fasta_to_parquet


if __name__ == "__main__":
    save_dir = "/SAN/orengolab/cath_plm/ProFam/data/foldseek/"
    completed = [int(f.split(".")[0]) for f in os.listdir(save_dir) if f.endswith(".parquet")]
    fasta_files = glob.glob("/SAN/orengolab/cath_plm/ProFam/data/foldseek/*.fasta")
    last_batch = max(completed)
    batch_id = last_batch + len(fasta_files)
    fasta_to_parquet(
        save_dir=save_dir,
        batch_id=batch_id,
    )



