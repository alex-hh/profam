import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def fasta_to_parquet(save_dir, batch_id):
    fastas = glob.glob(save_dir + "*.fasta")
    results = []
    for fasta in fastas:
        with open(fasta, "r") as f:
            text = f.read()
        results.append({'text': text})
        os.remove(fasta)
    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    output_file = f'{save_dir}/{batch_id}.parquet'
    pq.write_table(table, output_file)
    print(f"Saved batch {batch_id} to {output_file}")
    return output_file

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



