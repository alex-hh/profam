import os
import re
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import hashlib
import csv

"""
Created by jude wells 2024-08-28
iterate through all parquet files
remove any rows that are duplicates
either within or across parquet files
rows are uniquely identified by a 
hash of the sorted pfam-accessions
contained in that row.
Fams that are split over multiple 
parquet files (those with names like:
Domain_PF07859.18_0.parquet are 
ignored.
"""

def deduplicate_families(pfam_dir):
    processed_hashes = set()
    dropped_rows = defaultdict(int)
    new_index = []

    # Process all parquet files in the directory
    for filename in os.listdir(pfam_dir):
        if filename.endswith('.parquet') and not re.match(r'(Domain|Family)_PF\d+\.\d+_\d+\.parquet', filename):
            filepath = os.path.join(pfam_dir, filename)
            table = pq.read_table(filepath)
            df = table.to_pandas()

            # Create a new column with sorted and hashed accessions
            df['accessions_hash'] = df['accessions'].apply(lambda x: hashlib.md5(str(sorted(x)).encode()).hexdigest())

            # Identify duplicates
            duplicates = df[df.duplicated(subset='accessions_hash', keep=False)]

            # Keep only unique rows based on accessions_hash
            unique_df = df.drop_duplicates(subset='accessions_hash', keep='first')

            # Count dropped rows per family
            dropped_counts = duplicates.groupby('fam_id').size()
            for fam_id, count in dropped_counts.items():
                dropped_rows[fam_id] += count

            # cross file duplicates
            cross_file_duplicates = unique_df[
                unique_df['accessions_hash'].isin(processed_hashes)
            ]
            for i, row in cross_file_duplicates.iterrows():
                dropped_rows[row.fam_id] += 1

            # Remove processed hashes
            unique_df = unique_df[~unique_df['accessions_hash'].isin(processed_hashes)]
            if len(unique_df):
                # Update processed_hashes
                processed_hashes.update(unique_df['accessions_hash'])

                # Update new_index
                new_index.extend([(row['fam_id'], filename) for _, row in unique_df.iterrows()])

                # Write updated dataframe back to parquet file
                unique_df = unique_df.drop(columns=['accessions_hash'])
                pq.write_table(pa.Table.from_pandas(unique_df), filepath)
            else:
                os.remove(filepath)
            print(f"from {len(df)} to {len(unique_df)}")


    # Write new index.csv
    with open(os.path.join(pfam_dir, 'new_index.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fam_id', 'parquet_file'])
        writer.writerows(new_index)

    return dropped_rows

if __name__ == "__main__":
    pfam_dir = "../data/pfam/shuffled_parquets"
    dropped_rows = deduplicate_families(pfam_dir)
    print("Rows dropped per family:", dict(dropped_rows))