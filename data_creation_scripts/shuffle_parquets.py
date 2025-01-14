import os
import sys
import pandas as pd
import pyarrow.parquet as pq

"""
Shuffles documents across and within parquet files
designed to avoid loading all parquets into main memory
"""

class ParquetShuffle:
    def __init__(self, input_dir, output_dir, limit_mb=250):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.limit_mb = limit_mb
        self.family_df = None

    def get_size_of_row(self, row):
        """
        Calculates the size of a parquet row (family) in megabytes.
        Consistent with parquet_buffer_writer.py
        """
        seqs_mb = sum([sys.getsizeof(s) for s in row['sequences']]) / (1024 * 1024)
        return seqs_mb

    def create_family_dataframe(self):
        """
        Creates a DataFrame where each row corresponds to a family,
        containing the family ID, size in MB, and the file it came from.
        """
        df_rows = []
        parquet_files = [f for f in os.listdir(self.input_dir) if f.endswith('.parquet')]
        for parquet_file in parquet_files:
            filepath = os.path.join(self.input_dir, parquet_file)
            # Read only the 'fam_id' and 'sequences' columns to reduce memory usage
            df = pd.read_parquet(filepath, columns=['fam_id', 'sequences'])
            for index, row in df.iterrows():
                fam_id = row['fam_id']
                size_mb = self.get_size_of_row(row)
                df_rows.append({
                    'fam_id': fam_id,
                    'size_mb': size_mb,
                    'parquet_file': parquet_file
                })
        self.family_df = pd.DataFrame(df_rows)
        print(f"Total families found: {len(self.family_df)}")

    def make_shuffled_parquet_file(self, families_to_include, output_file):
        """
        Creates a shuffled parquet file containing the specified families.
        Only loads necessary data into memory.
        """
        # Map parquet files to fam_ids needed from them
        files_to_fam_ids = {}
        for item in families_to_include:
            parquet_file = item['parquet_file']
            fam_id = item['fam_id']
            if parquet_file not in files_to_fam_ids:
                files_to_fam_ids[parquet_file] = set()
            files_to_fam_ids[parquet_file].add(fam_id)

        # Collect dataframes for the required families
        dfs = []
        for parquet_file, fam_ids in files_to_fam_ids.items():
            filepath = os.path.join(self.input_dir, parquet_file)
            # Read only the rows with the required fam_ids
            df = pd.read_parquet(filepath)
            df = df[df['fam_id'].isin(fam_ids)]
            dfs.append(df)

        # Concatenate and shuffle the dataframes
        combined_df = pd.concat(dfs)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        # Write to parquet
        combined_df.to_parquet(output_file, index=False)
        print(f"Wrote {len(combined_df)} families to {output_file}")

    def shuffle_and_split(self):
        """
        Shuffles the family DataFrame and assigns families to new Parquet files,
        ensuring each file does not exceed the memory limit.
        """
        self.family_df = self.family_df.sample(frac=1).reset_index(drop=True)
        parq_ix = 0
        current_size = 0
        buffer_families = []
        for _, row in self.family_df.iterrows():
            fam_size = row['size_mb']
            if current_size + fam_size > self.limit_mb:
                # Write the buffer to a new Parquet file
                output_file = os.path.join(self.output_dir, f'shuffled_{parq_ix}.parquet')
                self.make_shuffled_parquet_file(buffer_families, output_file)
                # Reset buffer and counters
                parq_ix += 1
                current_size = 0
                buffer_families = []
            buffer_families.append(row.to_dict())
            current_size += fam_size
        # Write any remaining families
        if buffer_families:
            output_file = os.path.join(self.output_dir, f'shuffled_{parq_ix}.parquet')
            self.make_shuffled_parquet_file(buffer_families, output_file)

    def run(self):
        self.create_family_dataframe()
        self.shuffle_and_split()

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    limit_mb = 250  # Adjust as needed

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shuffler = ParquetShuffle(input_dir, output_dir, limit_mb=limit_mb)
    shuffler.run()
