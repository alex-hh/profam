import glob
import os
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import tqdm


def rename_column_in_parquet(file_path, old_column_name, new_column_name):
    """
    Renames a column in a Parquet file and saves the updated file.
    
    :param file_path: Path to the input Parquet file.
    :param old_column_name: The name of the column to rename.
    :param new_column_name: The new name for the column.
    :param output_dir: Directory where the updated file will be saved. If None, it overwrites the original file.
    """
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(file_path)

    # Rename the column
    if old_column_name in df.columns:
        df.rename(columns={old_column_name: new_column_name}, inplace=True)
        print(f"Renamed column '{old_column_name}' to '{new_column_name}' in {file_path}")
        df.to_parquet(file_path, index=False)
    else:
        print(f"Column '{old_column_name}' not found in {file_path}")
        return


def convert_row_to_float16(row, column_names):
    for column_name in column_names:
        row[column_name] = [np.array(arr, dtype='float16') for arr in row[column_name]]
    return row


def convert_coords_to_float16(file_path, column_names):
    """
    Converts columns in a Parquet file to float16.
    
    :param file_path: Path to the input Parquet file.
    :param column_names: List of column names to convert to float16.
    :param output_dir: Directory where the updated file will be saved. If None, it overwrites the original file.
    """
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(file_path)
    df.apply(lambda row: convert_row_to_float16(row, column_names), axis=1)

    # Convert columns to float16
    table = pa.Table.from_pandas(df)
    print(table.schema)

    # Save the updated file
    pq.write_table(table, file_path)


def process_parquet_files(file_list, old_column_name, new_column_name):
    """
    Processes a list of Parquet files, renaming a specific column in each file.
    
    :param file_list: List of paths to Parquet files.
    :param old_column_name: The name of the column to rename.
    :param new_column_name: The new name for the column.
    :param output_dir: Directory where the updated files will be saved. If None, it overwrites the original files.
    """
    for file_path in tqdm.tqdm(file_list):
        print(file_path)
        rename_column_in_parquet(file_path, old_column_name, new_column_name)
        convert_coords_to_float16(file_path, ["N", "CA", "C", "O", "plddts"])


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename a column in a Parquet file.")
    parser.add_argument("data_folder", help="Folder containing Parquet files.")
    parser.add_argument("--parquet_index", type=int, default=None)
    args = parser.parse_args()
    # List of Parquet files
    data_dir = os.environ.get("DATA_DIR", "/SAN/orengolab/cath_plm/ProFam/data")
    print("Data directory: (set DATA_DIR env var to override)", data_dir)
    if args.parquet_index is None:
        glob_str = os.path.join(data_dir, args.data_folder, '*.parquet')
        parquet_files = glob.glob(glob_str)
    else:
        parquet_files = [os.path.join(data_dir, args.data_folder, f"{args.parquet_index}.parquet")]

    # Rename 'cluster_id' to 'fam_id' in each file
    process_parquet_files(parquet_files, old_column_name='cluster_id', new_column_name='fam_id')
