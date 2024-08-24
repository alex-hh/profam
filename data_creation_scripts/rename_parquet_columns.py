import os
import pandas as pd


def rename_column_in_parquet(file_path, old_column_name, new_column_name, output_dir=None):
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
    else:
        print(f"Column '{old_column_name}' not found in {file_path}")

    # Determine the output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    else:
        output_file_path = file_path  # Overwrite the original file

    # Save the updated DataFrame back to Parquet without saving the index
    df.to_parquet(output_file_path, index=False)

def process_parquet_files(file_list, old_column_name, new_column_name, output_dir=None):
    """
    Processes a list of Parquet files, renaming a specific column in each file.
    
    :param file_list: List of paths to Parquet files.
    :param old_column_name: The name of the column to rename.
    :param new_column_name: The new name for the column.
    :param output_dir: Directory where the updated files will be saved. If None, it overwrites the original files.
    """
    for file_path in file_list:
        rename_column_in_parquet(file_path, old_column_name, new_column_name, output_dir)

# Example usage
if __name__ == "__main__":
    # List of Parquet files
    parquet_files = ['/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/0_copy.parquet']

    # Rename 'cluster_id' to 'fam_id' in each file
    process_parquet_files(parquet_files, old_column_name='cluster_id', new_column_name='fam_id', output_dir=None)
