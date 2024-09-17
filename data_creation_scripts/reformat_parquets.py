import glob
import os
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


def convert_coords_to_float16(file_path, column_names):
    """
    Converts columns in a Parquet file to float16.
    
    :param file_path: Path to the input Parquet file.
    :param column_names: List of column names to convert to float16.
    :param output_dir: Directory where the updated file will be saved. If None, it overwrites the original file.
    """
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(file_path)

    # Convert columns to float16
    df[column_names] = df[column_names].astype('float16')

    # Save the updated file
    df.to_parquet(file_path, index=False)


def process_parquet_files(file_list, old_column_name, new_column_name):
    """
    Processes a list of Parquet files, renaming a specific column in each file.
    
    :param file_list: List of paths to Parquet files.
    :param old_column_name: The name of the column to rename.
    :param new_column_name: The new name for the column.
    :param output_dir: Directory where the updated files will be saved. If None, it overwrites the original files.
    """
    for file_path in tqdm.tqdm(file_list):
        rename_column_in_parquet(file_path, old_column_name, new_column_name)
        convert_coords_to_float16(file_path, ["N", "CA", "C", "O", "plddts"])


# Example usage
if __name__ == "__main__":
    # List of Parquet files
    data_dir = os.environ.get("DATA_DIR", "/SAN/orengolab/cath_plm/ProFam/data")
    print("Data directory: (set DATA_DIR env var to override)", data_dir)
    parquet_files = glob.glob(os.path.join(data_dir, 'foldseek_struct/*.parquet'))

    # Rename 'cluster_id' to 'fam_id' in each file
    process_parquet_files(parquet_files, old_column_name='cluster_id', new_column_name='fam_id')
