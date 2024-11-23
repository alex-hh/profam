import argparse
import os
import json
import glob
import pandas as pd
from data_creation_scripts.parquet_buffer_writer import ParquetBufferWriter

"""
For TED and funfams splits use this json:
profam/data/val_test/topology_splits.json

for foldseek use this json:
profam/data/val_test/foldseek_cath_topology_splits.json


foldseek_cath_topology_splits.json:
{
    "train": [
        "A0A7K0JYB7",
        "A0A061NIN8",
        ...
        ],
    "validation": [...],
    "test": [...]
}

######## TED parquet ########:
df.iloc[0]['fam_id']
'3.30.342.10'

>>> df.iloc[0]['accessions']
array(['AF-A0A2E4ZF72-F1-model_v4_TED01',
       'AF-A0A6V8CLE5-F1-model_v4_TED01',
############################


######## funfams parquet ########:
>>> df.iloc[0].fam_id
'2.20.28.290-FF-000002_rep_seq'

>>> df.iloc[0].accessions
array(['AF-A0A182UD23-F1-model_v4_TED02',
       'AF-A0A182VZ20-F1-model_v4_TED02',       
############################

######## FSeek parquet ########:
>>> df.iloc[0].fam_id
'A0A7C4GD14'

>>> df.iloc[0].accessions
array(['A0A2E0X6R3', 'A0A518GGD7',
############################

"""
class BaseParquetSplitter:
    def __init__(self, json_path, parquet_dir, output_dir, mem_limit=125):
        self.json_path = json_path
        self.parquet_dir = parquet_dir
        self.output_dir = output_dir
        self.mem_limit = mem_limit
        self.load_splits()

    def load_splits(self):
        """
        Load the train, validation, and test family IDs from the JSON file.
        """
        with open(self.json_path, 'r') as f:
            splits = json.load(f)
        self.train_fam_ids = set(splits.get('train', []))
        self.val_fam_ids = set(splits.get('validation', []))
        self.test_fam_ids = set(splits.get('test', []))
        self.all_fam_ids = self.train_fam_ids.union(self.val_fam_ids, self.test_fam_ids)

    def reformat_fam_id(self, fam_id):
        """
        Maps the parquet fam_id to the format used in the split JSON.
        To be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def split_parquets(self):
        """
        Iterate through all parquet files and assign each row to the appropriate split
        based on the family IDs.
        """
        # Create output directories
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        test_dir = os.path.join(self.output_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_buffer = ParquetBufferWriter(train_dir, name="train", mem_limit=self.mem_limit)
        val_buffer = ParquetBufferWriter(val_dir, name="val", mem_limit=self.mem_limit)
        test_buffer = ParquetBufferWriter(test_dir, name="test", mem_limit=self.mem_limit)

        # Process each parquet file
        for parquet_file in glob.glob(os.path.join(self.parquet_dir, '*.parquet')):
            df = pd.read_parquet(parquet_file)
            # Apply reformat_fam_id to fam_id column
            df['split_fam_id'] = df['fam_id'].apply(self.reformat_fam_id)

            # Split the DataFrame based on split_fam_id
            train_df = df[df['split_fam_id'].isin(self.train_fam_ids)].drop(columns=['split_fam_id'])
            val_df = df[df['split_fam_id'].isin(self.val_fam_ids)].drop(columns=['split_fam_id'])
            test_df = df[df['split_fam_id'].isin(self.test_fam_ids)].drop(columns=['split_fam_id'])

            # Update buffers
            if not train_df.empty:
                train_buffer.update_buffer(train_df)
            if not val_df.empty:
                val_buffer.update_buffer(val_df)
            if not test_df.empty:
                test_buffer.update_buffer(test_df)

        # Write any remaining data in buffers
        train_buffer.write_dfs()
        val_buffer.write_dfs()
        test_buffer.write_dfs()

class CATHParquetSplitter(BaseParquetSplitter):
    def reformat_fam_id(self, fam_id):
        """
        Maps parquet fam_id to the CATH topology ID used in the split JSON.
        Example: '3.30.342.10-FF-100001' -> '3.30.342'
        """
        return ".".join(fam_id.split(".")[:3])

class FoldSeekParquetSplitter(BaseParquetSplitter):
    def reformat_fam_id(self, fam_id):
        """
        For FoldSeek splits, the fam_id in the parquet files matches the IDs in the split JSON.
        """
        return fam_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the data into train, val, and test sets.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the train/val/test IDs.",
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        required=True,
        help="Path to the directory containing the parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where the split data will be saved.",
    )
    parser.add_argument(
        "--splitter",
        type=str,
        required=True,
        choices=['CATH', 'FoldSeek'],
        help="Type of ParquetSplitter to use ('CATH' or 'FoldSeek').",
    )
    parser.add_argument(
        "--mem_limit",
        type=int,
        default=125,
        help="Memory limit (in MB) for the ParquetBufferWriter.",
    )

    args = parser.parse_args()

    # Map the splitter choice to the corresponding class
    if args.splitter == 'CATH':
        splitter_class = CATHParquetSplitter
    elif args.splitter == 'FoldSeek':
        splitter_class = FoldSeekParquetSplitter
    else:
        raise ValueError(f"Unknown splitter type: {args.splitter}")

    splitter = splitter_class(
        json_path=args.json_path,
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        mem_limit=args.mem_limit
    )

    splitter.split_parquets()
