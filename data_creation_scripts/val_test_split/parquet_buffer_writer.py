import pandas as pd
import numpy as np
import os
import sys


class ParquetBufferWriter:
    def __init__(self, outdir, name, mem_limit=125, SGE_TASK_ID=None):
        self.outdir = outdir
        self.name = name
        self.mem_limit = mem_limit
        self.index = 0
        self.dfs = []
        self.mem_use = 0
        self.SGE_TASK_ID = SGE_TASK_ID


    def update_buffer(self, df):
        """
        Increment memory usage. If over the memory limit,
        write dataframes to parquet and reset buffer.
        """
        seqs_mb = self.get_size_mb(df)
        # print(f"mem use {self.name}: {round(self.mem_use, 6)}")
        if self.mem_use + seqs_mb < self.mem_limit:
            self.dfs.append(df)
            self.mem_use += seqs_mb
        else:
            self.write_dfs()
            self.dfs = [df]
            self.mem_use = seqs_mb


    def write_dfs(self):
        if self.dfs:
            combi_df = pd.concat(self.dfs)
            if self.SGE_TASK_ID is None:
                filepath = os.path.join(self.outdir, f"{self.name}_{str(self.index).zfill(3)}.parquet")
            else:
                filepath = os.path.join(self.outdir, f"{self.name}_{self.SGE_TASK_ID}_{str(self.index).zfill(3)}.parquet")
            print(f"Writing {len(combi_df)} rows to {filepath}")
            combi_df.to_parquet(filepath)
            self.dfs = []
            self.index += 1
            self.mem_use = 0

    @staticmethod
    def get_size_mb(df):
        n_cols = len(df.columns)
        total_bytes = sum([sys.getsizeof(s) for i, s_array in df.sequences.items() for s in s_array]) * n_cols
        total_mb = total_bytes / (1024 ** 2)
        # print(f"Size of DataFrame {len(df)}: {round(total_mb, 6)} MB")
        return total_mb