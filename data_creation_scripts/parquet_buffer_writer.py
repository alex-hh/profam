
import pandas as pd
import os
import sys


class ParquetBufferWriter:
    def __init__(self, outdir, name, mem_limit=250):
        self.outdir = outdir
        self.name = name
        self.mem_limit = mem_limit
        self.index = 0
        self.dfs = []
        self.mem_use = 0


    def update_buffer(self, df):
        """
        increment mem use, if over mem_limit
        write dfs to parquet and reset buffer
        """
        seqs_mb = sum([sys.getsizeof(s) for i, s_ls in df.sequences.items() for s in s_ls]) / 1024 / 1024
        print(f"mem use {self.name}: {round(self.mem_use, 6)}")
        if self.mem_use + seqs_mb < self.mem_limit:
            self.dfs.append(df)
            self.mem_use += seqs_mb

        else:
            self.write_dfs()
            self.dfs = [df]
            self.mem_use = seqs_mb


    def write_dfs(self):
        if len(self.dfs):
            combi_df = pd.concat(self.dfs)
            filepath = os.path.join(self.outdir, f"{self.name}_{str(self.index).zfill(3)}.parquet")
            print(f"writing to {filepath}")
            combi_df.to_parquet(filepath)
            self.dfs = []
            self.index += 1
            self.mem_use = 0