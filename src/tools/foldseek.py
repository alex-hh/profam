import os
import subprocess

import pandas as pd


def convert_pdbs_to_3di(pdb_files, output_file):
    cmd = ["foldseek", "structureto3didescriptor"] + pdb_files + [output_file]
    filenames = [os.path.basename(f) for f in pdb_files]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"FoldMason stdout: {result.stdout}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"FoldMason execution failed: {e}", flush=True)
        print(f"FoldMason stderr: {e.stderr}", flush=True)
        raise
    mapping = pd.read_csv(output_file, sep="\t", names=["pdb", "aa", "3di", "ca"]).set_index("pdb")
    return mapping.loc[filenames]["3di"].tolist()
