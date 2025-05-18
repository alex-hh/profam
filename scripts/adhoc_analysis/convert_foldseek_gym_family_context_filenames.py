"""
Original filenames for the foldseek .a2m files
which are a stand-in for proteinGym context were
named wrong so we are now renaming them for backwards 
compatibility
"""

import pandas as pd
import os
import glob

df = pd.read_csv("../data/ProteinGym/DMS_substitutions.csv")
foldseek_files = glob.glob("../data/ProteinGym/foldseek_s50_DMS_msa_files/*.a2m")

for foldseek_file in foldseek_files:
    # Get the filename without the path
    filename = os.path.basename(foldseek_file)
    if filename in df.MSA_filename.values:
        new_string = '"'+filename.split(".")[0]+'",'
        print(new_string)
        continue
    matched_row  = df[df.DMS_id==filename.split(".")[0]]
    assert len(matched_row) == 1, f"Expected 1 row, got {len(matched_row)} for {filename}"
    matched_row = matched_row.iloc[0]
    new_name = f"{matched_row.MSA_filename}"
    os.rename(foldseek_file, os.path.join("../data/ProteinGym/foldseek_s50_DMS_msa_files", new_name))
    new_string = '"'+new_name.split(".")[0]+'",'
    print(new_string)

