import json

import numpy as np

from evoif.common.constants import DATA_PATH, SPLITS_PATH
from evoif.gvp.data import CATHDataset

data = CATHDataset(DATA_PATH, SPLITS_PATH)
for split_name in ["train", "val", "test"]:
    split_entries = getattr(data, split_name)
    # save jsonl file for each split
    # (in gvp format directly)
    with open(f"data/cath_{split_name}.jsonl", "w") as file:
        for entry in split_entries:
            # convert coords bact to dict from tuple
            coords_dict = {}
            for ix, atom_type in enumerate(["N", "CA", "C", "O"]):
                coords_dict[atom_type] = [coords[ix] for coords in entry["coords"]]
            entry["coords"] = coords_dict
            file.write(json.dumps(entry) + "\n")