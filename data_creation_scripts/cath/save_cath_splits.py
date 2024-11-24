import json
import os

from src.constants import BASEDIR
from src.data.builders.cath import load_cath43_coords, load_cath42_coords


def main():
    for split_name in ["train", "validation", "test"]:
        entries = load_cath43_coords(split_name=split_name)
        with open(os.path.join(BASEDIR, f"data/cath/cath43/{split_name}.jsonl"), "w") as f:
            for entry in entries:
                coords = entry.pop("coords")
                entry["N"] = coords["N"]
                entry["CA"] = coords["CA"]
                entry["C"] = coords["C"]
                entry["O"] = coords["O"]
                f.write(json.dumps(entry) + "\n")

        entries = load_cath42_coords(split_name=split_name)
        with open(
            os.path.join(BASEDIR, f"data/cath/cath42/{split_name}.jsonl"), "w"
        ) as f:
            for entry in entries:
                coords = entry.pop("coords")
                entry["N"] = coords["N"]
                entry["CA"] = coords["CA"]
                entry["C"] = coords["C"]
                entry["O"] = coords["O"]
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
