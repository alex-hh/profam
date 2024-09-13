import os
import re
import pickle
from src.constants import PROFAM_DATA_DIR
from .utils import make_zip_dictionary, make_af50_dictionary, make_cluster_dictionary, make_sequence_dictionary


# def make_sequence_dictionary(fasta_path):
#     """Should be <80 GB memory required to load all sequences;
#     considerably lower for just cluster representatives.

#     To handle larger files, we could partition the cluster dict
#     (or e.g. first process representatives then process the rest.)
#     """
#     sequence_dict = {}
#     with open(fasta_path, "r") as f:
#         lines = f.readlines()
#     for line in lines:
#         if line.startswith(">"):
#             # TODO: check this
#             uniprot_acc = re.search("UA=(\w+)", line).group(1)
#             sequence = next(f).strip()
#             sequence_dict[uniprot_acc] = sequence
#     print("Number of sequences in dictionary:", len(sequence_dict))
#     return sequence_dict


def main():
    af2zip = make_zip_dictionary()
    with open(os.path.join(PROFAM_DATA_DIR, "afdb", "af2zip.pkl"), "wb") as f:
        pickle.dump(af2zip, f)
    
    del af2zip

    af50_dict = make_af50_dictionary()
    with open(os.path.join(PROFAM_DATA_DIR, "afdb", "af50_cluster_dict.pkl"), "wb") as f:
        pickle.dump(af50_dict, f)

    del af50_dict

    cluster_dict = make_cluster_dictionary()
    with open(os.path.join(PROFAM_DATA_DIR, "afdb", "foldseek_cluster_dict.pkl"), "wb") as f:
        pickle.dump(cluster_dict, f)
    del cluster_dict

    sequence_dict = make_sequence_dictionary()
    with open(os.path.join(PROFAM_DATA_DIR, "afdb", "sequence_dict.pkl"), "wb") as f:
        pickle.dump(sequence_dict, f)


if __name__ == "__main__":
    main()
