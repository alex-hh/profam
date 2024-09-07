import os
import pickle
from src.constants import PROFAM_DATA_DIR
from .utils import make_zip_dictionary, make_af50_dictionary, make_cluster_dictionary


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
    with open(os.path.join(PROFAM_DATA_DIR, "foldseek_cluster_dict.pkl"), "wb") as f:
        pickle.dump(cluster_dict, f)


if __name__ == "__main__":
    main()
