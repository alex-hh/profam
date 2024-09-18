import os

BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BENCHMARK_RESULTS_DIR_NAME = "benchmark_results"
BENCHMARK_RESULTS_DIR = os.path.join(BASEDIR, BENCHMARK_RESULTS_DIR_NAME)


aa_letters = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


PROFAM_DATA_DIR = os.environ.get(
    "PROFAM_DATA_DIR", "/SAN/orengolab/cath_plm/ProFam/data"
)


# features whose first non-batch dim is equal to the number of residues
RESIDUE_LEVEL_FEATURES = [
    "input_ids",
    "attention_mask",
    "seq_pos",
    "backbone_coords",
    "backbone_coords_masks",
    "plddts",
    "suffix_masks",
    "seq_pos",
    "token_type_ids",
]
