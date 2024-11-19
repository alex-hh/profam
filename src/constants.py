import os

from datasets.features import Array3D, Sequence, Value
from datasets.features.features import _ArrayXD

BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BENCHMARK_RESULTS_DIR_NAME = "benchmark_results"
BENCHMARK_RESULTS_DIR = os.path.join(BASEDIR, BENCHMARK_RESULTS_DIR_NAME)

VOCAB_SIZE = 68

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

BACKBONE_ATOMS = ["N", "CA", "C", "O"]

PROFAM_DATA_DIR = os.environ.get("PROFAM_DATA_DIR", os.path.join(BASEDIR, "data"))


# features whose first non-batch dim is equal to the number of residues
RESIDUE_LEVEL_FEATURES = [
    "input_ids",
    "attention_mask",
    "residue_index",
    "coords",
    "coords_mask",
    "plddts",
    "plddt_mask",
    "aa_mask",
    "token_type_ids",
]

STRING_FEATURE_NAMES = [
    "ds_name",
    "identifier",
]

SEQUENCE_TENSOR_FEATURES = [
    "input_ids",
    "attention_mask",
    # "labels",  # added by collator
    "original_size",
    "residue_index",
]


STRUCTURE_TENSOR_FEATURES = [
    "coords",
    "coords_mask",
    "interleaved_coords_mask",
    "aa_mask",
    "plddts",
    "structure_mask",
]

TENSOR_FEATURES = SEQUENCE_TENSOR_FEATURES + STRUCTURE_TENSOR_FEATURES


SEQUENCE_FEATURE_NAMES = STRING_FEATURE_NAMES + SEQUENCE_TENSOR_FEATURES
ALL_FEATURE_NAMES = STRING_FEATURE_NAMES + TENSOR_FEATURES


TOKENIZED_FEATURE_TYPES = {
    "coords": Array3D(dtype="float32", shape=(None, 4, 3)),
    "plddts": Sequence(feature=Value(dtype="float32"), length=-1),
    "input_ids": Sequence(feature=Value(dtype="int32"), length=-1),
    "attention_mask": Sequence(feature=Value(dtype="int32"), length=-1),
    "labels": Sequence(feature=Value(dtype="int32"), length=-1),
    "residue_index": Sequence(feature=Value(dtype="int32"), length=-1),
    "original_size": Value(dtype="int32"),
    "aa_mask": Sequence(feature=Value(dtype="bool"), length=-1),
    "structure_mask": Sequence(feature=Value(dtype="bool"), length=-1),
    "interleaved_coords_mask": Array3D(dtype="bool", shape=(None, 4, 3)),
    "coords_mask": Array3D(dtype="bool", shape=(None, 4, 3)),
    "ds_name": Value(dtype="string"),
    "identifier": Value(dtype="string"),
}

ARRAY_TYPES = (Sequence, _ArrayXD)
