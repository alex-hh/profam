import os
from pathlib import Path

from datasets.features import Array3D, Sequence, Value
from datasets.features.features import _ArrayXD

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
HAS_SOURCE_LAYOUT = (REPO_ROOT / "configs").is_dir() and (REPO_ROOT / "data").is_dir()
RUNTIME_ROOT = REPO_ROOT if HAS_SOURCE_LAYOUT else Path.cwd()
CONFIGS_DIR = REPO_ROOT / "configs" if HAS_SOURCE_LAYOUT else PACKAGE_DIR / "configs"
TOKENIZER_FILE = (
    REPO_ROOT / "data" / "profam_tokenizer.json"
    if (REPO_ROOT / "data" / "profam_tokenizer.json").is_file()
    else PACKAGE_DIR / "data" / "profam_tokenizer.json"
)

os.environ.setdefault("PROFAM_PACKAGE_DIR", str(PACKAGE_DIR))
os.environ.setdefault("PROFAM_CONFIGS_DIR", str(CONFIGS_DIR))
os.environ.setdefault("PROFAM_TOKENIZER_FILE", str(TOKENIZER_FILE))

BASEDIR = str(RUNTIME_ROOT)
BENCHMARK_RESULTS_DIR_NAME = "benchmark_results"
BENCHMARK_RESULTS_DIR = str(RUNTIME_ROOT / BENCHMARK_RESULTS_DIR_NAME)

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
aa_letters_lower = [aa.lower() for aa in aa_letters]

BACKBONE_ATOMS = ["N", "CA", "C", "O"]

PROFAM_DATA_DIR = os.environ.get(
    "PROFAM_DATA_DIR",
    str((REPO_ROOT / "data") if (REPO_ROOT / "data").is_dir() else Path.cwd() / "data"),
)


def resolve_runtime_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    for base_dir in (Path.cwd(), RUNTIME_ROOT):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / path).resolve()


# features whose first non-batch dim is equal to the number of residues
RESIDUE_LEVEL_FEATURES = [
    "input_ids",
    "attention_mask",
    "aa_mask",
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
]


TENSOR_FEATURES = SEQUENCE_TENSOR_FEATURES


SEQUENCE_FEATURE_NAMES = STRING_FEATURE_NAMES + SEQUENCE_TENSOR_FEATURES
ALL_FEATURE_NAMES = STRING_FEATURE_NAMES + TENSOR_FEATURES


TOKENIZED_FEATURE_TYPES = {
    "input_ids": Sequence(feature=Value(dtype="int32"), length=-1),
    "attention_mask": Sequence(feature=Value(dtype="int32"), length=-1),
    "labels": Sequence(feature=Value(dtype="int32"), length=-1),
    "original_size": Value(dtype="float32"),  # with sequence packing we use the mean
    "aa_mask": Sequence(feature=Value(dtype="bool"), length=-1),
    "ds_name": Value(dtype="string"),
    "identifier": Value(dtype="string"),
    "batch_size": Value(dtype="int32"),
}

ARRAY_TYPES = (Sequence, _ArrayXD)
