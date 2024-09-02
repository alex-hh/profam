from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.data import transforms
from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences
from src.data.objects import ProteinDocument
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


@dataclass
class BasePreprocessorConfig:
    preprocessor: str = ""  # TODO: handle this better
    keep_insertions: bool = False
    to_upper: bool = False
    keep_gaps: bool = False
    document_token: str = "[RAW]"
    truncate_after_n_sequences: Optional[int] = None
    use_msa_pos: bool = False  # for msa sequences, if true, position index will be relative to alignment cols
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/182
    transforms: Optional[
        List[Any]
    ] = None  # making callable raises an omegaconf validationerror: unsupported value type 'callable'
    keep_columns: Optional[List[str]] = None
    return_all_fields: bool = (
        False  # if true return default values for coords etc if not being used
    )
    allow_unk: bool = False


@dataclass
class NullPreprocessorConfig(BasePreprocessorConfig):
    def __post_init__(self):
        self.preprocessor = None


@dataclass
class FastaPreprocessorConfig(BasePreprocessorConfig):
    def __post_init__(self):
        self.preprocessor = "fasta"


@dataclass
class ParquetSequencePreprocessorConfig(BasePreprocessorConfig):
    sequence_col: str = "sequences"

    def __post_init__(self):
        self.preprocessor = "parquet_sequence"


# TODO: make sure we can handle an aligned version - test
@dataclass
class ParquetStructureTokensPreprocessorConfig(BasePreprocessorConfig):
    sequence_col: str = "sequences"
    structure_tokens_col: str = "structure_tokens"
    is_aligned: bool = False

    def __post_init__(self):
        self.preprocessor = "parquet_structure_tokens"


def subsample_fasta_lines(lines, n_lines, shuffle=True):
    start_ix = np.array([i for i, l in enumerate(lines) if l[0] == ">"])
    end_ix = start_ix[1:]
    end_ix = np.append(end_ix, len(lines))
    lines_per_seq = len(lines) // len(start_ix)
    n_samples = min(n_lines // lines_per_seq, len(start_ix))
    if shuffle:
        sample_indices = np.random.choice(len(start_ix), n_samples, replace=False)
    else:
        sample_indices = np.arange(n_samples)
    starts = start_ix[sample_indices]
    ends = end_ix[sample_indices]
    sampled_lines = []
    for start, end in zip(starts, ends):
        assert lines[end - 1][0] != ">"
        sampled_lines.extend(lines[start:end])
    return sampled_lines


def random_subsample(arr, n, seed: Optional[int] = None):
    rnd = np_random(seed)
    return rnd.choice(arr, min(n, len(arr)), replace=False)


def _tokenize_protein_data(
    proteins: ProteinDocument,
    cfg,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    padding="max_length",
):
    tokenized = tokenizer.encode(
        proteins,
        document_token=cfg.document_token,
        padding=padding,
        max_length=max_tokens,
        add_final_sep=True,
        allow_unk=getattr(cfg, "allow_unk", False),
    )
    # tokenized.input_ids is flat now
    # n.b. this is after subsampling so not very informative
    tokenized.data["total_num_sequences"] = len(proteins)  # below length threshold

    return tokenized.data  # a dict


def subsample_and_tokenize_protein_data(
    proteins: ProteinDocument,
    cfg: BasePreprocessorConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    padding: str = "max_length",
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    proteins = transforms.sample_to_max_tokens(
        proteins,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=seed,
    )
    if cfg.return_all_fields:
        proteins = transforms.fill_missing_fields(proteins)
    proteins = transforms.replace_selenocysteine_pyrrolysine(proteins)
    proteins = transforms.apply_transforms(cfg.transforms, proteins, tokenizer)

    tokenized = _tokenize_protein_data(
        proteins,
        cfg=cfg,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        padding=padding,
    )
    return tokenized


def preprocess_protein_sequences(
    proteins: ProteinDocument,
    cfg: BasePreprocessorConfig,
    tokenizer: ProFamTokenizer,
):
    # TODO: assert that structure tokens, coords, plddt are all same shape as sequences post conversion or handle if not
    if tokenizer.use_seq_pos:
        proteins = transforms.convert_sequences_adding_positions(
            proteins,
            keep_gaps=cfg.keep_gaps,
            keep_insertions=cfg.keep_insertions,
            to_upper=cfg.to_upper,
            use_msa_pos=cfg.use_msa_pos,
            truncate_after_n_sequences=cfg.truncate_after_n_sequences,
        )
    else:
        proteins = proteins[: cfg.truncate_after_n_sequences or len(proteins)]
    return proteins


def preprocess_fasta_data(
    example: Dict[str, Any],
    cfg: FastaPreprocessorConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, Any]:
    lines = example["text"].split("\n")
    if not len(lines[-1]):
        lines = lines[:-1]
    # min 2 lines per seq, assume at least 10 tks per line
    max_fasta_lines_to_preprocess = (
        max_tokens or 1e8
    ) // 5  # upper bound on lines to proc.
    if len(lines) > max_fasta_lines_to_preprocess:
        lines = subsample_fasta_lines(
            lines,
            max_fasta_lines_to_preprocess,
            shuffle=shuffle,
        )
    sequences = [
        seq
        for seq in read_fasta_sequences(
            lines,
            # preserve original sequences before getting positions
            keep_gaps=True if tokenizer.use_seq_pos else cfg.keep_gaps,
            keep_insertions=True if tokenizer.use_seq_pos else cfg.keep_insertions,
            to_upper=False if tokenizer.use_seq_pos else cfg.to_upper,
        )
    ]
    proteins = ProteinDocument(sequences=sequences)
    proteins = preprocess_protein_sequences(proteins, cfg, tokenizer)
    return subsample_and_tokenize_protein_data(
        proteins,
        cfg=cfg,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
    )


def backbone_coords_from_example(example, sequence_col="sequences"):
    ns = example["N"]
    cas = example["CA"]
    cs = example["C"]
    oxys = example["O"]
    coords = []
    for seq, n, ca, c, o in zip(
        example[sequence_col],
        ns,
        cas,
        cs,
        oxys,
    ):
        recons_coords = np.zeros((len(seq), 4, 3))
        recons_coords[:, 0] = np.array(n).reshape(-1, 3)
        recons_coords[:, 1] = np.array(ca).reshape(-1, 3)
        recons_coords[:, 2] = np.array(c).reshape(-1, 3)
        recons_coords[:, 3] = np.array(o).reshape(-1, 3)
        coords.append(recons_coords)
    return coords


def preprocess_parquet_with_structure_tokens(
    example: Dict[str, Any],
    cfg: ParquetStructureTokensPreprocessorConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, Any]:
    # TODO: configure whether or not to use alignments, structure tokens col, etc.
    max_sequences_to_preprocess = (max_tokens or 1e8) // 10
    sequence_iterator = example[cfg.sequence_col]
    structure_tokens_iterator = example[cfg.structure_tokens_col]
    if shuffle:
        sequence_ids = random_subsample(
            np.arange(len(sequence_iterator)),
            max_sequences_to_preprocess,
        )
    else:
        sequence_ids = np.arange(
            min(max_sequences_to_preprocess, len(sequence_iterator))
        )
    sequences = [sequence_iterator[i] for i in sequence_ids]
    # we assume sequence processing and structure token processing are consistent.
    # later we will check that everything ends up the same length - which is important
    # because otherwise incorrect config could easily lead to misalignment
    structure_tokens = [
        convert_sequence_with_positions(
            structure_tokens_iterator[i],
            keep_gaps=cfg.keep_gaps,
            keep_insertions=cfg.keep_insertions,
            to_upper=cfg.to_upper,
        )[0].lower()
        for i in sequence_ids
    ]
    if "N" in example and not cfg.is_aligned:
        assert not any(["-" in seq for seq in sequences]) and not any(
            ["-" in seq for seq in structure_tokens]
        )
        coords = backbone_coords_from_example(example, sequence_col=cfg.sequence_col)
        coords = [coords[i] for i in sequence_ids]
        plddts = example["plddts"]
        plddts = [plddts[i] for i in sequence_ids]
    else:
        # TODO: support aligned coords, plddts
        coords = None
        plddts = None

    proteins = ProteinDocument(
        sequences=sequences,
        plddts=plddts,
        backbone_coords=coords,
        structure_tokens=structure_tokens,
    )
    proteins = preprocess_protein_sequences(proteins, cfg, tokenizer)
    return subsample_and_tokenize_protein_data(
        proteins,
        cfg=cfg,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
    )


def preprocess_parquet_sequence_data(
    example: Dict[str, Any],
    cfg: ParquetSequencePreprocessorConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, Any]:
    sequence_iterator = example[cfg.sequence_col]
    max_sequences_to_preprocess = max_tokens // 10
    # n.b. this also shuffles
    if shuffle:
        sequences = random_subsample(
            sequence_iterator,
            max_sequences_to_preprocess,
        )
    else:
        sequences = sequence_iterator[:max_sequences_to_preprocess]

    proteins = ProteinDocument(sequences=sequences)
    proteins = preprocess_protein_sequences(proteins, cfg, tokenizer)
    return subsample_and_tokenize_protein_data(
        proteins,
        cfg=cfg,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
    )


def get_preprocessor(preprocessor: str):
    if preprocessor == "fasta":
        return preprocess_fasta_data
    elif preprocessor == "parquet_sequence":
        return preprocess_parquet_sequence_data
    elif preprocessor == "parquet_structure_tokens":
        return preprocess_parquet_with_structure_tokens
    else:
        raise ValueError(f"Unknown preprocessor {preprocessor}")


def preprocess_protein_data(
    example: Dict[str, Any],
    cfg: BasePreprocessorConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, Any]:
    # N.B. for stockholm format we need to check that sequences aren't split over
    # multiple lines
    tokenized = get_preprocessor(cfg.preprocessor)(
        example, cfg, tokenizer, max_tokens=max_tokens, shuffle=shuffle
    )
    return tokenized
