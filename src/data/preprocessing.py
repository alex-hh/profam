import bisect
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


@dataclass
class BasePreprocessorConfig:
    preprocessor: str = ""  # TODO: handle this better
    keep_insertions: bool = False
    to_upper: bool = False
    keep_gaps: bool = False
    document_token: str = "[RAW]"
    max_tokens: Optional[int] = 5000
    shuffle: bool = True
    truncate_after_n_sequences: Optional[int] = None
    use_msa_pos: bool = True  # for msa sequences, if true, position index will be relative to alignment cols


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
    interleave_structure_tokens: bool = False  # when would we want false?
    is_aligned: bool = False

    def __post_init__(self):
        self.preprocessor = "parquet_structure_tokens"


def sample_to_max_tokens(
    sequences,
    extra_arrays: Optional[List[List[Any] | np.ndarray]] = None,
    max_tokens: Optional[int] = None,
    shuffle=True,
    seed: Optional[int] = None,
    drop_first: bool = False,
    extra_tokens_per_sequence: int = 1,  # just [SEP] for default sequences (this includes eos)
    extra_tokens_per_document: int = 2,
):
    rng = np.random.default_rng(seed)
    # TODO: implement keep first, drop first
    if drop_first:
        sequences = sequences[1:]
        if extra_arrays is not None:
            extra_arrays = [
                arr[1:] if arr is not None else None for arr in extra_arrays
            ]

    if shuffle:
        perm = rng.permutation(len(sequences))
        sequences = [sequences[i] for i in perm]
        if extra_arrays is not None:
            extra_arrays = [
                [arr[i] for i in perm] if arr is not None else None
                for arr in extra_arrays
            ]

    if max_tokens is not None:
        cumulative_lengths = list(
            itertools.accumulate(
                [len(s) + extra_tokens_per_sequence for s in sequences]
            )
        )  # +1 for separator
        insertion_point = bisect.bisect_left(
            cumulative_lengths,
            max_tokens - extra_tokens_per_document,
        )  # -2 for doc start tokens
    else:
        insertion_point = len(sequences)
    if extra_arrays is None:
        return sequences[:insertion_point]
    else:
        return sequences[:insertion_point], [
            arr[:insertion_point] if arr is not None else None for arr in extra_arrays
        ]


def check_array_lengths(*arrays):  # TODO: name better!
    sequence_lengths = []
    for arr in arrays:
        if arr is None:
            continue
        else:
            sequence_lengths.append(tuple([len(seq) for seq in arr]))

    assert all(
        l == sequence_lengths[0] for l in sequence_lengths
    ), f"{sequence_lengths} not all equal"
    return sequence_lengths


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
    sequences: List[str],
    cfg,
    tokenizer: ProFamTokenizer,
    positions: Optional[List[List[int]]] = None,
    coords: Optional[List[np.ndarray]] = None,
    plddts: Optional[List[np.ndarray]] = None,
):
    print("Tokenizing sequences", sequences)
    tokenized = tokenizer.encode_sequences(
        sequences,
        positions=positions,
        document_token=cfg.document_token,
        padding="max_length",
        max_length=cfg.max_tokens,
        coords=coords,
        plddts=plddts,
        add_final_sep=True,
    )
    # tokenized.input_ids is flat now
    tokenized.data["ds_name"] = cfg.name
    tokenized.data["total_num_sequences"] = len(sequences)  # below length threshold

    return tokenized.data  # a dict


def _subsample_and_tokenize_protein_data(
    sequence_iterator,
    cfg,
    tokenizer: ProFamTokenizer,
    coords: Optional[List[np.ndarray]] = None,
    plddts: Optional[List[np.ndarray]] = None,
    structure_tokens: Optional[List[str]] = None,
):
    # TODO: assert that structure tokens, coords, plddt are all same shape as sequences post conversion or handle if not
    if tokenizer.use_seq_pos:
        sequences = []
        positions = []
        for seq in itertools.islice(sequence_iterator, cfg.truncate_after_n_sequences):
            seq, pos, _ = convert_sequence_with_positions(
                seq,
                keep_gaps=cfg.keep_gaps,
                keep_insertions=cfg.keep_insertions,
                to_upper=cfg.to_upper,
            )
            sequences.append(seq)
            positions.append(pos)
    else:
        sequences = [
            seq
            for seq in itertools.islice(
                sequence_iterator, cfg.truncate_after_n_sequences
            )
        ]  # necessary for fasta iterator...
        positions = None

    extra_arrays = [positions, coords, plddts, structure_tokens]
    sequences, extra_arrays = sample_to_max_tokens(
        sequences,
        extra_arrays=extra_arrays,
        # TODO: we need to subtract the cost of the extra sep tokens also.
        max_tokens=cfg.max_tokens // 2
        if cfg.interleave_structure_tokens
        else cfg.max_tokens,
        shuffle=cfg.shuffle,
        extra_tokens_per_sequence=2 if cfg.interleave_structure_tokens else 1,
        extra_tokens_per_document=tokenizer.num_start_tokens,
    )
    positions, coords, plddts, structure_tokens = extra_arrays

    check_array_lengths(sequences, positions, coords, plddts, structure_tokens)
    if cfg.interleave_structure_tokens:
        sequences = [
            seq_3d + tokenizer.seq_struct_sep_token + seq
            for seq, seq_3d in zip(sequences, structure_tokens)
        ]
        coords = [
            np.concatenate([xyz, np.full((1, 4, 3), np.nan), xyz], axis=0)
            for xyz in coords
        ]
        assert isinstance(plddts[0], list)
        assert isinstance(positions[0], list)
        positions = [pos + [0] + pos for pos in positions]
        plddts = [vals + [np.nan] + vals for vals in plddts]

    tokenized = _tokenize_protein_data(
        sequences,
        cfg=cfg,
        tokenizer=tokenizer,
        positions=positions,
        coords=coords,
        plddts=plddts,
    )
    return tokenized


def preprocess_fasta_data(
    example: Dict[str, Any],
    cfg: FastaPreprocessorConfig,
    tokenizer: ProFamTokenizer,
) -> Dict[str, Any]:
    lines = example["text"].split("\n")
    if not len(lines[-1]):
        lines = lines[:-1]
    # min 2 lines per seq, assume at least 10 tks per line
    max_fasta_lines_to_preprocess = cfg.max_tokens // 5  # upper bound on lines to proc.
    if len(lines) > max_fasta_lines_to_preprocess:
        lines = subsample_fasta_lines(
            lines,
            max_fasta_lines_to_preprocess,
            shuffle=cfg.shuffle,
        )
    sequence_iterator = read_fasta_sequences(
        lines,
        # preserve original sequences before getting positions
        keep_gaps=True if tokenizer.use_seq_pos else cfg.keep_gaps,
        keep_insertions=True if tokenizer.use_seq_pos else cfg.keep_insertions,
        to_upper=False if tokenizer.use_seq_pos else cfg.to_upper,
    )
    return _subsample_and_tokenize_protein_data(
        sequence_iterator,
        cfg=cfg,
        tokenizer=tokenizer,
    )


def backbone_coords_from_example(example):
    ns = example["N"]
    cas = example["CA"]
    cs = example["C"]
    oxys = example["O"]
    coords = []
    for seq, n, ca, c, o in zip(
        example["sequences"],
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
) -> Dict[str, Any]:
    assert cfg.is_parquet
    # TODO: configure whether or not to use alignments, structure tokens col, etc.
    max_sequences_to_preprocess = cfg.max_tokens // 10
    sequence_iterator = example[cfg.sequence_col]
    structure_tokens_iterator = example[cfg.structure_tokens_col]
    if cfg.shuffle:
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
        coords = backbone_coords_from_example(example)
        coords = [coords[i] for i in sequence_ids]
        plddts = example["plddts"]
        plddts = [plddts[i] for i in sequence_ids]
    else:
        # TODO: support aligned coords, plddts
        coords = None
        plddts = None

    return _subsample_and_tokenize_protein_data(
        sequences,
        cfg=cfg,
        tokenizer=tokenizer,
        coords=coords,
        plddts=plddts,
        structure_tokens=structure_tokens,
    )


def preprocess_parquet_sequence_data(
    example: Dict[str, Any],
    cfg: ParquetSequencePreprocessorConfig,
    tokenizer: ProFamTokenizer,
) -> Dict[str, Any]:
    assert cfg.is_parquet
    sequence_iterator = example["sequences"]
    max_sequences_to_preprocess = cfg.max_tokens // 10
    # n.b. this also shuffles
    if cfg.shuffle:
        sequences = random_subsample(
            sequence_iterator,
            max_sequences_to_preprocess,
        )
    else:
        sequences = sequence_iterator[:max_sequences_to_preprocess]
    return _subsample_and_tokenize_protein_data(
        sequences,
        cfg=cfg,
        tokenizer=tokenizer,
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
) -> Dict[str, Any]:
    # N.B. for stockholm format we need to check that sequences aren't split over
    # multiple lines
    tokenized = get_preprocessor(cfg.preprocessor)(example, cfg, tokenizer)
    return tokenized
