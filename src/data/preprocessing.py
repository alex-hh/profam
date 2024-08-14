import hashlib
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizerFast

from src.data.dataclasses import ProteinDatasetConfig
from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences
from src.data.utils import get_seq_pos_from_positions, sample_to_max_tokens
from src.utils.utils import np_random


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
    return rnd.choice(arr, n, replace=False)


def _tokenize_protein_data(
    sequences: List[str],
    cfg: ProteinDatasetConfig,
    tokenizer,
    positions: Optional[List[List[int]]] = None,
    coords: Optional[List[np.ndarray]] = None,
    plddts: Optional[List[np.ndarray]] = None,
):
    # TODO: configure num_start_tokens (property of tokenizer perhaps? maybe we make a profam tokenizer class)
    concatenated_seqs = (
        cfg.document_tag
        + tokenizer.bos_token
        + tokenizer.sep_token.join(sequences)
        + tokenizer.sep_token
    )
    tokenized = tokenizer(
        concatenated_seqs,
        truncation=False,  # shouldnt be necessary: bisection should handle
        max_length=cfg.max_tokens,
        return_tensors="pt",
        # padding="longest",
        padding="max_length",
        add_special_tokens=False,
    )
    if cfg.max_tokens is not None:
        assert tokenized.input_ids.shape[1] <= cfg.max_tokens, (
            tokenized.input_ids.shape[1],
            cfg.max_tokens,
        )

    tokenized.data = {k: v.squeeze() for k, v in tokenized.data.items()}
    # tokenized.input_ids is flat now
    tokenized.data["ds_name"] = cfg.name
    tokenized.data["total_num_sequences"] = len(sequences)  # below length threshold

    if positions is not None:
        seq_pos = get_seq_pos_from_positions(
            tokenized.input_ids,
            positions,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_pos=cfg.max_seq_pos,
            num_start_tokens=2,
        )
        tokenized.data["seq_pos"] = seq_pos

    if coords is not None:
        # TODO: if cfg.load_structure is True maybe create null coords?
        tokenized.data["coords"] = coords
    if plddts is not None:
        tokenized.data["plddts"] = plddts
    return tokenized.data  # a dict


def _subsample_and_tokenize_protein_data(
    sequence_iterator,
    cfg: ProteinDatasetConfig,
    tokenizer: PreTrainedTokenizerFast,
    coords: Optional[List[np.ndarray]] = None,
    plddts: Optional[List[np.ndarray]] = None,
):
    if cfg.use_seq_pos:
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

    extra_arrays = [positions, coords, plddts]
    sequences, extra_arrays = sample_to_max_tokens(
        sequences,
        extra_arrays=extra_arrays,
        max_tokens=cfg.max_tokens,
        shuffle=cfg.shuffle,
    )
    positions, coords, plddts = extra_arrays
    tokenized = _tokenize_protein_data(
        sequences,
        cfg=cfg,
        tokenizer=tokenizer,
        positions=positions,
        coords=coords,
        plddts=plddts,
    )
    if cfg.include_doc_hashes:
        # identify documents by a hash of the first 512 characters
        tokenized["doc_hash"] = hashlib.md5(example["text"][:512].encode()).hexdigest()
    return tokenized


def preprocess_fasta_data(
    example: Dict[str, Any],
    cfg: ProteinDatasetConfig,
    tokenizer: PreTrainedTokenizerFast,
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
        keep_gaps=True if cfg.use_seq_pos else cfg.keep_gaps,
        keep_insertions=True if cfg.use_seq_pos else cfg.keep_insertions,
        to_upper=False if cfg.use_seq_pos else cfg.to_upper,
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
        recons_coords[:, 0] = n.reshape(-1, 3)
        recons_coords[:, 1] = ca.reshape(-1, 3)
        recons_coords[:, 2] = c.reshape(-1, 3)
        recons_coords[:, 3] = o.reshape(-1, 3)
        coords.append(recons_coords)
    return coords


def preprocess_parquet_data(
    example: Dict[str, Any],
    cfg: ProteinDatasetConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> Dict[str, Any]:
    sequence_iterator = example["sequences"]
    max_sequences_to_preprocess = cfg.max_tokens // 10
    # n.b. this also shuffles
    sequence_iterator = random_subsample(
        sequence_iterator,
        max_sequences_to_preprocess,
    )
    if "N" in example:
        coords = backbone_coords_from_example(example)
        plddts = example["plddts"]
    else:
        coords = None
        plddts = None
    return _subsample_and_tokenize_protein_data(
        sequence_iterator,
        cfg=cfg,
        tokenizer=tokenizer,
        coords=coords,
        plddts=plddts,
    )


def preprocess_protein_data(
    example: Dict[str, Any],
    cfg: ProteinDatasetConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> Dict[str, Any]:
    # N.B. for stockholm format we need to check that sequences aren't split over
    # multiple lines
    if "sequences" in example:
        return preprocess_parquet_data(example, cfg, tokenizer)
    else:
        return preprocess_fasta_data(example, cfg, tokenizer)
