from typing import List, Optional

import numpy as np

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer
from src.sequence.fasta import read_fasta_sequences

from .hf_datasets import (
    HFProteinDatasetConfig,
    IterableHFProteinDataset,
    MemoryMappedHFProteinDataset,
)


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


# TODO: infer identifier from different column
class FastaProteinDataset(MemoryMappedHFProteinDataset):
    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            required_keys=["text"],
        )

    def filter_fn(
        self,
        example,
        tokenizer: ProFamTokenizer,
    ):
        super_filter = super().filter_fn(example, tokenizer=tokenizer)
        if super_filter:
            assert (
                self.cfg.holdout_identifiers is None
            ), "Holdout identifiers not supported for fasta"
            filter_num_seqs = len(example["text"].split("\n")) // 2 >= (
                self.cfg.minimum_sequences or 1
            )
            return filter_num_seqs
        return False

    @staticmethod
    def build_document(
        text,
        max_sequences: Optional[int] = None,
        identifier: Optional[str] = None,
    ):
        lines = text.split("\n")
        if not len(lines[-1]):
            lines = lines[:-1]
        # rough upper bound: min 2 lines per seq, assume at least 10 tks per line
        max_fasta_lines_to_preprocess = (
            max_sequences * 50 if max_sequences is not None else len(lines)
        )
        if len(lines) > max_fasta_lines_to_preprocess:
            lines = subsample_fasta_lines(
                lines,
                max_fasta_lines_to_preprocess,
                shuffle=False,
            )

        sequences = [
            seq
            for seq in read_fasta_sequences(
                lines,
                # preserve original sequences before further preprocessing
                keep_gaps=True,
                keep_insertions=True,
                to_upper=False,
            )
        ]

        return ProteinDocument(
            sequences=sequences,
            original_size=len(lines) // 2,
            identifier=identifier,
        )  # upper bound estimate of number of sequences

    def _build_document(self, example):
        if isinstance(example, str):
            return self.build_document(
                example,
            )
        else:
            return self.build_document(
                example["text"],
                max_sequences=self.cfg.max_sequences_per_document,
                identifier=self.name + "/" + example[self.cfg.identifier_col]
                if self.cfg.identifier_col is not None
                else self.name + "/None",  # avoid Nones
            )


# -----------------------------------------------------------------------------
# Iterable dataset version of FastaProteinDataset
# -----------------------------------------------------------------------------


class FastaProteinIterableDataset(IterableHFProteinDataset):
    """Iterable (streaming) version of :class:`FastaProteinDataset`.

    equivalent in behaviour to ``FastaProteinDataset`` but utilises the
    :class:`IterableHFProteinDataset` base so that the HuggingFace ``datasets``
    library streams the underlying FASTA text files instead of memory-mapping
    them. Necessary to add fasta dataset (such as Protein Gym MSAs)
    to the combined interleaved training dataset.
    """

    # We reuse the static ``build_document`` implementation from the map style
    # dataset to avoid duplicating code.
    build_document = staticmethod(FastaProteinDataset.build_document)

    def __init__(
        self,
        name: str,
        cfg: HFProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
    ):
        # IterableHFProteinDataset expects ``required_keys`` to flag the columns
        # that must be present in every example; for FASTA text files this is
        # just the "text" column.
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            required_keys=["text"],
        )

    # Filtering logic is identical to the map-style dataset
    def filter_fn(
        self,
        example,
        tokenizer: ProFamTokenizer,
    ):
        super_filter = super().filter_fn(example, tokenizer=tokenizer)
        if super_filter:
            assert (
                self.cfg.holdout_identifiers is None
            ), "Holdout identifiers not supported for fasta"
            filter_num_seqs = len(example["text"].split("\n")) // 2 >= (
                self.cfg.minimum_sequences or 1
            )
            return filter_num_seqs
        return False

    def _build_document(self, example):
        if isinstance(example, str):
            return self.build_document(example)
        else:
            return self.build_document(
                example["text"],
                max_sequences=self.cfg.max_sequences_per_document,
                identifier=self.name + "/" + example[self.cfg.identifier_col]
                if self.cfg.identifier_col is not None
                else self.name + "/None",  # avoid Nones
            )
