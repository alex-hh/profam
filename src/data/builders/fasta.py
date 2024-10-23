from typing import List, Optional

import numpy as np

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer
from src.sequence.fasta import read_fasta_sequences

from .hf_datasets import HFProteinDatasetConfig, MemoryMappedHFProteinDataset


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
        min_sequences: Optional[int] = None,
        holdout_identifiers: Optional[List[str]] = None,
        tokenizer: ProFamTokenizer = None,
    ):
        super_filter = super().filter_fn(example)
        if super_filter:
            assert (
                holdout_identifiers is None
            ), "Holdout identifiers not supported for fasta"
            filter_num_seqs = len(example["text"].split("\n")) // 2 >= (
                min_sequences or 1
            )
            return filter_num_seqs
        return False

    @staticmethod
    def build_document(
        text,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        max_sequences: Optional[int] = None,
        identifier: Optional[str] = None,
    ):
        lines = text.split("\n")
        if not len(lines[-1]):
            lines = lines[:-1]
        # rough upper bound: min 2 lines per seq, assume at least 10 tks per line
        max_fasta_lines_to_preprocess = (
            (max_tokens or 1e8) // 5 if max_sequences is None else max_sequences * 50
        )
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

    def _build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        if isinstance(example, str):
            return self.build_document(
                example,
                max_tokens,
                shuffle,
                max_sequences=self.cfg.max_sequences_per_document,
            )
        else:
            return self.build_document(
                example["text"],
                max_tokens,
                shuffle,
                max_sequences=self.cfg.max_sequences_per_document,
                identifier=self.name + "/" + example[self.cfg.identifier_col]
                if self.cfg.identifier_col is not None
                else None,
            )
