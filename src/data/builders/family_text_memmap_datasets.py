from typing import List, Optional

import numpy as np

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from .hf_datasets import HFProteinDatasetConfig, MemoryMappedHFProteinDataset


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
