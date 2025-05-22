from typing import List, Optional

import numpy as np

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from .hf_datasets import HFProteinDatasetConfig, MemoryMappedHFProteinDataset
from ..text_memmap_datasets import TextMemMapDataset

class MappingProteinFamilyMemmapDataset(TextMemMapDataset):
    """
    A *.mapping FASTA dataset.
    """
    def __init__(
        self,
        dataset_paths,
        workers=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_paths: list of paths to text files
            workers: number of workers to use for parallel data indexing (on first run)
            sort_dataset_paths: whether to sort dataset paths by name
            index_mapping_dir: directory to store index mapping cached files
        """
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=ord(">"),
            header_lines=1,  # skip first line since it is not an empty sequence
            workers=workers,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )
        
        self._data_sep = "\n"        

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize sequences
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        text_fields = text.split(self._data_sep)

        fam_id = text_fields[0]
        sample_indices = {}
        for line in text_fields[1:]:
            seq_fname, seq_ind = line.trim().split(":")
            seq_ind = [int(i) for i in seq_ind.split(",")]
            sample_indices[seq_fname] = seq_ind
            
        data = {
            "fam_id": fam_id,
            "sample_indices": sample_indices,
        }

        return data

class SequencesProteinFamilyMemmapDataset(TextMemMapDataset):
    """
    A *.mapping FASTA dataset.
    """
    def __init__(
        self,
        dataset_paths,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_paths: list of paths to text files
            workers: number of workers to use for parallel data indexing (on first run)
            tokenizer: tokenizer to use for tokenization
            sort_dataset_paths: whether to sort dataset paths by name
            index_mapping_dir: directory to store index mapping cached files
        """
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=ord(">"),
            header_lines=1,  # skip first line since it is not an empty sequence
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )
        
        self._data_sep = "\n"        

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize sequences
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        text_fields = text.split(self._data_sep)

        data = {
            "accession": text_fields[0],
            # tokenize sequence if tokenizer is provided
            "sequence": _build_data_from_text(text_fields[1]),
        }

        return data


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
