import glob
import os
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
from torch.utils.data import Dataset

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from ..text_memmap_datasets import TextMemMapDataset


class MappingProteinFamilyMemmapDataset(TextMemMapDataset):
    """
    A *.mapping FASTA dataset, holding family id and mapping of sequences files and corresponding indices (per file), for each family.
    """

    def __init__(
        self,
        dataset_root: str,
        workers=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            workers: number of workers to use for parallel data indexing (on first run)
            sort_dataset_paths: whether to sort dataset paths by name
            index_mapping_dir: directory to store index mapping cached files
        """
        dataset_paths = glob.glob(f"{dataset_root}/*.mapping")
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=ord(">"),
            header_lines=1,  # skip first line since it is an empty sequence
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
            line = line.strip()
            if not line:
                continue
            seq_fname, seq_ind = line.split(":")
            seq_ind = [int(i) for i in seq_ind.split(",")]
            sample_indices[seq_fname] = seq_ind

        data = {
            "fam_id": fam_id,
            "sample_indices": sample_indices,
        }

        return data


class SequencesProteinFamilyMemmapDataset(Dataset):
    """
    A *.sequences FASTA dataset, holding accession and sequence for all families.
    We treat each line in the *.sequences files independently even though every 2 lines create a sample with accession + sqeuence. We do so to be able to read sequence size efficiently.
    """

    def __init__(
        self,
        dataset_root: str,
        workers=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            workers: number of workers to use for parallel data indexing (on first run)
            tokenizer: tokenizer to use for tokenization
            sort_dataset_paths: whether to sort dataset paths by name
            index_mapping_dir: directory to store index mapping cached files
        """
        dataset_paths = glob.glob(f"{dataset_root}/*.sequences")
        # We read the sequences files as text lines, so we can use TextMemMapDataset
        self.lines_ds = TextMemMapDataset(
            dataset_paths=dataset_paths,
            newline_int=ord("\n"),
            header_lines=0,  # no header lines in sequences files
            workers=workers,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

        if len(self.lines_ds) % 2 != 0:
            raise ValueError(
                "The number of lines in the sequences files must be even (each sequence has an accession and a sequence line)."
            )

        # build mapping from file name to base index to support relative indices for each sequences file
        self._file_to_base_idx = {}
        for base_idx, fn_path in zip(
            [0] + list(self.lines_ds.midx_bins), self.lines_ds._files_list
        ):
            fn = os.path.basename(fn_path)
            self._file_to_base_idx[fn] = base_idx

        # build mapping from file name to file index to support fast access to each sequences file
        self._file_to_file_idx = {}
        for file_idx, fn_path in enumerate(self.lines_ds._files_list):
            fn = os.path.basename(fn_path)
            self._file_to_file_idx[fn] = file_idx

    def __len__(self):
        """Return the number of sequences in the dataset."""
        # Each sequence is represented by 2 lines (accession and sequence)
        return len(self.lines_ds) // 2

    def __getitem__(self, idx):
        """Return the sequence and its accession for the given index."""
        # Get the text lines for the accession and sequence
        accession_line = self.lines_ds[idx * 2]
        sequence_line = self.lines_ds[idx * 2 + 1]

        # Build data from text lines
        data = {
            # skip the first character (">") in the accession line
            "accession": accession_line[1:].strip(),
            "sequence": sequence_line.strip(),
        }

        return data

    def get_global_sequence_indices(self, fn, local_indices):
        """
        Get the absolute index of the sequence in the dataset given relative index and file name.
        """
        # get the base index for the file
        base_idx = self._file_to_base_idx[fn]
        # return the absolues index
        return [idx + (base_idx // 2) for idx in local_indices]

    def get_sequence_sizes(self, fn: str, local_indices: list):
        """
        Compute and return the number of tokens in each sequence without reading the full sequence data.
        This function uses numpy.diff to efficiently compute the difference between newline positions,
        subtracting one to exclude newline characters. If a list of indices is provided, the sizes for
        those specific sequences are returned. Otherwise, sizes for all sequences are computed.

        Args:
            fn (str): The file name to compute sizes for.
            local_indices (Optional[List[int]]): Specific sequence indices to compute sizes for. Defaults to None.

        Returns:
            List[int]: A list containing the token counts for each sequence.
        """
        sizes = []
        file_dx = self._file_to_file_idx[fn]
        _, midx = self.lines_ds.mdata_midx_list[file_dx]
        # return sizes for the given indices
        for idx in local_indices:
            sizes.append(midx[idx * 2 + 1] - midx[idx * 2] - 1)

        return sizes


class ProteinFamilyMemmapDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_root: str,
        preprocessor: ProteinDocumentPreprocessor,
        tokenizer: ProFamTokenizer,
        max_tokens_per_family: Optional[
            int
        ] = None,  # CAUTION: caching results in same sequences being sampled from the family across epoch, we recommend setting max_tokens_per_example in the preprocessor instead
        max_families: Optional[int] = None,
        shuffle_family_sequences: bool = True,
        sample_cache_size: int = 1000,
        seed: Optional[int] = 1,
        **kwargs,
    ):
        """
        Args:
            name: name of the dataset
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            tokenizer: tokenizer to use to convert sequences to tokens.
            max_families: maximum number of families to use (useful for validation)
            kwargs: additional arguments to pass to the dataset
        """
        super().__init__()
        self.name = name
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.max_tokens_per_family = max_tokens_per_family
        self.max_families = max_families
        self.shuffle_family_sequences = shuffle_family_sequences
        self.sample_cache_size = sample_cache_size
        self.seed = seed
        self.mapping_ds = MappingProteinFamilyMemmapDataset(
            dataset_root=dataset_root,
            # make sure order of files is deterministic
            sort_dataset_paths=True,
            **kwargs,
        )
        self.sequences_ds = SequencesProteinFamilyMemmapDataset(
            dataset_root=dataset_root,
            # make sure order of files is deterministic
            sort_dataset_paths=True,
            **kwargs,
        )

        self.local_rng = np.random.RandomState(seed=self.seed)

        # NOTE: MAKE __getitem__ A CACHED FUNCTION!!!
        # We need it since sampler will also load samples from the dataset to compute samples size.
        if self.sample_cache_size is not None and self.sample_cache_size > 0:
            self.__getitem__ = lru_cache(maxsize=self.sample_cache_size, typed=False)(
                self.__getitem__
            )

    def __len__(self):
        length = len(self.mapping_ds)
        if self.max_families is not None:
            length = min(length, self.max_families)
        return length

    def __getitem__(self, idx):
        mapping_data = self.mapping_ds[idx]
        sequence_indices = []
        sequence_sizes = []
        # collect samples from all files
        for fn, indices in mapping_data["sample_indices"].items():
            # get size of each sequenece in the file
            sequence_sizes.extend(self.sequences_ds.get_sequence_sizes(fn, indices))
            # project each relative index to absolute index
            sequence_indices.extend(
                self.sequences_ds.get_global_sequence_indices(fn, indices)
            )

        # randomize order of sequences within a family
        if self.shuffle_family_sequences:
            family_idx = list(range(len(sequence_indices)))
            self.local_rng.shuffle(family_idx)

        # Limit tokens per family if specified
        if self.max_tokens_per_family is not None:
            cur_tokens = 0
            for cur_family_i, cur_family_idx in enumerate(family_idx):
                cur_tokens += sequence_sizes[cur_family_idx]
                if cur_tokens > self.max_tokens_per_family:
                    family_idx = family_idx[:cur_family_i]
                    break

        # reorder and subset the family sequences
        sequence_indices = [sequence_indices[i] for i in family_idx]
        # get the actual sequence data for the selected indices
        sequences_data = [self.sequences_ds[i] for i in sequence_indices]
        protein_doc = ProteinDocument(
            sequences=[sd["sequence"] for sd in sequences_data],
            identifier=mapping_data["fam_id"],
            accessions=[sd["accession"] for sd in sequences_data],
        )
        processed = self.preprocessor.preprocess_protein_data(
            protein_doc,
            tokenizer=self.tokenizer,
        )
        processed["ds_name"] = self.name
        return processed
