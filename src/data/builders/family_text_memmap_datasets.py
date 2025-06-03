import glob
import os
from typing import Any, List, Optional

import numpy as np
from torch.utils.data import Dataset

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from ..text_memmap_datasets import TextMemMapDataset
from .base import BaseProteinDataset


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
        for base_idx, fn_path in zip([0] + list(self.lines_ds.midx_bins), self.lines_ds._files_list):
            fn = os.path.basename(fn_path)
            self._file_to_base_idx[fn] = base_idx


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

    def get_absolute_indices(self, fn, indices):
        """
        Get the absolute index of the sequence in the dataset given relative index and file name.
        """
        # get the base index for the file
        base_idx = self._file_to_base_idx[fn]
        # return the absolues index
        return [idx + (base_idx // 2) for idx in indices]

    def get_sequence_sizes(self):
        """
        Compute and return the number of tokens in each sequence without loading sample data.
        Uses np.diff for efficient computation.
        
        Returns:
            List[int]: A list with the number of tokens for each sequence.
        """
        sizes = []
        for _, midx in self.lines_ds.mdata_midx_list:
            # diff minus one to exclude newline char
            sizes.extend(map(int, np.diff(midx)[::2] - 1))
        return sizes


class ProteinFamilyMemmapDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_root: str,
        preprocessor: ProteinDocumentPreprocessor,
        tokenizer: ProFamTokenizer,
        **kwargs,
    ):
        """
        Args:
            name: name of the dataset
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            tokenizer: tokenizer to use to convert sequences to tokens.
            kwargs: additional arguments to pass to the dataset
        """
        super().__init__(name=name)
        self.name = name
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
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
        pass

    def __len__(self):
        return len(self.mapping_ds)

    def __getitem__(self, idx):
        mapping_data = self.mapping_ds[idx]
        sequence_indices = []
        # collect samples from all files
        for fn, indices in mapping_data["sample_indices"].items():
            # project each relative index to absolute index
            sequence_indices.extend(self.sequences_ds.get_absolute_indices(fn, indices))

        # TODO: add sampling of sequences from a family here
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

class ProteinFamilyMemmapDatasetBuilder(ProteinFamilyMemmapDataset, BaseProteinDataset):
    """A builder that wraps :class:`ProteinFamilyMemmapDataset` so it can be used
    interchangeably with existing HF-style dataset builders inside
    :class:`ProteinDataMixture`.  The builder inherits from both
    ``ProteinFamilyMemmapDataset`` (so it *is* a PyTorch dataset) and
    ``BaseProteinDataset`` (so it exposes the ``load``/``process`` interface
    expected by the trainer).

    No additional preprocessing is required because the underlying
    ``ProteinFamilyMemmapDataset`` already performs tokenisation on-the-fly via
    the supplied ``ProteinDocumentPreprocessor``.
    """

    def __init__(
        self,
        name: str,
        dataset_root: str,
        preprocessor: ProteinDocumentPreprocessor,
        tokenizer: ProFamTokenizer,
        **kwargs,
    ):
        # Initialise BaseProteinDataset first (sets attributes used elsewhere)
        BaseProteinDataset.__init__(self, name=name, preprocessor=preprocessor)

        # Now initialise the actual mem-mapped dataset
        ProteinFamilyMemmapDataset.__init__(
            self,
            name=name,
            dataset_root=dataset_root,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Keep a reference to root for relative path resolution in load()
        self._dataset_root = dataset_root

    # ------------------------------------------------------------------
    # The following methods implement the *builder* API expected by
    # ProteinDataMixture.  Because the dataset is already constructed in
    # __init__, these mostly do nothing or are simple wrappers.
    # ------------------------------------------------------------------

    def load(self, data_dir: str = "data", world_size: int = 1, verbose: bool = False):
        """Return the constructed dataset.  If ``dataset_root`` was given as a
        relative path we resolve it with respect to ``data_dir`` the first
        time ``load`` is called.  This mirrors the behaviour of the HF
        builders which resolve file patterns relative to *data_dir*.
        """

        # # Resolve the path only once (important when called on every GPU)
        # if not os.path.isabs(self._dataset_root):
        #     abs_root = os.path.join(data_dir, self._dataset_root)
        #     # If the resolved path is different, rebuild the internal dataset
        #     if abs_root != self._dataset_root:
        #         self._dataset_root = abs_root
        #         ProteinFamilyMemmapDataset.__init__(
        #             self,
        #             name=self.name,
        #             dataset_root=self._dataset_root,
        #             preprocessor=self.preprocessor,
        #             tokenizer=self.tokenizer,
        #         )

        return self  # the dataset itself

    def process(
        self,
        dataset: Any,
        tokenizer: ProFamTokenizer,
        feature_names: Optional[List[str]] = None,
        pack_to_max_tokens: Optional[int] = None,
    ):
        """No offline processing necessary – everything happens on-the-fly.

        The method exists purely to satisfy the builder interface.
        """

        # No dataset-level packing; packing handled in collator ring buffer
        return dataset  # already tokenised on access

    # We never call _build_document for this builder.
    def _build_document(self, example):  # pragma: no cover – defensive only
        raise NotImplementedError(
            "ProteinFamilyMemmapDatasetBuilder performs preprocessing internally and does not use _build_document()."
        )
