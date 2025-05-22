from typing import List, Optional, Any

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from .base import BaseProteinDataset
from ..text_memmap_datasets import TextMemMapDataset
import glob
import os
from torch.utils.data import Dataset

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

class SequencesProteinFamilyMemmapDataset(TextMemMapDataset):
    """
    A *.sequences FASTA dataset, holding accession and sequence for all families.
    """
    def __init__(
        self,
        dataset_root: str,
        workers=None,
        tokenizer=None,
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
        
        # build mapping from file name to base index to support relative indices for each sequences file
        self._file_to_base_idx = {}
        for base_idx, fn_path in zip([0] + list(self.midx_bins), self._files_list):
            fn = os.path.basename(fn_path)
            self._file_to_base_idx[fn] = base_idx

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize sequences
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        text_fields = text.split(self._data_sep)

        data = {
            "accession": text_fields[0].strip(),
            # tokenize sequence if tokenizer is provided
            "sequence": _build_data_from_text(text_fields[1].strip()),
        }

        return data
    
    def get_absolute_indices(self, fn, indices):
        """
        Get the relative index of the sequence in the dataset.
        """
        # get the base index for the file
        base_idx = self._file_to_base_idx[fn]
        # return the absolues index
        return [idx + base_idx for idx in indices]

class ProteinFamilyMemmapDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_root: str,
        tokenizer: Optional[ProFamTokenizer] = None,
        **kwargs,
    ):
        """
        Args:
            name: name of the dataset
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            tokenizer: tokenizer to use to convert sequences to tokens.
            kwargs: additional arguments to pass to the dataset
        """
        super().__init__()
        self.name = name
        
        self.mapping_ds = MappingProteinFamilyMemmapDataset(
            dataset_root=dataset_root,
            **kwargs,
        )
        self.sequences_ds = SequencesProteinFamilyMemmapDataset(
            dataset_root=dataset_root,
            # only sequence dataset requires a tokenizer
            tokenizer=tokenizer,
            **kwargs,
        )
        
    def __len__(self):
        return len(self.mapping_ds)

    def __getitem__(self, idx):
        mapping_data = self.mapping_ds[idx]
        sequences_data = []
        # collect samples from all files
        for fn, indices in mapping_data["sample_indices"].items():
            # project each relative index to absolute index
            sequence_indices = self.sequences_ds.get_absolute_indices(fn, indices)
            sequences_data.extend([self.sequences_ds[i] for i in sequence_indices])
        
        # TODO: add sampling of sequences from a family here
        return ProteinDocument(
            sequences=[sd["sequence"] for sd in sequences_data],
            identifier=mapping_data["fam_id"],
            accessions=[sd["accession"] for sd in sequences_data],
        )

# FIXME: need to finish implementing the dataset builder
class ProteinFamilyMemmapDatasetBuilder(BaseProteinDataset):
    def __init__(
        self,
        name: str,
        dataset_root: str,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        tokenizer: Optional[ProFamTokenizer] = None,
        **kwargs,
    ):
        """
        Args:
            name: name of the dataset
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            preprocessor: preprocessor to use for tokenization
            tokenizer: tokenizer to use to convert sequences to tokens.
            kwargs: additional arguments to pass to the dataset
        """
        super().__init__(
            name=name,
            preprocessor=preprocessor,
        )
        
        self.protein_family_ds = ProteinFamilyMemmapDataset(
            name=name,
            dataset_root=dataset_root,
            tokenizer=tokenizer,
            **kwargs,
        )

    def process(
        self,
        dataset: Any,
        tokenizer: ProFamTokenizer,
    ):
        # nothing to do here since tokenization is done on-the-fly
        return dataset

    def load(self, data_dir="data", world_size: int = 1, verbose: bool = False):
        # Nothing to do here since loading is very fast
        return self

    def _build_document(self, example):
        # private method has fixed signature; static methods can have variable signature
        return ProteinDocument(
            sequences=sequences,
            original_size=len(lines) // 2,
            identifier=identifier,
        )  # upper bound estimate of number of sequences
