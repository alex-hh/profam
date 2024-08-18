from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.data.fasta import read_fasta_lines


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
@dataclass
class ProteinDocument:
    identifier: str
    sequences: List[str]
    accessions: List[str]
    plddts: Optional[List[float]] = None
    backbone_coords: Optional[np.ndarray] = None
    prompt_indices: Optional[
        List[int]
    ] = None  # indices of sequences selected for prompt

    @classmethod
    def from_fasta_str(cls, identifier: str, fasta_str: str):
        lines = fasta_str.split("\n")
        sequences = []
        accessions = []
        for accession, seq in read_fasta_lines(lines):
            sequences.append(seq)
            accessions.append(accession)
        return cls(identifier, sequences, accessions)
