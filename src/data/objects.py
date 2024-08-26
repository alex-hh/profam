from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.data.fasta import read_fasta_lines


@dataclass
class Protein:
    sequence: str
    accession: str
    positions: Optional[List[int]] = None
    plddt: Optional[np.ndarray] = None
    backbone_coords: Optional[np.ndarray] = None
    structure_tokens: Optional[str] = None


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


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
@dataclass
class ProteinDocument:
    identifier: str
    sequences: List[str]
    accessions: List[str]
    positions: Optional[List[List[int]]] = None
    plddts: Optional[List[np.ndarray]] = None
    backbone_coords: Optional[List[np.ndarray]] = None
    structure_tokens: Optional[List[str]] = None

    @classmethod
    def from_fasta_str(cls, identifier: str, fasta_str: str):
        lines = fasta_str.split("\n")
        sequences = []
        accessions = []
        for accession, seq in read_fasta_lines(lines):
            sequences.append(seq)
            accessions.append(accession)
        return cls(identifier, sequences, accessions)

    def __post_init__(self):
        check_array_lengths(
            self.sequences,
            self.accessions,
            self.plddts,
            self.backbone_coords,
            self.structure_tokens,
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ProteinDocument(
                self.identifier,
                self.sequences[key],
                self.accessions[key],
                self.plddts[key] if self.plddts is not None else None,
                self.backbone_coords[key] if self.backbone_coords is not None else None,
                self.structure_tokens[key]
                if self.structure_tokens is not None
                else None,
            )
        elif isinstance(key, np.ndarray) or isinstance(key, list):
            return ProteinDocument(
                self.identifier,
                [self.sequences[i] for i in key],
                [self.accessions[i] for i in key],
                [self.plddts[i] for i in key] if self.plddts is not None else None,
                [self.backbone_coords[i] for i in key]
                if self.backbone_coords is not None
                else None,
                [self.structure_tokens[i] for i in key]
                if self.structure_tokens is not None
                else None,
            )
        elif isinstance(key, int):
            return Protein(
                self.sequences[key],
                self.accessions[key],
                self.plddts[key] if self.plddts is not None else None,
                self.backbone_coords[key] if self.backbone_coords is not None else None,
                self.structure_tokens[key]
                if self.structure_tokens is not None
                else None,
            )
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return len(self.sequences)

    @property
    def has_all_structure_arrays(self):
        return all(
            arr is not None
            for arr in [self.plddts, self.backbone_coords, self.structure_tokens]
        )

    def clone(self, **kwargs):
        return ProteinDocument(
            kwargs.get("identifier", self.identifier),
            kwargs.get("sequences", self.sequences),
            kwargs.get("accessions", self.accessions),
            kwargs.get("positions", self.positions),
            kwargs.get("plddts", self.plddts),
            kwargs.get("backbone_coords", self.backbone_coords),
            kwargs.get("structure_tokens", self.structure_tokens),
        )
