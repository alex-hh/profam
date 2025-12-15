import io
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, ClassVar, List, Optional

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from biotite import structure as struc
from biotite.sequence import ProteinSequence
from biotite.structure import io as strucio

from src.constants import BACKBONE_ATOMS
from src.sequence.fasta import read_fasta_lines


# copying here to avoid circular imports
def _superimpose_np(reference, coords):
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [N, 3] reference array
        coords:
            [N, 3] array
    Returns:
        A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()


def plddt_to_color(plddt):
    if plddt > 90:
        return "#0053D6"
    elif plddt > 70:
        return "#65CBF3"
    elif plddt > 50:
        return "#FFDB13"
    else:
        return "#FF7D45"


class StringObject:
    """
    Custom class to allow for
    non-tensor elements in batch
    """

    text: List[str]

    def to(self, device):
        return self


@dataclass
class Protein:
    sequence: str
    accession: Optional[str] = None

    def __len__(self):
        return len(self.sequence)

    def clone(self, **kwargs):
        return Protein(
            sequence=kwargs.get("sequence", self.sequence),
            accession=kwargs.get("accession", self.accession),
        )

    def slice_arrays(self, slice_or_indices):
        return Protein(
            sequence=self.sequence[slice_or_indices]
            if isinstance(slice_or_indices, slice)
            else "".join([self.sequence[i] for i in slice_or_indices]),
            accession=self.accession,
        )


def convert_list_of_arrays_to_list_of_lists(list_of_arrays):
    if list_of_arrays is None:
        return None
    elif isinstance(list_of_arrays[0], np.ndarray):
        return [arr.tolist() for arr in list_of_arrays]
    else:
        return list_of_arrays


@dataclass
class ProteinDocument:
    # TODO: make this a mapping?
    # fields that are present on individual protein instances
    protein_fields: ClassVar[List[str]] = [
        "sequences",
        "accessions",
        "sequence_similarities",
        "coverages",
        "sequence_weights",
    ]
    sequences: List[str]
    accessions: Optional[List[str]] = None
    identifier: Optional[str] = None
    representative_accession: Optional[
        str
    ] = None  # e.g. seed or cluster representative
    original_size: Optional[int] = None  # total number of proteins in original set
    # Per-sequence coverage and similarity data against WT sequence
    sequence_similarities: Optional[List[float]] = None
    coverages: Optional[List[float]] = None
    sequence_weights: Optional[List[float]] = None

    def __post_init__(self):
        for field in [
            "sequence_weights",
        ]:
            attr = getattr(self, field)
            if attr is not None and isinstance(attr[0], list):
                setattr(self, field, [np.array(arr) for arr in getattr(self, field)])

    def __len__(self):
        return len(self.sequences)

    @property
    def sequence_lengths(self):
        return [len(seq) for seq in self.sequences]

    def present_fields(self, residue_level_only: bool = False):
        if residue_level_only:
            return [
                field
                for field in self.protein_fields
                if getattr(self, field) is not None
            ]
        else:
            return [
                field
                for field in self.__dataclass_fields__.keys()
                if getattr(self, field) is not None
            ]

    @classmethod
    def from_proteins(cls, individual_proteins: List[Protein], **kwargs):
        # N.B. we ignore representative_accession here
        renaming = {
            "sequence": "sequences",
            "accession": "accessions",
        }
        reverse_naming = {v: k for k, v in renaming.items()}
        attr_dict = {}
        for field in cls.protein_fields:
            single_field = reverse_naming.get(field, field)
            if any(getattr(p, single_field) is not None for p in individual_proteins):
                assert all(
                    getattr(p, single_field) is not None for p in individual_proteins
                ), f"Missing {single_field} for some proteins"
                attr_dict[field] = [
                    getattr(p, single_field) for p in individual_proteins
                ]
            else:
                attr_dict[field] = None
        return cls(
            **attr_dict,
            **kwargs,
        )

    @classmethod
    def from_json(cls, json_file, strict: bool = False):
        with open(json_file, "r") as f:
            protein_dict = json.load(f)

        if strict:
            assert all(
                field in protein_dict for field in cls.__dataclass_fields__.keys()
            ), f"Missing fields in {json_file}"
        return cls(**protein_dict)

    def to_json(self, json_file):
        with open(json_file, "w") as f:
            protein_dict = {
                k: convert_list_of_arrays_to_list_of_lists(v)
                for k, v in asdict(self).items()
            }
            json.dump(protein_dict, f)

    @property
    def representative(self):  # use as target for e.g. inverse folding evaluations
        assert self.representative_accession is not None
        rep_index = self.accessions.index(self.representative_accession)
        return self[rep_index]

    def pop_representative(self):
        assert self.representative_accession is not None
        representative_index = self.accessions.index(self.representative_accession)
        return self.pop(representative_index)

    def filter(self, filter_fn: Callable):
        """Filter by filter_fn.

        Filter_fn should take a protein and return True if it should be kept.
        """
        indices = [i for i in range(len(self)) if filter_fn(self[i])]
        return self[indices]

    def pop(self, index):
        return Protein(
            sequence=self.sequences.pop(index),
            accession=self.accessions.pop(index)
            if self.accessions is not None
            else None,
        )

    @classmethod
    def from_fasta_str(cls, identifier: str, fasta_str: str):
        lines = fasta_str.split("\n")
        sequences = []
        accessions = []
        for accession, seq in read_fasta_lines(lines):
            sequences.append(seq)
            accessions.append(accession)
        return cls(identifier, sequences, accessions)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ProteinDocument(
                identifier=self.identifier,
                sequences=self.sequences[key],
                accessions=self.accessions[key]
                if self.accessions is not None
                else None,
                sequence_similarities=self.sequence_similarities[key]
                if self.sequence_similarities is not None
                else None,
                coverages=self.coverages[key] if self.coverages is not None else None,
                sequence_weights=self.sequence_weights[key]
                if self.sequence_weights is not None
                else None,
                representative_accession=self.representative_accession,
                original_size=self.original_size,
            )
        elif isinstance(key, np.ndarray) or isinstance(key, list):
            assert len(key) > 0, "Empty key"
            return ProteinDocument(
                identifier=self.identifier,
                sequences=[self.sequences[i] for i in key],
                accessions=[self.accessions[i] for i in key]
                if self.accessions is not None
                else None,
                representative_accession=self.representative_accession,
                original_size=self.original_size,
                sequence_similarities=[self.sequence_similarities[i] for i in key]
                if self.sequence_similarities is not None
                else None,
                coverages=[self.coverages[i] for i in key]
                if self.coverages is not None
                else None,
                sequence_weights=[self.sequence_weights[i] for i in key]
                if self.sequence_weights is not None
                else None,
            )
        elif isinstance(key, int):
            return Protein(
                sequence=self.sequences[key],
                accession=self.accessions[key] if self.accessions is not None else None,
            )
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def slice_arrays(self, slices):
        assert len(slices) == len(self.sequences)
        return ProteinDocument(
            identifier=self.identifier,
            sequences=[seq[s] for seq, s in zip(self.sequences, slices)],
            accessions=self.accessions,
            representative_accession=self.representative_accession,
            original_size=self.original_size,
            # Per-sequence scalar fields are unchanged by per-residue slicing
            sequence_similarities=self.sequence_similarities.copy()
            if self.sequence_similarities is not None
            else None,
            coverages=self.coverages.copy() if self.coverages is not None else None,
            sequence_weights=self.sequence_weights.copy()
            if self.sequence_weights is not None
            else None,
        )

    def __len__(self):
        return len(self.sequences)

    def clone(self, **kwargs):
        return ProteinDocument(
            identifier=kwargs.pop("identifier", self.identifier),
            sequences=kwargs.pop("sequences", self.sequences.copy()),
            accessions=kwargs.pop(
                "accessions",
                self.accessions.copy() if self.accessions is not None else None,
            ),
            representative_accession=kwargs.pop(
                "representative_accession", self.representative_accession
            ),
            original_size=kwargs.pop("original_size", self.original_size),
            sequence_similarities=kwargs.pop(
                "sequence_similarities",
                self.sequence_similarities.copy()
                if self.sequence_similarities is not None
                else None,
            ),
            coverages=kwargs.pop(
                "coverages",
                self.coverages.copy() if self.coverages is not None else None,
            ),
            sequence_weights=kwargs.pop(
                "sequence_weights",
                self.sequence_weights.copy()
                if self.sequence_weights is not None
                else None,
            ),
            **kwargs,
        )

    def extend(self, proteins: "ProteinDocument"):
        # n.b. extend may be a bad name as this is not in place
        constructor_kwargs = {}
        for field in self.present_fields(residue_level_only=True):
            attr = getattr(self, field)
            if isinstance(attr, list):
                constructor_kwargs[field] = attr + getattr(proteins, field)
            elif isinstance(attr, np.ndarray):
                constructor_kwargs[field] = np.concatenate(
                    [attr, getattr(proteins, field)]
                )
            else:
                raise ValueError(f"Unexpected type: {field} {type(attr)}")
        if self.original_size is not None and proteins.original_size is not None:
            constructor_kwargs["original_size"] = (
                self.original_size + proteins.original_size
            )
        return ProteinDocument(**constructor_kwargs)

    def truncate_single(self, index: int, start: int, end: int):
        """
        Truncate the sequence and associated fields at the given index from start to end indices.

        Args:
            index (int): The index of the sequence to truncate.
            start (int): The starting index of the truncation.
            end (int): The ending index of the truncation.
        """
        self.sequences[index] = self.sequences[index][start:end]
