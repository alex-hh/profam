import copy
import json
from dataclasses import asdict, dataclass
from typing import Callable, ClassVar, Dict, List, Optional

import numpy as np

from src.data.fasta import read_fasta_lines


class StringObject:
    """
    Custom class to allow for
    non-tensor elements in batch
    """

    text: List[str]

    def to(self, device):
        return self


# TODO: how would we extend to include ligands or complexes?
# maybe the protein would have a ligands attribute
# or we could make a MolecularAssembly class, which could have a list of proteins and ligands
@dataclass
class Protein:
    array_shapes: ClassVar[Dict] = {
        "plddt": (None,),
        "backbone_coords": (None, 4, 3),
        "backbone_coords_mask": (None, 4, 3),
    }
    sequence: str
    accession: Optional[str] = None
    positions: Optional[List[int]] = None
    plddt: Optional[np.ndarray] = None
    backbone_coords: Optional[np.ndarray] = None
    backbone_coords_mask: Optional[np.ndarray] = None
    structure_tokens: Optional[str] = None

    def __len__(self):
        assert len(self.sequence) == len(self.plddt)
        return len(self.sequence)

    def __post_init__(self):
        # TODO: check types
        check_array_lengths(
            [self.sequence],
            [self.plddt] if self.plddt is not None else None,
            [self.backbone_coords] if self.backbone_coords is not None else None,
            [self.backbone_coords_mask]
            if self.backbone_coords_mask is not None
            else None,
            [self.structure_tokens] if self.structure_tokens is not None else None,
        )
        if self.backbone_coords_mask is None and self.backbone_coords is not None:
            self.backbone_coords_mask = np.where(
                np.isnan(self.backbone_coords),
                np.zeros_like(self.backbone_coords),
                np.ones_like(self.backbone_coords),
            )

    @property
    def null_fields(self):
        fields = []
        for k, v in asdict(self).items():
            if v is None:
                fields.append(k)
        return set(fields)


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


def convert_list_of_arrays_to_list_of_lists(list_of_arrays):
    if list_of_arrays is None:
        return None
    elif isinstance(list_of_arrays[0], np.ndarray):
        return [arr.tolist() for arr in list_of_arrays]
    else:
        return list_of_arrays


class BaseProteinDocument:
    protein_field_mapping = {
        "sequence": "sequences",
        "accession": "accessions",
        "plddts": "plddts",
        "backbone_coords": "backbone_coords",
        "backbone_coords_mask": "backbone_coords_masks",
        "structure_tokens": "structure_tokens",
    }
    metadata_fields = [
        "identifier",
        "representative_accession",
        "original_size",
    ]

    def __init__(
        self,
        identifier: Optional[str] = None,
        representative_accession: Optional[str] = None,
        original_size: Optional[int] = None,
    ):
        self.identifier = identifier
        self.representative_accession = representative_accession
        self.original_size = original_size

    @property
    def metadata(self):
        return {field: getattr(self, field) for field in self.metadata_fields}

    @property
    def representative(self):  # use as target for e.g. inverse folding evaluations
        assert self.representative_accession is not None
        rep_index = self.accessions.index(self.representative_accession)
        return self[rep_index]

    def pop_representative(self):
        assert self.representative_accession is not None
        representative_index = self.accessions.index(self.representative_accession)
        return self.pop(representative_index)

    def to_json(self, json_file):
        with open(json_file, "w") as f:
            proteins_dict = self.to_dict(
                include_metadata=True, convert_arrays_to_lists=True
            )
            json.dump(proteins_dict, f)


# want to be consistent with fields in parquet files so we can load from there
# TODO: look into how openai evals uses data classes or similar
# TODO: consider how to represent masks
class ProteinDocument(BaseProteinDocument):
    metadata_fields = BaseProteinDocument.metadata_fields + ["suffix_masks"]

    def __init__(
        self,
        proteins: List[Protein],
        suffix_masks: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proteins = proteins
        self.suffix_masks = suffix_masks

    def __post_init__(self):
        assert all(
            prot.null_fields == self.proteins[0].null_fields for prot in self.proteins
        )
        if self.suffix_masks is not None:
            check_array_lengths(self.sequences, self.suffix_masks)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, key):
        return self.proteins[key]

    def __len__(self):
        return len(self.proteins)

    def __iter__(self):
        for i in range(len(self)):
            yield self.proteins[i]

    @classmethod
    def from_dict(cls, proteins_dict):
        proteins = []
        keys = list(proteins_dict.keys())
        inverse_field_mapping = {v: k for k, v in cls.protein_field_mapping.items()}
        protein_keys = (
            key
            for key in keys
            if inverse_field_mapping[key] in Protein.__dataclass_fields__.keys()
        )
        metadata_keys = (key for key in keys if key not in protein_keys)
        metadata = {key: proteins_dict.pop(key) for key in metadata_keys}

        num_proteins = len(proteins_dict[keys[0]])
        for i in range(num_proteins):
            single_protein_dict = {
                inverse_field_mapping[key]: proteins_dict[key][i] for key in keys
            }
            proteins.append(Protein(**single_protein_dict))
        return cls(proteins, **metadata)

    @classmethod
    def from_fields(cls, **kwargs):
        return cls.from_dict(kwargs)

    @classmethod
    def from_json(cls, json_file, strict: bool = False):
        with open(json_file, "r") as f:
            protein_dict = json.load(f)

        if strict:
            assert all(
                field in protein_dict for field in cls.__dataclass_fields__.keys()
            ), f"Missing fields in {json_file}"
        return cls.from_dict(protein_dict)

    @classmethod
    def from_fasta_str(cls, identifier: str, fasta_str: str):
        lines = fasta_str.split("\n")
        proteins = []
        for accession, seq in read_fasta_lines(lines):
            proteins.append(Protein(sequence=seq, accession=accession))
        return cls(proteins, identifier=identifier)

    def _attr_list(self, attr):
        if attr in self.null_fields:
            return None
        else:
            inverse_field_mapping = {
                v: k for k, v in self.protein_field_mapping.items()
            }
            attr_list = [
                getattr(protein, inverse_field_mapping[attr])
                for protein in self.proteins
            ]
            assert not any(a is None for a in attr_list)
            return attr_list

    @property
    def sequences(self):
        return [protein.sequence for protein in self.proteins]

    @property
    def accessions(self):
        return self._attr_list("accessions")

    @property
    def positions(self):
        return self._attr_list("positions")

    @property
    def plddts(self):
        return self._attr_list("plddts")

    @property
    def backbone_coords(self):
        return self._attr_list("backbone_coords")

    @property
    def backbone_coords_masks(self):
        return self._attr_list("backbone_coords_masks")

    @property
    def structure_tokens(self):
        return self._attr_list("structure_tokens")

    @property
    def sequence_lengths(self):
        return [len(protein) for protein in self.proteins]

    def to_dict(
        self, include_metadata: bool = True, convert_arrays_to_lists: bool = False
    ):
        protein_dict = {
            proteins_key: [] for proteins_key in self.protein_field_mapping.values()
        }
        for protein in self.proteins:
            for key in protein_dict.keys():
                protein_dict[self.protein_field_mapping[key]].append(
                    convert_list_of_arrays_to_list_of_lists(getattr(protein, key))
                    if convert_arrays_to_lists
                    else getattr(protein, key)
                )
        if include_metadata:
            protein_dict.update(self.metadata)
            if convert_arrays_to_lists:
                protein_dict["suffix_masks"] = self.suffix_masks.tolist()
        return protein_dict

    def clone(self, **kwargs):
        protein_document_dict = self.to_dict(include_metadata=True)
        protein_document_dict.update(kwargs)
        return ProteinDocument.from_dict(protein_document_dict)

    def null_fields(self):
        return {self.protein_field_mapping[f] for f in self.proteins[0].null_fields}

    def filter(self, filter_fn: Callable):
        """Filter by filter_fn.

        Filter_fn should take a protein and return True if it should be kept.
        """
        proteins = [protein for protein in self if filter_fn(protein)]
        return ProteinDocument(proteins=proteins, **self.metadata)

    def pop(self, index):
        return self.proteins.pop(index)

    @property
    def has_all_structure_arrays(self):
        # TODO: apply mapping
        assert not any(
            f in self.null_fields
            for f in ["backbone_coords", "structure_tokens", "plddts"]
        )

    def fill_missing_structure_arrays(
        self, coords_fill=np.nan, plddts_fill=np.nan, tokens_fill="?"
    ):
        assert isinstance(tokens_fill, str)
        return self.clone(
            plddts=self.plddts
            or [np.full(len(seq), plddts_fill) for seq in self.sequences],
            backbone_coords=self.backbone_coords
            or [np.full((len(seq), 4, 3), coords_fill) for seq in self.sequences],
            structure_tokens=self.structure_tokens
            or [tokens_fill * len(seq) for seq in self.sequences],
        )

    def extend(self, proteins: "ProteinDocument"):
        # n.b. extend may be a bad name as this is not in place
        constructor_kwargs = {}
        assert self.null_fields == proteins.null_fields
        for field in self.protein_field_mapping.values():
            if field not in self.null_fields:
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

    def interleave(self, sequence_separator: str = "|"):
        """Just copy each protein to form a new document containing prefix and suffix proteins.

        In practice, use transforms to mask different modalities from prefix and suffix.
        """
        return InterleavedProteinDocument(
            prefix_proteins=copy.deepcopy(self.proteins),
            suffix_proteins=copy.deepcopy(self.proteins),
            sequence_separator=sequence_separator,
        )


class InterleavedProteinDocument(BaseProteinDocument):
    """ProteinDocument where each protein is duplicated to form a prefix and suffix.

    Typical use will involve partial masking of sequence and structure in the prefix.
    The pretraining task will then be completion of the masked regions.

    Masking of the prefix is handled by transforms. By converting to InterleavedProteinDocument,
    standard transforms can operate independently on the prefix and suffix regions.

    self.suffix_mask can be used to control which regions are predicted.
    """

    default_fill_values = {
        "backbone_coords": np.nan,
        "backbone_coords_masks": 0.0,
        "plddts": 100.0,
    }
    metadata_fields = ProteinDocument.metadata_fields + ["sequence_separator"]

    def __init__(
        self,
        prefix_proteins: List[Protein],
        suffix_proteins: List[Protein],
        sequence_separator: str = "|",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_separator = sequence_separator
        self.prefix_proteins = prefix_proteins
        self.suffix_proteins = suffix_proteins

    def __iter__(self):
        for i in range(len(self)):
            yield self.prefix_proteins[i], self.suffix_proteins[i]

    def __getitem__(self, key):
        return self.prefix_proteins[key], self.suffix_proteins[key]

    @property
    def suffix_masks(self):
        """
        Boolean mask indicating suffix regions in concatenated arrays.
        """
        suffix_masks = []
        for prefix, suffix in self:
            suffix_masks.append(
                np.concatenate(
                    [
                        np.zeros(len(prefix.sequence) + 1, dtype=bool),
                        np.ones(len(suffix.sequence), dtype=bool),
                    ]
                )
            )
        return suffix_masks

    def interleave_proteins(self, prefix, suffix):
        return Protein(
            sequence=prefix.sequence + self.sequence_separator + suffix.sequence,
            accession=prefix.accession,
            positions=prefix.positions + [prefix.positions[-1] + 1] + suffix.positions,
            plddt=np.concatenate(
                [
                    prefix.plddt,
                    np.full((1,), self.default_fill_values["plddt"]),
                    suffix.plddt,
                ]
            ),
            backbone_coords=np.concatenate(
                [
                    prefix.backbone_coords,
                    np.full((1, 4, 3), self.default_fill_values["backbone_coords"]),
                    suffix.backbone_coords,
                ]
            ),
            backbone_coords_mask=np.concatenate(
                [
                    prefix.backbone_coords_mask,
                    np.full(
                        (1, 4, 3), self.default_fill_values["backbone_coords_mask"]
                    ),
                    suffix.backbone_coords_mask,
                ]
            ),
            structure_tokens=prefix.structure_tokens
            + self.sequence_separator
            + suffix.structure_tokens,
        )

    def to_protein_document(self):
        interleaved_proteins = []
        for prefix, suffix in self:
            interleaved_proteins.append(self.interleave_proteins(prefix, suffix))
        metadata = {
            field: getattr(self, field) for field in ProteinDocument.metadata_fields
        }
        return ProteinDocument(interleaved_proteins, **metadata)

    def to_dict(self, include_metadata: bool = True):
        # to convert to concatenated arrays - use self.to_protein_document().to_dict() instead
        prefix_dict = ProteinDocument(self.prefix_proteins).to_dict(
            include_metadata=False
        )
        suffix_dict = ProteinDocument(self.suffix_proteins).to_dict(
            include_metadata=False
        )
        interleaved_dict = {"prefix": prefix_dict, "suffix": suffix_dict}
        if include_metadata:
            interleaved_dict.update(self.metadata)
        return interleaved_dict

    # n.b. to clone, we clone independently
    def clone(
        self,
        prefix_proteins: List[Protein] = None,
        suffix_proteins: List[Protein] = None,
        **kwargs,
    ):
        return InterleavedProteinDocument(
            prefix_proteins=prefix_proteins or self.prefix_proteins,
            suffix_proteins=suffix_proteins or self.suffix_proteins,
            sequence_separator=self.sequence_separator,
            **kwargs,
        )
