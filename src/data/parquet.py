from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.datasets import ProteinDatasetConfig, StreamedProteinDatasetBuilder
from src.data.objects import Protein, ProteinDocument
from src.data.preprocessing import (
    ProteinDocumentPreprocessor,
    backbone_coords_from_example,
)
from src.data.utils import random_subsample
from src.utils.tokenizers import ProFamTokenizer


def build_representative_df(
    cluster_df, has_structure: bool = False, document_id_col="fam_id"
):
    """Assumes that the document id is the accession of the representative."""
    records = []
    for _, row in cluster_df.iterrows():
        rep_index = list(row["accessions"]).index(row[document_id_col])
        rep_dict = {
            "sequence": row["sequences"][rep_index],
            "accession": row["accessions"][rep_index],
            document_id_col: row[document_id_col],
            "length": len(row["sequences"][rep_index]),
        }
        if has_structure:
            rep_dict["plddt"] = (row["plddts"][rep_index],)
            rep_dict["N"] = (row["N"][rep_index],)
            rep_dict["CA"] = (row["CA"][rep_index],)
            rep_dict["C"] = (row["C"][rep_index],)
            rep_dict["O"] = (row["O"][rep_index],)
            rep_dict["mean_plddt"] = row["plddts"][rep_index].mean()
        records.append(rep_dict)
    return pd.DataFrame(records)


def export_protein_from_cluster_df(cluster_df, cluster_id, accession):
    row = cluster_df[(cluster_df["fam_id"] == cluster_id)].iloc[0]
    rep_index = list(row["accessions"]).index(accession)
    backbone_coords, _ = backbone_coords_from_example(row)[rep_index]
    protein = Protein(
        sequence=row["sequences"][rep_index],
        accession=row["accessions"][rep_index],
        plddt=row["plddts"][rep_index],
        backbone_coords=backbone_coords,
    )
    return protein


def export_pdb_from_cluster_df(cluster_df, cluster_id, accession, filepath):
    protein = export_protein_from_cluster_df(cluster_df, cluster_id, accession)
    protein.to_pdb_file(filepath)


def uniformly_sample_clusters(
    sequences, cluster_ids, max_total_length, tokens_per_sequence=1
):
    # Step 1: Group sequences by their attributes
    clusters = defaultdict(list)
    for ix, (seq, cl_id) in enumerate(zip(sequences, cluster_ids)):
        clusters[cl_id].append((ix, seq))

    selected_ids = []
    total_length = 0

    # Step 2: Sample the same number of items from each stratum
    while True:
        unique_cluster_ids = list(clusters.keys())
        cluster = np.random.choice(unique_cluster_ids)
        candidates = clusters[cluster]
        ix, seq = candidates.pop(np.random.choice(len(candidates)))

        if (
            total_length + len(seq) + tokens_per_sequence > max_total_length
            or not clusters
        ):
            break

        selected_ids.append(ix)
        total_length += len(seq) + tokens_per_sequence

        if not candidates:
            del clusters[cluster]

    return selected_ids


def filter_on_length(
    example,
    max_tokens,
    tokenizer,
    sequence_col="sequences",
    filter_type=None,
    interleave_structure_sequence=False,
):
    if filter_type is None:
        return True
    elif filter_type == "max_seq_pos":
        return any([len(s) <= tokenizer.max_seq_pos - 1 for s in example[sequence_col]])
    elif filter_type == "max_tokens":
        # relevant for e.g. non-batched processing for if where max_tokens could be lower than max_seq_pos
        if max_tokens is None:
            return True
        elif interleave_structure_sequence:
            return (
                max([len(s) for s in example[sequence_col]])
                <= (max_tokens // 2) - tokenizer.num_start_tokens - 2
            )
        else:
            return (
                max([len(s) for s in example[sequence_col]])
                <= max_tokens - tokenizer.num_start_tokens - 1
            )
    else:
        raise ValueError(f"Unknown length filter {filter_type}")


class ParquetDatasetBuilder(StreamedProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        batched_map: bool = False,
        map_batch_size: int = 100,
        identifier_col: str = "fam_id",
        sequence_col: str = "sequences",
        required_keys: Optional[List[str]] = None,
        max_sequences_per_document: Optional[int] = None,
        length_filter: Optional[str] = None,  # max_tokens, max_seq_pos
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            batched_map=batched_map,
            map_batch_size=map_batch_size,
            required_keys=required_keys,
            max_sequences_per_document=max_sequences_per_document,
        )
        if self.preprocessor is not None:
            self.interleave_structure_sequence = (
                self.preprocessor.interleave_structure_sequence
            )
        else:
            self.interleave_structure_sequence = False
        self.identifier_col = identifier_col
        self.sequence_col = sequence_col
        self.length_filter = length_filter

    def build_documents(
        self, examples, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        examples = super().build_documents(examples, max_tokens, shuffle)
        if self.identifier_col is not None:
            examples["identifier"] = examples[self.identifier_col]
        return examples

    def filter(
        self,
        example,
        min_sequences: Optional[int] = None,
        holdout_identifiers: Optional[List[str]] = None,
        tokenizer: ProFamTokenizer = None,
    ):
        super_filter = super().filter(
            example,
            min_sequences=min_sequences,
            holdout_identifiers=holdout_identifiers,
            tokenizer=tokenizer,
        )
        if super_filter:
            filter_num_seqs = len(example[self.sequence_col]) >= (min_sequences or 1)
            # TODO: we need to be very careful with this!
            filter_identifier = (
                holdout_identifiers is None
                or example[self.identifier_col] not in holdout_identifiers
            )
            length_filter = filter_on_length(
                example,
                filter_type=self.length_filter,
                max_tokens=None,
                tokenizer=tokenizer,
                sequence_col=self.sequence_col,
                interleave_structure_sequence=self.interleave_structure_sequence,
            )
            return filter_num_seqs and filter_identifier and length_filter
        return False


class ParquetSequenceDatasetBuilder(ParquetDatasetBuilder):
    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        batched_map: bool = False,
        map_batch_size: int = 100,
        identifier_col: str = "fam_id",
        sequence_col: str = "sequences",
        length_filter: Optional[str] = None,  # max_tokens, max_seq_pos
        infer_representative_from_identifier: bool = False,
        sample_uniformly_from_col: Optional[
            str
        ] = None,  # for redundancy-aware sampling
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            batched_map=batched_map,
            map_batch_size=map_batch_size,
            identifier_col=identifier_col,
            sequence_col=sequence_col,
            required_keys=[sequence_col],
            length_filter=length_filter,
        )
        self.infer_representative_from_identifier = infer_representative_from_identifier
        self.infer_representative_from_identifier = infer_representative_from_identifier
        self.sample_uniformly_from_col = sample_uniformly_from_col

    @staticmethod
    def build_document(
        example,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        sequence_col: str = "sequences",
        identifier_col: str = "fam_id",
        max_sequences: Optional[int] = None,
        infer_representative_from_identifier: bool = False,
    ):
        sequence_iterator = example[sequence_col]
        max_sequences_to_preprocess = (
            (max_tokens // 40) if max_sequences is None else max_sequences
        )
        # n.b. this also shuffles
        if shuffle:
            sequences = random_subsample(
                sequence_iterator,
                max_sequences_to_preprocess,
            )
        else:
            sequences = sequence_iterator[:max_sequences_to_preprocess]

        return ProteinDocument(
            sequences=sequences,
            representative_accession=example[identifier_col]
            if infer_representative_from_identifier
            else None,
            original_size=len(sequence_iterator),
        )

    def _build_document(
        self,
        example,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ):
        return self.build_document(
            example,
            max_tokens=max_tokens,
            shuffle=shuffle,
            max_sequences=self.max_sequences_per_document,
            sequence_col=self.sequence_col,
            identifier_col=self.identifier_col,
            infer_representative_from_identifier=self.infer_representative_from_identifier,
        )


class ParquetStructureDatasetBuilder(StreamedProteinDatasetBuilder):
    def __init__(
        self,
        name: str,
        cfg: ProteinDatasetConfig,
        preprocessor: Optional[ProteinDocumentPreprocessor] = None,
        batched_map: bool = False,
        map_batch_size: int = 100,
        sequence_col: str = "sequences",
        structure_tokens_col: Optional[str] = None,
        identifier_col: str = "fam_id",
        infer_representative_from_identifier: bool = False,
        minimum_mean_plddt: Optional[float] = None,
        length_filter: Optional[str] = None,  # max_tokens, max_seq_pos
        sample_uniformly_from_col: Optional[
            str
        ] = None,  # for redundancy-aware sampling
    ):
        super().__init__(
            name=name,
            cfg=cfg,
            preprocessor=preprocessor,
            batched_map=batched_map,
            map_batch_size=map_batch_size,
            required_keys=[sequence_col, structure_tokens_col]
            if structure_tokens_col is not None
            else [sequence_col],
        )
        self.sequence_col = sequence_col
        self.structure_tokens_col = structure_tokens_col
        self.identifier_col = identifier_col
        self.infer_representative_from_identifier = infer_representative_from_identifier
        self.sample_uniformly_from_col = sample_uniformly_from_col
        self.minimum_mean_plddt = minimum_mean_plddt
        self.length_filter = length_filter

    @staticmethod
    def build_document(
        example,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        max_sequences: Optional[int] = None,
        sample_uniformly_from_col: Optional[str] = None,
        structure_tokens_col: Optional[str] = None,
        sequence_col: str = "sequences",
        identifier_col: str = "fam_id",
        infer_representative_from_identifier: bool = False,
    ):
        # TODO: configure whether or not to use alignments, structure tokens col, etc.
        max_sequences_to_preprocess = (
            (max_tokens or 1e8) // 40 if max_sequences is None else max_sequences
        )
        if sample_uniformly_from_col is not None:
            assert shuffle
            sequence_ids = uniformly_sample_clusters(
                example[sequence_col],
                example[sample_uniformly_from_col],
                max_tokens - 3,
            )
        elif shuffle:
            sequence_ids = random_subsample(
                np.arange(len(example[sequence_col])),
                max_sequences_to_preprocess,
            )
        else:
            sequence_ids = np.arange(
                min(max_sequences_to_preprocess, len(example[sequence_col]))
            )
        sequences = [example[sequence_col][i] for i in sequence_ids]
        accessions = [example["accessions"][i] for i in sequence_ids]
        # we assume sequence processing and structure token processing are consistent.
        # later we will check that everything ends up the same length - which is important
        # because otherwise incorrect config could easily lead to misalignment
        if structure_tokens_col is not None:
            structure_tokens_iterator = example[structure_tokens_col]
            if structure_tokens_col == "msta_3di":
                # TODO: fix this; Hardcoded for now until we support aligning all representations
                structure_tokens = [
                    structure_tokens_iterator[i].replace("-", "").lower()
                    for i in sequence_ids
                ]
            else:
                structure_tokens = [
                    structure_tokens_iterator[i].lower() for i in sequence_ids
                ]
        else:
            # in fill missing values this gets set to mask, which in collate gets set to -100 in labels
            structure_tokens = None
        if "N" in example:
            coords, is_pdb = backbone_coords_from_example(
                example, sequence_col=sequence_col
            )
            coords = [coords[i] for i in sequence_ids]
            plddts = example["plddts"]
            plddts = [plddts[i] for i in sequence_ids]
        else:
            # TODO: support aligned coords, plddts
            coords = None
            plddts = None

        return ProteinDocument(
            sequences=sequences,
            accessions=accessions,
            plddts=plddts,
            backbone_coords=coords,
            structure_tokens=structure_tokens,
            representative_accession=example[identifier_col]
            if infer_representative_from_identifier
            else None,
            original_size=len(example["sequences"]),
        )

    def _build_document(
        self,
        example,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ):
        return self.build_document(
            example,
            max_tokens=max_tokens,
            shuffle=shuffle,
            max_sequences=self.max_sequences_per_document,
            sample_uniformly_from_col=self.sample_uniformly_from_col,
            structure_tokens_col=self.structure_tokens_col,
            sequence_col=self.sequence_col,
            identifier_col=self.identifier_col,
            infer_representative_from_identifier=self.infer_representative_from_identifier,
        )

    # TODO: write a test for this
    def filter(
        self,
        example,
        min_sequences: Optional[int] = None,
        holdout_identifiers: Optional[List[str]] = None,
        tokenizer: ProFamTokenizer = None,
    ):
        super_filter = super().filter(
            example,
            min_sequences=min_sequences,
            holdout_identifiers=holdout_identifiers,
            tokenizer=tokenizer,
        )
        if (
            self.structure_tokens_col is not None
            and example[self.structure_tokens_col] is None
        ):
            return False
        if super_filter and self.minimum_mean_plddt is not None:
            if "plddts" in example:
                mean_plddt = np.mean([np.mean(plddt) for plddt in example["plddts"]])
                filter_plddt = mean_plddt >= (self.cfg.minimum_mean_plddt or 0.0)
                return filter_plddt
            else:
                return True
        return super_filter
