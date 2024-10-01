import functools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.constants import BASEDIR
from src.data import transforms
from src.data.objects import ProteinDocument
from src.data.utils import examples_to_list_of_dicts
from src.sequence.fasta import convert_sequence_with_positions, read_fasta_sequences
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


def load_named_preprocessor(preprocessor_name, overrides: Optional[List[str]] = None):
    with initialize_config_dir(
        os.path.join(BASEDIR, "configs/preprocessor"), version_base="1.3"
    ):
        preprocessor_cfg = compose(config_name=preprocessor_name, overrides=overrides)
    return instantiate(preprocessor_cfg, _convert_="partial")


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


@dataclass
class PreprocessingConfig:
    keep_insertions: bool = False
    to_upper: bool = False
    keep_gaps: bool = False
    document_token: str = "[RAW]"
    truncate_after_n_sequences: Optional[int] = None
    use_msa_pos: bool = False  # for msa sequences, if true, position index will be relative to alignment cols
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/182
    allow_unk: bool = False


def subsample_fasta_lines(lines, n_lines, shuffle=True):
    start_ix = np.array([i for i, l in enumerate(lines) if l[0] == ">"])
    end_ix = start_ix[1:]
    end_ix = np.append(end_ix, len(lines))
    lines_per_seq = len(lines) // len(start_ix)
    n_samples = min(n_lines // lines_per_seq, len(start_ix))
    if shuffle:
        sample_indices = np.random.choice(len(start_ix), n_samples, replace=False)
    else:
        sample_indices = np.arange(n_samples)
    starts = start_ix[sample_indices]
    ends = end_ix[sample_indices]
    sampled_lines = []
    for start, end in zip(starts, ends):
        assert lines[end - 1][0] != ">"
        sampled_lines.extend(lines[start:end])
    return sampled_lines


def random_subsample(arr, n, seed: Optional[int] = None):
    rnd = np_random(seed)
    return rnd.choice(arr, min(n, len(arr)), replace=False)


def default_transforms(
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    return [
        functools.partial(
            transforms.sample_to_max_tokens,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
        ),
        transforms.fill_missing_fields,
        transforms.replace_selenocysteine_pyrrolysine,
    ]


def preprocess_protein_sequences(
    proteins: ProteinDocument,
    cfg: PreprocessingConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    transform_fns: Optional[List[Callable]] = None,
):
    assert isinstance(proteins, ProteinDocument), type(proteins)
    transform_fns = transform_fns or []
    # TODO: assert that structure tokens, coords, plddt are all same shape as sequences post conversion or handle if not
    if tokenizer.use_seq_pos:
        proteins = transforms.convert_sequences_adding_positions(
            proteins,
            keep_gaps=cfg.keep_gaps,
            keep_insertions=cfg.keep_insertions,
            to_upper=cfg.to_upper,
            use_msa_pos=cfg.use_msa_pos,
            truncate_after_n_sequences=cfg.truncate_after_n_sequences,
        )
    else:
        proteins = proteins[: cfg.truncate_after_n_sequences or len(proteins)]
    proteins = transforms.apply_transforms(
        transform_fns, proteins, tokenizer, max_tokens=max_tokens
    )
    return proteins


def backbone_coords_from_example(
    example,
    selected_ids: Optional[List[int]] = None,
    sequence_col="sequences",
    use_pdb_if_available_prob: float = 0.0,
):
    ns = example["N"]
    cas = example["CA"]
    cs = example["C"]
    oxys = example["O"]
    sequences = example[sequence_col]
    prot_has_pdb = (
        example["pdb_index_mask"] if "pdb_index_mask" in example else [False] * len(ns)
    )
    coords = []
    is_pdb = []
    if selected_ids is None:
        selected_ids = range(len(ns))

    for ix in selected_ids:
        seq = sequences[ix]
        has_pdb = prot_has_pdb[ix]
        use_pdb = has_pdb and np_random().rand() < use_pdb_if_available_prob

        if use_pdb:
            # I guess test that this is working is that lengths line up
            pdb_index = list(np.argwhere(example["extra_pdb_mask"]).reshape(-1)).index(
                ix
            )
            n = example["pdb_N"][pdb_index]
            ca = example["pdb_CA"][pdb_index]
            c = example["pdb_C"][pdb_index]
            o = example["pdb_O"][pdb_index]
            is_pdb.append(True)
        else:
            n = ns[ix]
            ca = cas[ix]
            c = cs[ix]
            o = oxys[ix]
            is_pdb.append(False)

        recons_coords = np.zeros((len(seq), 4, 3))
        recons_coords[:, 0] = np.array(n).reshape(-1, 3)
        recons_coords[:, 1] = np.array(ca).reshape(-1, 3)
        recons_coords[:, 2] = np.array(c).reshape(-1, 3)
        recons_coords[:, 3] = np.array(o).reshape(-1, 3)
        coords.append(recons_coords)

    return coords, is_pdb


class BasePreprocessor:
    def __init__(
        self,
        config: PreprocessingConfig,  # configures preprocessing of individual proteins
        transform_fns: Optional[List[Callable]] = None,
        interleave_structure_sequence: bool = False,
        batched_map: bool = False,  # should map be called with batched=True
        map_batch_size: int = 100,
        sample_uniformly_from_col: Optional[
            str
        ] = None,  # for redundancy-aware sampling
        max_sequences_per_document: Optional[int] = None,
    ):
        self.cfg = config
        self.transform_fns = transform_fns
        self.interleave_structure_sequence = (
            interleave_structure_sequence  # should this be part of config?
        )
        self.batched_map = batched_map
        self.map_batch_size = map_batch_size
        self.sample_uniformly_from_col = sample_uniformly_from_col
        self.max_sequences = max_sequences_per_document
        if self.sample_uniformly_from_col is not None:
            # instead of sampling sequences uniformly, we sample from unique values in this column
            # then sample within those values uniformly to build a batch.
            raise NotImplementedError()

    def build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        raise NotImplementedError()

    def _batched_preprocess_protein_data(
        self,
        examples: Dict[str, List[Any]],
        tokenizer: ProFamTokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        """
        a batched map is an instruction for converting a set of examples to a
        new set of examples (not necessarily of the same size). it should return a dict whose
        values are lists, where the length of the lists determines the size of the new set of examples.
        """
        # TODO: remove max tokens from build document?
        proteins_list = self.build_documents(
            examples, tokenizer, max_tokens=max_tokens, shuffle=shuffle
        )
        transform_fns = default_transforms(max_tokens=max_tokens, shuffle=shuffle)
        transform_fns += self.transform_fns or []
        proteins_list = [
            preprocess_protein_sequences(
                proteins,
                self.cfg,
                tokenizer,
                max_tokens=max_tokens,
                transform_fns=transform_fns,
            )
            for proteins in proteins_list
        ]
        return tokenizer.batched_encode(
            proteins_list,
            document_token=self.cfg.document_token,
            padding="max_length",
            max_length=max_tokens,
            add_final_sep=True,
            allow_unk=getattr(self.cfg, "allow_unk", False),
        )

    def _preprocess_protein_data(
        self,
        example: Dict[str, Any],
        tokenizer: ProFamTokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        # N.B. for stockholm format we need to check that sequences aren't split over
        # multiple lines
        proteins = self.build_document(example, max_tokens=max_tokens, shuffle=shuffle)
        transform_fns = default_transforms(max_tokens=max_tokens, shuffle=shuffle)
        transform_fns += self.transform_fns or []
        proteins = preprocess_protein_sequences(
            proteins,
            self.cfg,
            tokenizer,
            max_tokens=max_tokens,
            transform_fns=transform_fns,
        )
        tokenized = tokenizer.encode(
            proteins,
            document_token=self.cfg.document_token,
            padding="max_length",
            max_length=max_tokens,
            add_final_sep=True,
            allow_unk=getattr(self.cfg, "allow_unk", False),
        )
        if max_tokens is not None:
            assert tokenized.input_ids.shape[-1] <= max_tokens, (
                tokenized.input_ids.shape[-1],
                max_tokens,
            )
        return tokenized.data

    def preprocess_protein_data(
        self,
        examples: Dict[str, Any],
        tokenizer: ProFamTokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        if self.batched_map:
            return self._batched_preprocess_protein_data(
                examples, tokenizer, max_tokens=max_tokens, shuffle=shuffle
            )
        else:
            return self._preprocess_protein_data(
                examples, tokenizer, max_tokens=max_tokens, shuffle=shuffle
            )

    def build_documents(
        self,
        examples,
        tokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ):
        """We assume that documents should be concatenated up to max_tokens.

        TODO: implement document-aware attention masking
        """
        example_dicts = examples_to_list_of_dicts(examples)
        proteins_list = [
            self.build_document(example_dict, max_tokens=max_tokens, shuffle=shuffle)
            for example_dict in example_dicts
        ]
        document_lengths = [
            sum(proteins.sequence_lengths) for proteins in proteins_list
        ]
        merged_documents = []
        current_document = None
        total_sequence_length = 0
        for proteins, length in zip(proteins_list, document_lengths):
            if current_document is None:
                current_document = proteins.clone()
            else:
                if sum(current_document.sequence_lengths) + length <= (
                    max_tokens or 1e8
                ):
                    current_document = current_document.extend(proteins)
                    total_sequence_length += sum(proteins.sequence_lengths)
                else:
                    merged_documents.append(current_document)
                    current_document = proteins.clone()
                    total_sequence_length = sum(current_document.sequence_lengths)
        if current_document is not None:
            merged_documents.append(current_document)

        return merged_documents


class FastaPreprocessor(BasePreprocessor):
    @property
    def required_keys(self):
        return ["text"]

    def build_document_from_text(
        self, text, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        lines = text.split("\n")
        if not len(lines[-1]):
            lines = lines[:-1]
        # rough upper bound: min 2 lines per seq, assume at least 10 tks per line
        max_fasta_lines_to_preprocess = (
            (max_tokens or 1e8) // 5
            if self.max_sequences is None
            else self.max_sequences * 50
        )
        if len(lines) > max_fasta_lines_to_preprocess:
            lines = subsample_fasta_lines(
                lines,
                max_fasta_lines_to_preprocess,
                shuffle=shuffle,
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
            sequences=sequences, original_size=len(lines) // 2
        )  # upper bound estimate of number of sequences

    def build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        if isinstance(example, str):
            return self.build_document_from_text(example, max_tokens, shuffle)
        else:
            return self.build_document_from_text(example["text"], max_tokens, shuffle)


class ParquetSequencePreprocessor(BasePreprocessor):
    def __init__(
        self,
        config: PreprocessingConfig,
        sequence_col: str = "sequences",
        identifier_col: str = "fam_id",
        transform_fns: Optional[List[Callable]] = None,
        infer_representative_from_identifier: bool = False,
        batched_map: bool = False,  # should map be called with batched=True
        map_batch_size: int = 100,
        sample_uniformly_from_col: Optional[
            str
        ] = None,  # for redundancy-aware sampling
        max_sequences_per_document: Optional[int] = None,
    ):
        super().__init__(
            config,
            transform_fns,
            interleave_structure_sequence=False,
            batched_map=batched_map,
            map_batch_size=map_batch_size,
            sample_uniformly_from_col=sample_uniformly_from_col,
        )
        self.sequence_col = sequence_col
        self.identifier_col = identifier_col
        self.infer_representative_from_identifier = infer_representative_from_identifier

    @property
    def required_keys(self):
        return [self.sequence_col]

    def build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        sequence_iterator = example[self.sequence_col]
        max_sequences_to_preprocess = (
            (max_tokens // 40) if self.max_sequences is None else self.max_sequences
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
            representative_accession=example[self.identifier_col]
            if self.infer_representative_from_identifier
            else None,
            original_size=len(sequence_iterator),
        )


# TODO: make sure we can handle an aligned version - test
class ParquetStructurePreprocessor(BasePreprocessor):
    def __init__(
        self,
        config: PreprocessingConfig,
        sequence_col: str = "sequences",
        structure_tokens_col: Optional[str] = None,
        interleave_structure_sequence: bool = False,
        structure_first_prob: float = 1.0,
        identifier_col: str = "fam_id",
        prefer_pdb_if_available_prob: float = 0.0,
        infer_representative_from_identifier: bool = False,
        transform_fns: Optional[List[Callable]] = None,
        batched_map: bool = False,  # should map be called with batched=True
        map_batch_size: int = 100,
        upsample_pdb: bool = False,
        sample_uniformly_from_col: Optional[
            str
        ] = None,  # for redundancy-aware sampling
        max_sequences_per_document: Optional[int] = None,
    ):
        if interleave_structure_sequence:
            # handle like this because useful to have an interleave_structure_sequence attribute for lenght filtering
            transform_fns = transform_fns or []
            transform_fns.append(
                functools.partial(
                    transforms.interleave_structure_sequence,
                    structure_first_prob=structure_first_prob,
                )
            )
        super().__init__(
            config,
            transform_fns,
            interleave_structure_sequence=interleave_structure_sequence,
            batched_map=batched_map,
            map_batch_size=map_batch_size,
            sample_uniformly_from_col=sample_uniformly_from_col,
            max_sequences_per_document=max_sequences_per_document,
        )
        # Two options for pdb handling:
        # 1. include pdbs as explicit things in document
        # 2. just choose either pdb or af structure when building document
        self.prefer_pdb_if_available_prob = prefer_pdb_if_available_prob
        self.sequence_col = sequence_col
        self.structure_tokens_col = structure_tokens_col
        self.identifier_col = identifier_col
        self.infer_representative_from_identifier = infer_representative_from_identifier
        self.upsample_pdb = upsample_pdb

    @property
    def required_keys(self):
        if self.structure_tokens_col is None:
            return [self.sequence_col]
        return [self.sequence_col, self.structure_tokens_col]

    def build_document(
        self, example, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        # TODO: configure whether or not to use alignments, structure tokens col, PDB, etc.
        has_pdb = "pdb_ids" in example
        max_sequences_to_preprocess = (
            (max_tokens or 1e8) // 40
            if self.max_sequences is None
            else self.max_sequences
        )
        if self.sample_uniformly_from_col is not None:
            assert shuffle
            sequence_ids = uniformly_sample_clusters(
                example["sequences"],
                example[self.sample_uniformly_from_col],
                max_tokens - 3,
            )
        elif shuffle:
            sequence_ids = random_subsample(
                np.arange(len(example["sequences"])),
                max_sequences_to_preprocess,
            )
        else:
            sequence_ids = np.arange(
                min(max_sequences_to_preprocess, len(example["sequences"]))
            )

        if has_pdb and self.upsample_pdb:
            extra_pdb_ids = [
                ix
                for ix in np.argwhere(example["pdb_index_mask"])
                if ix not in sequence_ids
            ]
            if shuffle:
                extra_pdb_ids = list(np.random.shuffle(extra_pdb_ids))

            sequence_ids = (
                sequence_ids
                + extra_pdb_ids[: min(len(extra_pdb_ids), len(sequence_ids))]
            )  # at most double the number of sequences to preprocess
        sequences = [example["sequences"][i] for i in sequence_ids]
        accessions = [example["accessions"][i] for i in sequence_ids]

        # we assume sequence processing and structure token processing are consistent.
        # later we will check that everything ends up the same length - which is important
        # because otherwise incorrect config could easily lead to misalignment

        # CAREFUL: STRUCTURE TOKENS ARE NOT DERIVED FROM PDB WHERE AVAILABLE
        if self.structure_tokens_col is not None:
            structure_tokens_iterator = example[self.structure_tokens_col]
            structure_tokens = [
                convert_sequence_with_positions(
                    structure_tokens_iterator[i],
                    keep_gaps=self.cfg.keep_gaps,
                    keep_insertions=self.cfg.keep_insertions,
                    to_upper=self.cfg.to_upper,
                )[0].lower()
                for i in sequence_ids
            ]
        else:
            # in fill missing values this gets set to mask, which in collate gets set to -100 in labels
            structure_tokens = None

        if "N" in example and not self.cfg.keep_gaps:
            assert not any(["-" in seq for seq in sequences])
            if structure_tokens is not None:
                assert not any(["-" in seq for seq in structure_tokens])
            coords, struct_is_pdb = backbone_coords_from_example(
                example,
                selected_ids=sequence_ids,
                sequence_col=self.sequence_col,
                use_pdb_if_available_prob=self.prefer_pdb_if_available_prob,
            )
            # TODO: same thing for structure tokens?
            plddts = []
            for is_pdb, ix in zip(struct_is_pdb, sequence_ids):
                if is_pdb:
                    coords_mask = np.isnan(coords[ix]).any(axis=-1)
                    plddt = np.full_like(example["plddts"][ix], 100)
                    plddt[coords_mask] = 0
                    plddts.append(plddt)
                else:
                    plddts.append(example["plddts"][ix])

        else:
            # TODO: support aligned coords, plddts
            coords = None
            plddts = None

        return ProteinDocument(
            sequences=sequences,
            accessions=accessions,
            plddts=plddts,
            backbone_coords=coords,
            struct_is_pdb=struct_is_pdb,
            structure_tokens=structure_tokens,
            representative_accession=example[self.identifier_col]
            if self.infer_representative_from_identifier
            else None,
            original_size=len(example["sequences"]),
        )
