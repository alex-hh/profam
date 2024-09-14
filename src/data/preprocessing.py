import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.data import transforms
from src.data.fasta import convert_sequence_with_positions, read_fasta_sequences
from src.data.objects import ProteinDocument
from src.data.utils import examples_to_list_of_dicts
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


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
    batched_map: bool = False  # should map be called with batched=True
    map_batch_size: int = 100


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


def subsample_and_tokenize_protein_data(
    proteins: ProteinDocument,
    cfg: PreprocessingConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    padding: str = "max_length",
    shuffle: bool = True,
    seed: Optional[int] = None,
    transform_fns: Optional[List[Callable]] = None,
):
    proteins = transforms.sample_to_max_tokens(
        proteins,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=seed,
    )
    # if cfg.fill_missing_fields:
    proteins = transforms.fill_missing_fields(proteins)
    proteins = transforms.replace_selenocysteine_pyrrolysine(proteins)
    proteins = transforms.apply_transforms(transform_fns, proteins, tokenizer)

    tokenized = tokenizer.encode(
        proteins,
        document_token=cfg.document_token,
        padding=padding,
        max_length=max_tokens,
        add_final_sep=True,
        allow_unk=getattr(cfg, "allow_unk", False),
    )
    # tokenized.input_ids is flat now

    return tokenized.data


def batched_subsample_and_tokenize_protein_data(
    proteins_list: List[ProteinDocument],
    cfg: PreprocessingConfig,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    padding: str = "max_length",
    shuffle: bool = True,
    seed: Optional[int] = None,
    transform_fns: Optional[List[Callable]] = None,
):
    # N.B. right now this is equivalent to just looping over subsample_and_tokenize_protein_data
    # but in the future we might want to do something more sophisticated
    new_proteins_list = []
    for proteins in proteins_list:
        proteins = transforms.sample_to_max_tokens(
            proteins,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
        )
        # if cfg.fill_missing_fields:
        proteins = transforms.fill_missing_fields(proteins)
        proteins = transforms.replace_selenocysteine_pyrrolysine(proteins)
        proteins = transforms.apply_transforms(transform_fns, proteins, tokenizer)
        new_proteins_list.append(proteins)

    return tokenizer.batched_encode(
        new_proteins_list,
        document_token=cfg.document_token,
        padding=padding,
        max_length=max_tokens,
        add_final_sep=True,
        allow_unk=getattr(cfg, "allow_unk", False),
    )


def preprocess_protein_sequences(
    proteins: ProteinDocument,
    cfg: PreprocessingConfig,
    tokenizer: ProFamTokenizer,
):
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
    return proteins


def backbone_coords_from_example(example, sequence_col="sequences"):
    ns = example["N"]
    cas = example["CA"]
    cs = example["C"]
    oxys = example["O"]
    coords = []
    for seq, n, ca, c, o in zip(
        example[sequence_col],
        ns,
        cas,
        cs,
        oxys,
    ):
        recons_coords = np.zeros((len(seq), 4, 3))
        recons_coords[:, 0] = np.array(n).reshape(-1, 3)
        recons_coords[:, 1] = np.array(ca).reshape(-1, 3)
        recons_coords[:, 2] = np.array(c).reshape(-1, 3)
        recons_coords[:, 3] = np.array(o).reshape(-1, 3)
        coords.append(recons_coords)
    return coords


class BasePreprocessor:
    def __init__(
        self,
        config: PreprocessingConfig,
        transform_fns: Optional[List[Callable]] = None,
        interleave_structure_sequence: bool = False,
    ):
        self.cfg = config
        self.transform_fns = transform_fns
        self.interleave_structure_sequence = interleave_structure_sequence

    def build_document(
        self, example, tokenizer, max_tokens: Optional[int] = None, shuffle: bool = True
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
        proteins_list = self.build_documents(
            examples, tokenizer, max_tokens=max_tokens, shuffle=shuffle
        )
        proteins_list = [
            preprocess_protein_sequences(proteins, self.cfg, tokenizer)
            for proteins in proteins_list
        ]
        return batched_subsample_and_tokenize_protein_data(
            proteins_list,
            self.cfg,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            shuffle=shuffle,
            transform_fns=self.transform_fns,
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
        proteins = self.build_document(
            example, tokenizer, max_tokens=max_tokens, shuffle=shuffle
        )
        proteins = preprocess_protein_sequences(proteins, self.cfg, tokenizer)
        return subsample_and_tokenize_protein_data(
            proteins,
            self.cfg,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            shuffle=shuffle,
            transform_fns=self.transform_fns,
        )

    def preprocess_protein_data(
        self,
        examples: Dict[str, Any],
        tokenizer: ProFamTokenizer,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        if self.cfg.batched_map:
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
            self.build_document(
                example_dict, tokenizer, max_tokens=max_tokens, shuffle=shuffle
            )
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
        self, text, tokenizer, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        lines = text.split("\n")
        if not len(lines[-1]):
            lines = lines[:-1]
        # min 2 lines per seq, assume at least 10 tks per line
        max_fasta_lines_to_preprocess = (
            max_tokens or 1e8
        ) // 5  # upper bound on lines to proc.
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
                # preserve original sequences before getting positions
                keep_gaps=True if tokenizer.use_seq_pos else self.cfg.keep_gaps,
                keep_insertions=True
                if tokenizer.use_seq_pos
                else self.cfg.keep_insertions,
                to_upper=False if tokenizer.use_seq_pos else self.cfg.to_upper,
            )
        ]
        return ProteinDocument(
            sequences=sequences, original_size=len(lines) // 2
        )  # upper bound estimate of number of sequences

    def build_document(
        self, example, tokenizer, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        if isinstance(example, str):
            return self.build_document_from_text(
                example, tokenizer, max_tokens, shuffle
            )
        else:
            return self.build_document_from_text(
                example["text"], tokenizer, max_tokens, shuffle
            )


class ParquetSequencePreprocessor(BasePreprocessor):
    def __init__(
        self,
        config: PreprocessingConfig,
        sequence_col: str = "sequences",
        transform_fns: Optional[List[Callable]] = None,
    ):
        super().__init__(config, transform_fns)
        self.sequence_col = sequence_col

    @property
    def required_keys(self):
        return [self.sequence_col]

    def build_document(
        self, example, tokenizer, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        sequence_iterator = example[self.sequence_col]
        max_sequences_to_preprocess = max_tokens // 10
        # n.b. this also shuffles
        if shuffle:
            sequences = random_subsample(
                sequence_iterator,
                max_sequences_to_preprocess,
            )
        else:
            sequences = sequence_iterator[:max_sequences_to_preprocess]

        return ProteinDocument(
            sequences=sequences, original_size=len(sequence_iterator)
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
        infer_representative_from_identifier: bool = False,
        transform_fns: Optional[List[Callable]] = None,
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
        )
        self.sequence_col = sequence_col
        self.structure_tokens_col = structure_tokens_col
        self.identifier_col = identifier_col
        self.infer_representative_from_identifier = infer_representative_from_identifier

    @property
    def required_keys(self):
        if self.structure_tokens_col is None:
            return [self.sequence_col]
        return [self.sequence_col, self.structure_tokens_col]

    def build_document(
        self, example, tokenizer, max_tokens: Optional[int] = None, shuffle: bool = True
    ):
        # TODO: configure whether or not to use alignments, structure tokens col, etc.
        max_sequences_to_preprocess = (max_tokens or 1e8) // 10
        if shuffle:
            sequence_ids = random_subsample(
                np.arange(len(example["sequences"])),
                max_sequences_to_preprocess,
            )
        else:
            sequence_ids = np.arange(
                min(max_sequences_to_preprocess, len(example["sequences"]))
            )
        sequences = [example["sequences"][i] for i in sequence_ids]
        accessions = [example["accessions"][i] for i in sequence_ids]
        # we assume sequence processing and structure token processing are consistent.
        # later we will check that everything ends up the same length - which is important
        # because otherwise incorrect config could easily lead to misalignment
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
            coords = backbone_coords_from_example(
                example, sequence_col=self.sequence_col
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
            representative_accession=example[self.identifier_col]
            if self.infer_representative_from_identifier
            else None,
            original_size=len(example["sequences"]),
        )
