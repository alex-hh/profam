from typing import Dict, Optional

import torch

from src.constants import RESIDUE_LEVEL_FEATURES
from src.data.objects import ProteinDocument
from src.data.preprocessing import (
    BasePreprocessorConfig,
    preprocess_protein_sequences,
    subsample_and_tokenize_protein_data,
)
from src.models.base import BaseFamilyLitModule
from src.utils.tokenizers import ProFamTokenizer


class PromptBuilder:
    def __init__(
        self,
        preprocessor: BasePreprocessorConfig,
        max_tokens: int,
        seed: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        self.seed = seed
        self.max_tokens = max_tokens

    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        proteins = preprocess_protein_sequences(proteins, self.preprocessor, tokenizer)
        max_length = max(len(seq) for seq in proteins.sequences)
        batch = subsample_and_tokenize_protein_data(
            proteins,
            cfg=self.preprocessor,
            tokenizer=tokenizer,
            shuffle=True,
            seed=self.seed,
            padding="longest",
            max_tokens=self.max_tokens - max_length,
        )  # a dictionary
        return batch


class InterleavedInverseFoldingPromptBuilder(PromptBuilder):
    """Prompt builder for interleaved inverse folding tasks.

    Instead of finishing with a sep, we finish with a structure sequence sep
    We also know the ground truth sequence: the seed sequence in the protein
    document: ProteinDocument.seed
    """

    def __init__(
        self,
        preprocessor: BasePreprocessorConfig,
        max_tokens: int,
        seed: Optional[int] = None,
        representative_only: bool = False,
    ):
        super().__init__(preprocessor, max_tokens, seed)
        assert self.preprocessor.interleave_structure_sequence
        self.representative_only = representative_only

    # we need to exclude token space for length seed*2 from preprocessing
    # TODO: write tests for this
    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        representative = proteins.pop_representative()
        proteins = preprocess_protein_sequences(proteins, self.preprocessor, tokenizer)
        if not self.representative_only:
            batch = subsample_and_tokenize_protein_data(
                proteins,
                cfg=self.preprocessor,
                tokenizer=tokenizer,
                shuffle=True,
                seed=self.seed,
                padding="longest",
                max_tokens=self.max_tokens - len(representative),
                exclude_tokens=2 * len(proteins.seed),
            )
        # TODO: tokenize representative
        representative_doc = ProteinDocument.from_proteins([representative])
        representative_doc = preprocess_protein_sequences(
            representative_doc, self.preprocessor, tokenizer
        )
        representative_batch = subsample_and_tokenize_protein_data(
            representative_doc,
            cfg=self.preprocessor,
            tokenizer=tokenizer,
            shuffle=True,
            seed=self.seed,
            padding="longest",
            max_tokens=len(proteins.seed) + 3,  # TODO: fix?
        )
        seq_start = (
            torch.argwhere(
                batch["input_ids"][0] == tokenizer.seq_struct_sep_token_id
            ).min()
            + 1
        )
        print("SEQ START", seq_start)
        assert batch["input_ids"][0, seq_start - 1] == tokenizer.seq_struct_sep_token_id
        for feat in representative_batch.keys():
            if feat in RESIDUE_LEVEL_FEATURES:
                representative_batch[feat] = representative_batch[feat][:, :seq_start]
            else:
                print("NOT RESIDUE LEVEL", feat)

        if self.representative_only:
            return representative_batch
        else:
            for feat in representative_batch.keys():
                if feat in RESIDUE_LEVEL_FEATURES:
                    batch[feat] = torch.cat(
                        (batch[feat], representative_batch[feat]), dim=1
                    )
                else:
                    print("NOT RESIDUE LEVEL", feat)
            return batch


class ProFamSampler:
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: PromptBuilder,
        sampling_kwargs: Optional[Dict] = None,
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        self.sampling_kwargs = sampling_kwargs

    def to(self, device):
        self.model.to(device)

    def sample_seqs(self, protein_document: ProteinDocument, num_samples: int):
        prompt = self.prompt_builder(protein_document, self.model.tokenizer)
        with torch.no_grad():  # prob unnecessary
            tokens = self.model._sample_seqs(
                prompt["input_ids"].unsqueeze(0).to(self.model.device),
                num_samples=num_samples,
                input_seq_pos=prompt["seq_pos"].unsqueeze(0).to(self.model.device),
                **self.sampling_kwargs
            )
            return self.model.tokenizer.decode_tokens(tokens)
